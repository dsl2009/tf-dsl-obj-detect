import tensorflow as tf
import config
import math
from itertools import product as product
import numpy as np
import random
import cv2
from skimage import transform
def normalize_to_target(inputs,
                        target_norm_value,
                        dim,
                        epsilon=1e-7,
                        trainable=True,
                        scope='NormalizeToTarget',
                        summarize=True):
  """L2 normalizes the inputs across the specified dimension to a target norm.

  This op implements the L2 Normalization layer introduced in
  Liu, Wei, et al. "SSD: Single Shot MultiBox Detector."
  and Liu, Wei, Andrew Rabinovich, and Alexander C. Berg.
  "Parsenet: Looking wider to see better." and is useful for bringing
  activations from multiple layers in a convnet to a standard scale.

  Note that the rank of `inputs` must be known and the dimension to which
  normalization is to be applied should be statically defined.

  TODO: Add option to scale by L2 norm of the entire input.

  Args:
    inputs: A `Tensor` of arbitrary size.
    target_norm_value: A float value that specifies an initial target norm or
      a list of floats (whose length must be equal to the depth along the
      dimension to be normalized) specifying a per-dimension multiplier
      after normalization.
    dim: The dimension along which the input is normalized.
    epsilon: A small value to add to the inputs to avoid dividing by zero.
    trainable: Whether the norm is trainable or not
    scope: Optional scope for variable_scope.
    summarize: Whether or not to add a tensorflow summary for the op.

  Returns:
    The input tensor normalized to the specified target norm.

  Raises:
    ValueError: If dim is smaller than the number of dimensions in 'inputs'.
    ValueError: If target_norm_value is not a float or a list of floats with
      length equal to the depth along the dimension to be normalized.
  """
  with tf.variable_scope(scope, 'NormalizeToTarget', [inputs]):
    if not inputs.get_shape():
      raise ValueError('The input rank must be known.')
    input_shape = inputs.get_shape().as_list()
    input_rank = len(input_shape)
    if dim < 0 or dim >= input_rank:
      raise ValueError(
          'dim must be non-negative but smaller than the input rank.')
    if not input_shape[dim]:
      raise ValueError('input shape should be statically defined along '
                       'the specified dimension.')
    depth = input_shape[dim]
    if not (isinstance(target_norm_value, float) or
            (isinstance(target_norm_value, list) and
             len(target_norm_value) == depth) and
            all([isinstance(val, float) for val in target_norm_value])):
      raise ValueError('target_norm_value must be a float or a list of floats '
                       'with length equal to the depth along the dimension to '
                       'be normalized.')
    if isinstance(target_norm_value, float):
      initial_norm = depth * [target_norm_value]
    else:
      initial_norm = target_norm_value
    target_norm = tf.contrib.framework.model_variable(
        name='weights', dtype=tf.float32,
        initializer=tf.constant(initial_norm, dtype=tf.float32),
        trainable=trainable)
    if summarize:
      mean = tf.reduce_mean(target_norm)
      mean = tf.Print(mean, ['NormalizeToTarget:', mean])
      tf.summary.scalar(tf.get_variable_scope().name, mean)
    lengths = epsilon + tf.sqrt(tf.reduce_sum(tf.square(inputs), dim, True))
    mult_shape = input_rank*[1]
    mult_shape[dim] = depth
    return tf.reshape(target_norm, mult_shape) * tf.truediv(inputs, lengths)


def get_prio_box(cfg = config.voc_vgg_300):
    mean = []
    for k, f in enumerate(cfg['feature_maps']):
        for i, j in product(range(f), repeat=2):
            f_k = cfg['min_dim'] / cfg['steps'][k]
            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = cfg['min_sizes'][k] / cfg['min_dim']
            mean.append([cx, cy, s_k, s_k])

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (cfg['max_sizes'][k] / cfg['min_dim']))
            mean.append([cx, cy, s_k_prime, s_k_prime])

            # rest of aspect ratios
            for ar in cfg['aspect_ratios'][k]:
                mean.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                mean.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])
                # back to torch land
    out = np.asarray(mean,dtype=np.float32)

    out = np.clip(out,a_min=0.0,a_max=1.0)
    return tf.constant(value=out,dtype=tf.float32)




def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps

def trim_zeros_graph(boxes):

    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros)
    return boxes, non_zeros

def encode_box(matched,prios,variance=None):
    '''
    :param matched: 匹配到的gtbox
    :param prios:  预选框
    :param variance: 偏差值
    :return: 
    '''
    if variance is None:
        variance =[0.1, 0.2]
    g_cxcy = (matched[:,:2]+matched[:,2:])/2 - prios[:,:2]
    g_cxcy = g_cxcy/(prios[:,:2]*variance[0])

    g_wh = (matched[:,2:]-matched[:,:2])/prios[:,2:]
    g_wh = tf.log(g_wh)/variance[1]

    return tf.concat([g_cxcy,g_wh],1)

def decode_box(prios,pred_loc,variance=None):
    if variance is None:
        variance =[0.1, 0.2]
    boxes = tf.concat((
        prios[:, :2] + pred_loc[:, :2] * variance[0] * prios[:, 2:],
        prios[:, 2:] * tf.exp(pred_loc[:, 2:] * variance[1])), 1)
    xy_min = boxes[:, :2] -boxes[:, 2:] / 2
    xy_max = boxes[:, 2:] + xy_min
    return  tf.concat([xy_min,xy_max],axis=1)


def batch_slice(inputs, graph_fn, batch_size):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))


    result = [tf.stack(o, axis=0)
              for o in outputs]
    if len(result) == 1:
        result = result[0]

    return result

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop



    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)

        scale = max_dim / image_max

    # Resize image using bilinear interpolation

    if scale != 1:
        image = cv2.resize(image, dsize=(round(w * scale), round(h * scale)),interpolation=cv2.INTER_AREA)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def revert_image(scale,padding,image_size,box):

    box = box*image_size

    box[:, 0] = box[:, 0] - padding[1][0]
    box[:, 1] = box[:, 1] - padding[0][0]
    box[:, 2] = box[:, 2] - padding[1][1]
    box[:, 3] = box[:, 3] - padding[0][1]
    box = box/scale
    box = np.asarray(box,dtype=np.int32)
    return box


def count_parm():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)
def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        x1, y1, x2, y2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = transform.resize(m, mini_shape, order=1, mode="constant")
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask