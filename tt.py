from torch import nn
import torch
import numpy as np
import tensorflow as tf
import utils
tf.enable_eager_execution()

def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def trim_zeros_graph(boxes, name=None):

    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros



def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    print(x_max)
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

a = [[1,3,2,0],[1,2,3,4]]
a = [1,3,2,0]
b = [1,2,3,4]
ide = tf.nn.top_k(a,k=4).indices
#ide = tf.reshape(ide,(-1,1))
ide = tf.reverse(ide,axis=[0])
print(tf.gather(b,ide))
print(ide)