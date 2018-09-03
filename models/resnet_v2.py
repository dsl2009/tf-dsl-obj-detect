from nets.resnet_v2 import resnet_v2_block,resnet_v2,resnet_arg_scope
import tensorflow as tf
from tensorflow.contrib import slim

def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=2),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)


def fpn(img):
    with slim.arg_scope(resnet_arg_scope()):
        _, endpoint = resnet_v2_50(img)
    c1 = endpoint['resnet_v2_50/block1']
    c2 = endpoint['resnet_v2_50/block2']
    c3 = endpoint['resnet_v2_50/block3']
    c4 = endpoint['resnet_v2_50/block4']

    p5 = slim.conv2d(c3, 256, 1, activation_fn=None)
    p5_upsample = tf.image.resize_bilinear(p5, tf.shape(c2)[1:3])
    p5 = slim.conv2d(p5, 256, 3, rate=4)
    p5 = slim.conv2d(p5, 256, 3, activation_fn=None)

    p4 = slim.conv2d(c2, 256, 1, activation_fn=None)
    p4 = p4 + p5_upsample
    p4_upsample = tf.image.resize_bilinear(p4, tf.shape(c1)[1:3])
    p4 = slim.conv2d(p4, 256, 3, rate=4)
    p4 = slim.conv2d(p4, 256, 3, activation_fn=None)

    p3 = slim.conv2d(c1, 256, 1, activation_fn=None)
    p3 = p3 + p4_upsample
    p3 = slim.conv2d(p3, 256, 3, rate=4)
    p3 = slim.conv2d(p3, 256, 3, activation_fn=None)

    p6 = slim.conv2d(c4,1024,kernel_size=1)
    p6 = slim.conv2d(p6, 256, 3, rate=4)
    p6 = slim.conv2d(p6, 256, kernel_size=3, stride=1, activation_fn=None)

    p7 = slim.nn.relu(p6)
    p7 = slim.conv2d(p7, 256, kernel_size=3, stride=2, activation_fn=None)
    return [p3, p4, p5, p6, p7]

