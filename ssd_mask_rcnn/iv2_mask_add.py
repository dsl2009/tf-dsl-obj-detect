from nets import inception_v2,resnet_v2
import tensorflow as tf
import utils
from tensorflow.contrib import slim
from parm import parm_args
def inception_v2_ssd(img):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        logits, end_point = inception_v2.inception_v2_base(img)
        vbs = slim.get_variables_to_restore()
        #vbs = None

    c1 = end_point['Mixed_3c']
    c2 = end_point['Mixed_4e']
    c3 = end_point['Mixed_5c']
    return c1,c2,c3,vbs

def resnetv2_ssd(img):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_50(img,is_training=True)

    c1 = end_points['resnet_v2_50/block1']
    c2 = end_points['resnet_v2_50/block2']
    base_16_0 = end_points['resnet_v2_50/block3']
    base_16_1 = end_points['resnet_v2_50/block4']
    vbs = slim.get_trainable_variables()
    # vbs = None
    base_16_0 = slim.conv2d(base_16_0, 512, 1)
    base_16_1 = slim.conv2d(base_16_1, 512, 1)
    c3 = tf.concat([base_16_0,base_16_1],axis=3)


    return c1,c2,c3,vbs

def block_no_add(ip, filter_num,scope,rate=1):
    with tf.variable_scope(scope, 'Block35', [ip]):
        k = slim.conv2d(ip,filter_num/2, kernel_size=1, stride=1, padding='SAME')
        k = slim.conv2d(k, filter_num, kernel_size=3, stride=1,rate=rate, padding='SAME')
    return k

def inception(ip,filter_num,scope,stride):
    with tf.variable_scope(scope):
        cv8 = slim.conv2d(ip, filter_num/2, kernel_size=1, stride=1)
        c1 = slim.conv2d(cv8,filter_num/4,kernel_size=1,stride=stride)
        c2 = slim.conv2d(cv8, filter_num/4, kernel_size=3, stride=stride)
        c3 = slim.conv2d(cv8, filter_num/4, kernel_size=5, stride=stride)

    return tf.concat([c1,c2,c3],axis=3)

def detect_layer(img,cfg,is_train=True):
    c1,c2,c3,vbs = inception_v2_ssd(img)
    c1 = slim.conv2d(c1,512,kernel_size=3,stride=1,rate=4)

    c3 = slim.conv2d(c3, 256, 1, 1, activation_fn=None)

    c2 = slim.conv2d(c2, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c3, size=tf.shape(c2)[1:3])

    c1 = slim.conv2d(c1, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c2, size=tf.shape(c1)[1:3])



    return [c1,c2,c3],vbs


def gen_box(img,cfg):
    loc = []
    conf = []
    source,vbs = detect_layer(img,cfg)
    for cv, num in zip(source, cfg.Config['aspect_num']):

        loc.append(slim.conv2d(cv, num * 4, kernel_size=3, stride=1, activation_fn=None))

        conf.append(
            slim.conv2d(cv, num * cfg.Config['num_classes'], kernel_size=3, stride=1, activation_fn=None))

    loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
    conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, cfg.Config['num_classes'])) for o in conf],
                     axis=1)

    return loc, conf, source,vbs


