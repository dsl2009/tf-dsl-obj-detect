from nets import inception_v2,resnet_v2,inception_v3,resnet_v1
import tensorflow as tf
import utils
from tensorflow.contrib import slim
from parm import parm_args
from models.coor_conv import CoordConv
def inception_v2_ssd(img):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        logits, end_point = inception_v2.inception_v2_base(img)
        vbs = slim.get_variables_to_restore()

    c1 = end_point['Mixed_3c']
    c2 = end_point['Mixed_4e']
    c3 = end_point['Mixed_5c']

    return c1,c2,c3,vbs

def inception_v3_ssd(img):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_point = inception_v3.inception_v3_base(img)
        vbs = slim.get_variables_to_restore()


    c1 = end_point['Mixed_5d']
    c2 = end_point['Mixed_6e']
    c3 = end_point['Mixed_7c']

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

def detect_layer11(img,cfg,is_train=True):
    c1,c2,c3,vbs = inception_v2_ssd(img)

    #c3 = CoordConv(x_dim=16,y_dim=16,with_r=False)(c3,num_outputs=512,kernel_size=3)


    c3 = slim.conv2d(c3, 256, 3, activation_fn=None)
    c2 = slim.conv2d(c2, 256, 3)
    c1 = slim.conv2d(c1, 256, 3)

    c2 = slim.conv2d(c2, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c3, size=tf.shape(c2)[1:3])

    c1 = slim.conv2d(c1, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c2, size=tf.shape(c1)[1:3])

    #c2 = tf.concat((c2,tf.image.resize_bilinear(c3, size=tf.shape(c2)[1:3])),axis=3)

    #c1 = tf.concat((c1, tf.image.resize_bilinear(c3, size=tf.shape(c1)[1:3])), axis=3)


    return [c1,c2,c3],vbs


def detect_layer_v3(img,cfg,is_train=True):
    c1,c2,c3,vbs = inception_v3_ssd(img)

    c1 = CoordConv(x_dim=61,y_dim=61,with_r=False)(c1,num_outputs=512,kernel_size=3)

    c3 = CoordConv(x_dim=14,y_dim=14,  with_r=False)(c3, num_outputs = 256, kernel_size=1,activation_fn=None)
    c2 = CoordConv(x_dim=30, y_dim=30, with_r=False)(c2, num_outputs=256, kernel_size=1, activation_fn=None)
    c1 = CoordConv(x_dim=61, y_dim=61, with_r=False)(c1, num_outputs=256, kernel_size=1, activation_fn=None)

    #c2 = slim.conv2d(c2, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c3, size=tf.shape(c2)[1:3])

    #c1 = slim.conv2d(c1, 256, 1, 1, activation_fn=None) + tf.image.resize_bilinear(c2, size=tf.shape(c1)[1:3])

    c2 = tf.concat((c2,tf.image.resize_bilinear(c3, size=tf.shape(c2)[1:3])),axis=3)

    c1 = tf.concat((c1, tf.image.resize_bilinear(c3, size=tf.shape(c1)[1:3])), axis=3)


    return [c1,c2,c3],vbs


def detect_layer(img,cfg,is_train=True):
    c1,c2,c3,vbs = inception_v2_ssd(img)
    tmp1 = slim.conv2d(c1, num_outputs=256, kernel_size=[1, 13])
    tmp1 = slim.conv2d(tmp1, num_outputs=128, kernel_size=[13, 1],activation_fn=None)

    tmp2 = slim.conv2d(c1, num_outputs=256, kernel_size=[13, 1])
    tmp2 = slim.conv2d(tmp2, num_outputs=128, kernel_size=[1, 13],activation_fn=None)
    c1 = tmp1+tmp2

    tmp1 = slim.conv2d(c2, num_outputs=256, kernel_size=[1, 13])
    tmp1 = slim.conv2d(tmp1, num_outputs=128, kernel_size=[13, 1],activation_fn=None)
    tmp2 = slim.conv2d(c2, num_outputs=256, kernel_size=[13, 1])
    tmp2 = slim.conv2d(tmp2, num_outputs=128, kernel_size=[1, 13],activation_fn=None)
    c2 = tmp1 + tmp2

    tmp1 = slim.conv2d(c3, num_outputs=256, kernel_size=[1, 13])
    tmp1 = slim.conv2d(tmp1, num_outputs=128, kernel_size=[13, 1],activation_fn=None)

    tmp2 = slim.conv2d(c3, num_outputs=256, kernel_size=[13, 1])
    tmp2 = slim.conv2d(tmp2, num_outputs=128, kernel_size=[1, 13],activation_fn=None)
    c3 = tmp1 + tmp2




    return [c1,c2,c3],vbs


def gen_box(img,cfg):
    loc = []
    conf = []
    source,vbs = detect_layer(img,cfg)
    for cv, num in zip(source, cfg.Config['aspect_num']):

        loc.append(slim.conv2d(cv, num * 4, kernel_size=3, stride=1, activation_fn=None))

        conf.append(
            slim.conv2d(cv, num * cfg.Config['num_classe7256s'], kernel_size=3, stride=1, activation_fn=None))

    loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
    conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, cfg.Config['num_classes'])) for o in conf],
                     axis=1)

    return loc, conf, vbs

def gen_box_cai(img,cfg):
    loc = []
    conf = []
    source,vbs = detect_layer(img,cfg)
    with tf.variable_scope('regression_bb', reuse=tf.AUTO_REUSE):
        for cv, num in zip(source, cfg.Config['aspect_num']):
            tmp = slim.conv2d(cv, 256, kernel_size=[1,13], stride=1,scope='cv1')
            tmp = slim.conv2d(tmp, num * 4, kernel_size=[13,1], stride=1,activation_fn=None,scope='cv2')
            loc.append(tmp)

            tmp = slim.conv2d(tmp, num * cfg.Config['num_classes'], kernel_size=3, stride=1, activation_fn=None,scope='cv3')
            conf.append(tmp)

    loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
    conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, cfg.Config['num_classes'])) for o in conf],
                     axis=1)

    return loc, conf, vbs

def gen_box_fen(img, cfg):
    loc = []
    conf = []
    source, vbs = detect_layer(img, cfg)
    for cv, num in zip(source, cfg.Config['aspect_num']):
        tmp_loc = []
        tmp_conf = []
        for x in range(3):
            tmp_loc.append(slim.conv2d(cv, num * 4/3, kernel_size=3, stride=1,rate=x+1, activation_fn=None))

            tmp_conf.append(
                slim.conv2d(cv, num * cfg.Config['num_classes']/3, kernel_size=3, stride=1, rate=x+1,activation_fn=None))
        loc.append(tf.concat(tmp_loc,axis=3))
        conf.append(tf.concat(tmp_conf))

    loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
    conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, cfg.Config['num_classes'])) for o in conf],
                     axis=1)

    return loc, conf, vbs