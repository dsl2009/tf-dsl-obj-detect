from nets import inception_v2
import tensorflow as tf
import utils
from tensorflow.contrib import slim
from parm import parm_args
def inception_v2_ssd(img):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        logits, end_point = inception_v2.inception_v2_base(img)
        vbs = slim.get_variables_to_restore()

    c1 = end_point['Mixed_3c']
    c2 = end_point['Mixed_4e']
    c3 = end_point['Mixed_5c']
    return c1,c2,c3,vbs
def block_no_add(ip, filter_num,scope,rate=1):
    with tf.variable_scope(scope, 'Block35', [ip]):
        k = slim.conv2d(ip,filter_num/2, kernel_size=1, stride=1, padding='SAME')
        k = slim.conv2d(k, filter_num, kernel_size=3, stride=1,rate=rate, padding='SAME')
    return k


def detect_layer(img,cfg,is_train=True):
    c1,c2,c3,vbs = inception_v2_ssd(img)
    with slim.arg_scope(parm_args.mul_channel_arg_scope(is_train=is_train)):
        c3 = slim.repeat(c3,2,block_no_add,1024,scope='c3_block',rate = 4)
        k3 = slim.conv2d(c3, 512, kernel_size=1, stride=1, padding='SAME',scope='c3_block_k1_Conv')
        c3 = slim.conv2d(k3, 1024, kernel_size=3, stride=1, padding='SAME',scope='c3_block_k2_Conv')



        c3_2 = slim.conv2d(k3, 256, kernel_size=1, stride=1, padding='SAME',scope='c3_11_Conv')
        c3_2 = tf.image.resize_bilinear(c3_2,tf.shape(c2)[1:3])
        c2 = tf.concat([c3_2,c2],axis=3)

        c2 = slim.repeat(c2, 2, block_no_add, 512, scope='c2_block')
        k2 = slim.conv2d(c2, 256, kernel_size=1, stride=1, padding='SAME',scope='c2_block_k1_Conv')
        c2 = slim.conv2d(k2, 512, kernel_size=3, stride=1, padding='SAME',scope='c2_block_k2_Conv')


        c2_2 = slim.conv2d(k2, 128, kernel_size=1, stride=1, padding='SAME',scope='c2_11_Conv')
        c2_2 = tf.image.resize_bilinear(c2_2, tf.shape(c1)[1:3])
        c1 = tf.concat([c2_2, c1], axis=3)

        c1 = slim.repeat(c1, 2, block_no_add, 256, scope='c1_block')
        k1 = slim.conv2d(c1, 128, kernel_size=1, stride=1, padding='SAME',scope='c1_block_k1_Conv')
        c1 = slim.conv2d(k1, 256, kernel_size=3, stride=1, padding='SAME',scope='c1_block_k2_Conv')


    #c1 = utils.normalize_to_target(c1, target_norm_value=cfg.norm_value, dim=1, scope='norm1',summarize=False)
    #c2 = utils.normalize_to_target(c2, target_norm_value=cfg.norm_value, dim=1, scope='norm2',summarize=False)
    #c3 = utils.normalize_to_target(c3, target_norm_value=cfg.norm_value, dim=1, scope='norm3',summarize=False)


    return [c1,c2,c3],vbs


def gen_box(img,cfg):
    loc = []
    conf = []
    source,vbs = detect_layer(img,cfg)
    for cv, num in zip(source, cfg.Config['aspect_num']):
        print(num)
        loc.append(slim.conv2d(cv, num * 4, kernel_size=3, stride=1, activation_fn=None))

        conf.append(
            slim.conv2d(cv, num * cfg.Config['num_classes'], kernel_size=3, stride=1, activation_fn=None))
        print(loc)
    loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
    conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, cfg.Config['num_classes'])) for o in conf],
                     axis=1)

    return loc, conf, vbs

