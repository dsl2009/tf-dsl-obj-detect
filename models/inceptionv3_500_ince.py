from nets import inception_v3
import tensorflow as tf

from tensorflow.contrib import slim
import utils


def inception(cv,out_put,name,stride):
    with tf.variable_scope(name):
        cv8 = slim.conv2d(cv, 256, kernel_size=1, stride=1)
        c1 = slim.conv2d(cv8,out_put/4,kernel_size=1,stride=stride)
        c2 = slim.conv2d(cv8, out_put/4, kernel_size=3, stride=stride)
        c3 = slim.conv2d(cv8, out_put/4, kernel_size=5, stride=stride)
        c =  tf.concat([c1,c2,c3],axis=3)
        c = slim.conv2d(c,256,1,1)
    return c


def inception_v2_ssd(img,cfg):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_point = inception_v3.inception_v3_base(img)

        Mixed_5d = end_point['Mixed_5d']
        Mixed_6e = end_point['Mixed_6e']
        cell_11 = end_point['Mixed_7c']
        print(Mixed_5d,Mixed_6e,cell_11)
        vbs = slim.get_trainable_variables()
        #cell_11 = tf.image.resize_bilinear(cell_11,size=[32,32])
        cell_11 = Mixed_6e

        #cell_7 = tf.image.resize_bilinear(Mixed_6e,size=[64,64])
        cell_7 = Mixed_5d

    cell_11 = slim.conv2d(cell_11,1024,kernel_size=1,activation_fn=slim.nn.relu)

    cell_7 = slim.conv2d(cell_7, 512, kernel_size=3, activation_fn=slim.nn.relu)
    cell_7 = slim.conv2d(cell_7, 256, kernel_size=1, activation_fn=slim.nn.relu)

    cv6 = slim.conv2d(cell_11, 1024, kernel_size=3, rate=6, activation_fn=slim.nn.relu, scope='conv6')
    cv7 = slim.conv2d(cv6, 1024, kernel_size=1, activation_fn=slim.nn.relu, scope='conv7')

    s = utils.normalize_to_target(cell_7, target_norm_value=16.0, dim=1)

    cv8 = inception(cv7, out_put=512, name='cv8', stride=2)
    cv9 = inception(cv8, out_put=512, name='cv9', stride=2)
    cv10 = inception(cv9, out_put=512, name='cv10', stride=2)
    cv11 = inception(cv10, out_put=512,name= 'cv11', stride=2)

    source = [s, cv7, cv8, cv9, cv10, cv11]
    conf = []
    loc = []
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
