from nets import inception_v2
import tensorflow as tf

from tensorflow.contrib import slim
import utils
from config import voc_inceptionv2_mobile_500

def inception_v2_ssd(img,cfg):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        logits, end_point = inception_v2.inception_v2_base(img)
        cell_7 = end_point['Mixed_4e']
        cell_11 = end_point['Mixed_5c']
        vbs = slim.get_trainable_variables()

        cv6 = slim.conv2d(cell_11, 1024, kernel_size=3, rate=6, activation_fn=slim.nn.relu, scope='conv6')
        cv7 = slim.conv2d(cv6, 1024, kernel_size=1, activation_fn=slim.nn.relu, scope='conv7')

        s = utils.normalize_to_target(cell_7, target_norm_value=2.0, dim=1)

        cv8 = slim.conv2d(cv7, 256, kernel_size=1, stride=1, scope='conv8_0')
        cv8 = slim.conv2d(cv8, 512, kernel_size=3, stride=2, scope='conv8_1')

        cv9 = slim.conv2d(cv8, 128, kernel_size=1, stride=1, scope='conv9_0')
        cv9 = slim.conv2d(cv9, 256, kernel_size=3, stride=2, scope='conv9_1')

        cv10 = slim.conv2d(cv9, 128, kernel_size=1, stride=1, scope='conv10_0')
        cv10 = slim.conv2d(cv10, 256, kernel_size=3, stride=2, scope='conv10_1')

        cv11 = slim.conv2d(cv10, 128, kernel_size=1, stride=1, scope='conv11_0')
        cv11 = slim.conv2d(cv11, 256, kernel_size=3, stride=2, scope='conv11_1')
        source = [s, cv7, cv8, cv9, cv10, cv11]
        conf = []
        loc = []
        for cv, num in zip(source, voc_inceptionv2_mobile_500['aspect_num']):
            print(num)
            loc.append(slim.conv2d(cv, num * 4, kernel_size=3, stride=1, activation_fn=None))

            conf.append(
                slim.conv2d(cv, num * voc_inceptionv2_mobile_500['num_classes'], kernel_size=3, stride=1, activation_fn=None))

        loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
        conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, voc_inceptionv2_mobile_500['num_classes'])) for o in conf],
                         axis=1)

        return loc, conf, vbs



