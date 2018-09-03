from nets import inception_v2,resnet_v2,inception_v3
import tensorflow as tf
import config
from tensorflow.contrib import slim
import config
from parm import parm_args
from models import dense
from models import resnet_v2
def inception_v2_ssd(img):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        logits, end_point = inception_v2.inception_v2_base(img)
    c1 = end_point['Mixed_3c']
    c2 = end_point['Mixed_4e']
    c3 = end_point['Mixed_5c']
    return c1,c2,c3





def fpn(img):
    with tf.variable_scope(name_or_scope='Densenet_121'):
        c1, c2, c3 = dense.densenet(img)
    p5 = slim.conv2d(c3, 256, 1, activation_fn=None)
    p5_upsample = tf.image.resize_bilinear(p5, tf.shape(c2)[1:3])
    p5 = slim.conv2d(p5, 256, 3, activation_fn=None)

    p4 = slim.conv2d(c2, 256, 1, activation_fn=None)
    p4 = p4 + p5_upsample
    p4_upsample = tf.image.resize_bilinear(p4, tf.shape(c1)[1:3])
    p4 = slim.conv2d(p4, 256, 3, activation_fn=None)

    p3 = slim.conv2d(c1, 256, 1, activation_fn=None)
    p3 = p3 + p4_upsample
    p3 = slim.conv2d(p3, 256, 3, activation_fn=None)

    p6 = slim.conv2d(c3, 256, kernel_size=3, stride=2, activation_fn=None)

    p7 = slim.nn.relu(p6)
    p7 = slim.conv2d(p7, 256, kernel_size=3, stride=2, activation_fn=None)

    return [p3, p4, p5, p6, p7]

def classfy_model(feature_map,ix):
    with tf.variable_scope('classfy'+str(ix),reuse=tf.AUTO_REUSE):
        with slim.arg_scope(parm_args.mul_channel_arg_scope()):
            feature_map = slim.repeat(feature_map,4,slim.conv2d,num_outputs=256,kernel_size=3,stride=1,scope='classfy_repeat')
        out_puts = slim.conv2d(feature_map, config.Config['num_classes'] * 9, kernel_size=3, stride=1,scope='classfy_conv',
                               weights_initializer=tf.initializers.zeros,activation_fn=None)
        out_puts = tf.reshape(out_puts,shape=(config.batch_size,-1, config.Config['num_classes']))
        #out_puts = slim.nn.sigmoid(out_puts)
    return out_puts

def regression_model(feature_map,ix):
    with tf.variable_scope('regression'+str(ix), reuse=tf.AUTO_REUSE):
        with slim.arg_scope(parm_args.mul_channel_arg_scope()):
            feature_map = slim.repeat(feature_map, 4, slim.conv2d, num_outputs=256, kernel_size=3, stride=1,scope='regression_repeat')
        out_puts = slim.conv2d(feature_map, 4 * 9, kernel_size=3, stride=1,scope='regression',activation_fn=None)
        out_puts = tf.reshape(out_puts, shape=(config.batch_size,-1, 4))

    return out_puts,feature_map

def hebing(feature_map,scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with slim.arg_scope(parm_args.mul_channel_arg_scope()):
            feature_map = slim.repeat(feature_map, 4, slim.conv2d, num_outputs=256, kernel_size=3, stride=1,scope='regression_repeat')
        box = slim.conv2d(feature_map, 4 * 9, kernel_size=3, stride=1,scope='regression',activation_fn=None)
        box = tf.reshape(box, shape=(config.batch_size,-1, 4))

        logits = slim.conv2d(feature_map, config.Config['num_classes'] * 9, kernel_size=3, stride=1,
                               scope='classfy_conv',
                               weights_initializer=tf.initializers.zeros, activation_fn=None)
        logits = tf.reshape(logits, shape=(config.batch_size, -1, config.Config['num_classes']))


    return box, logits


def model(img,cfg):
    fpns = resnet_v2.fpn(img)
    print(fpns)
    logits = []
    boxes = []
    mask_fp = []
    for ix, fp in enumerate(fpns):
        logits.append(classfy_model(fp,0))
        box, m_fp = regression_model(fp, 0)
        boxes.append(box)
        if ix!=4:
            mask_fp.append(m_fp)


    logits = tf.concat(logits, axis=1)
    boxes = tf.concat(boxes, axis=1)

    return boxes,logits,mask_fp