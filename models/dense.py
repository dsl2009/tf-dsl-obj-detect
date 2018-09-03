from tensorflow.contrib import slim
import tensorflow as tf
from models import coor_conv
bn = ['gamma','beta','mean','std']
def dense_block(x, blocks, name):

    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    x = slim.batch_norm(x, epsilon=1.001e-5, scope=name + '_bn',scale=True)
    x = slim.nn.relu(x, name=name + '_relu')
    x = slim.conv2d(x,int(x.get_shape().as_list()[3] * reduction),kernel_size=1,biases_initializer=None,scope=name + '_conv', activation_fn=None)
    x = slim.avg_pool2d(x,2,2,scope=name + '_pool')
    return x


def conv_block(x, growth_rate, name):
    x1 = slim.batch_norm(x, epsilon=1.001e-5, scope=name + '_0_bn',scale=True)
    x1 = slim.nn.relu(x1, name=name + '_0_relu')
    x1 = slim.conv2d(x1, 4 * growth_rate, 1, scope=name + '_1_conv', biases_initializer=None, activation_fn=None)

    x1 = slim.batch_norm(x1, epsilon=1.001e-5, scope=name + '_1_bn',scale=True)
    x1 = slim.nn.relu(x1, name=name + '_1_relu')
    x1 = slim.conv2d(x1, growth_rate, 3, scope=name + '_2_conv', padding='SAME', biases_initializer=None, activation_fn=None)
    x = tf.concat([x, x1], axis=3, name=name + '_concat')
    return x

def densenet(img_input,blocks = [6, 12, 24, 16]):
    x = slim.conv2d(img_input, 64, 7, stride=2, biases_initializer=None,activation_fn=None,scope='conv1/conv')
    x = slim.batch_norm(x, epsilon=1.001e-5, scope='conv1/bn',scale=True)
    x = slim.nn.relu(x, name='conv1/relu')
    x = slim.max_pool2d(x,kernel_size=3, stride=2, padding='SAME', scope='pool1')
    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x1 = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x1, 0.5, name='pool3')
    x2 = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x2, 0.5, name='pool4')
    x3 = dense_block(x, blocks[3], name='conv5')
    return x1,x2,x3


def densenet_with_coor(img_input,blocks = [6, 12, 24, 16]):
    x = coor_conv.CoordConv(x_dim=512, y_dim=512, with_r=False)(img_input)
    x = slim.conv2d(x, 64, 7, stride=2, biases_initializer=None,activation_fn=None,scope='conv1/conv')
    x = slim.batch_norm(x, epsilon=1.001e-5, scope='conv1/bn',scale=True)
    x = slim.nn.relu(x, name='conv1/relu')
    x = slim.max_pool2d(x,kernel_size=3, stride=2, padding='SAME', scope='pool1')
    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x1 = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x1, 0.5, name='pool3')
    x2 = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x2, 0.5, name='pool4')
    x3 = dense_block(x, blocks[3], name='conv5')
    return x1,x2,x3
