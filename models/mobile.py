from nets.mobilenet import mobilenet_v2
import tensorflow as tf

from tensorflow.contrib import slim
import utils

import config

def nana_mobile(img):
    logits, end_point = mobilenet_v2.mobilenet(img,num_classes=1001,depth_multiplier=1.4)
    for f in end_point:
        print(f,end_point[f])

x = tf.placeholder(dtype=tf.float32,shape=[1,512,512,4])
nana_mobile(x)