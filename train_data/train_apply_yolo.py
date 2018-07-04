from models import iv2_mult_chan
import tensorflow as tf
import time
import config

import config
from model import get_loss,predict
import data_gen
from tensorflow.contrib import slim
import np_utils
import glob
import cv2
import utils
import numpy as np
import time
import visual

def train():
    img = tf.placeholder(shape=[config.batch_size, config.Config['min_dim'], config.Config['min_dim'], 3], dtype=tf.float32)
    anchors_num = sum(
        [config.Config['feature_maps'][s] ** 2 * config.Config['aspect_num'][s] for s in range(3)])

    loc = tf.placeholder(shape=[config.batch_size, anchors_num, 4], dtype=tf.float32)
    conf = tf.placeholder(shape=[config.batch_size, anchors_num], dtype=tf.float32)

    pred_loc, pred_confs, vbs = iv2_mult_chan.gen_box(img,config)


    train_tensors = get_loss(conf, loc, pred_loc, pred_confs,config)

    gen = data_gen.get_batch_inception(batch_size=config.batch_size,image_size=config.Config['min_dim'],max_detect=50)

    global_step = slim.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=10000,
        decay_rate=0.7,
        staircase=True)

    tf.summary.scalar('lr', lr)
    sum_op = tf.summary.merge_all()

    optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)

    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, '/home/dsl/all_check/inception_v2.ckpt')

    sv = tf.train.Supervisor(logdir='/home/dsl/all_check/face_detect/voc_ssd_yolo', summary_op=None, init_fn=restore)

    with sv.managed_session() as sess:
        for step in range(1000000000):

            images, true_box, true_label = next(gen)

            loct, conft = np_utils.get_loc_conf(true_box, true_label, batch_size=config.batch_size,cfg=config.Config)
            feed_dict = {img: images, loc: loct,
                         conf: conft}

            ls, step = sess.run([train_op, global_step], feed_dict=feed_dict)

            if step % 10 == 0:
                print('step:' + str(step) +
                      ' ' + 'class_loss:' + str(ls[0]) +
                      ' ' + 'loc_loss:' + str(ls[1])
                      )
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)

                print(ls)

train()