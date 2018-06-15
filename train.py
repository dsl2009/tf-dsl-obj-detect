from models import mobile
import tensorflow as tf
import config
from config import voc_nana_mobile_448 as cfg
from model import get_loss, predict
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
    img = tf.placeholder(
        shape=[config.batch_size, cfg["min_dim"], cfg["min_dim"], 3], dtype=tf.float32
    )
    anchors_num = sum(
        [cfg["feature_maps"][s] ** 2 * cfg["aspect_num"][s] for s in range(6)]
    )

    loc = tf.placeholder(shape=[config.batch_size, anchors_num, 4], dtype=tf.float32)
    conf = tf.placeholder(shape=[config.batch_size, anchors_num], dtype=tf.float32)

    pred_loc, pred_confs, vbs = mobile.nana_mobile(img, config)

    train_tensors, sum_op = get_loss(conf, loc, pred_loc, pred_confs, config)

    gen = data_gen.get_batch(batch_size=config.batch_size, image_size=cfg["min_dim"])
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)

    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, "/home/dsl/all_check/nasnet-a_mobile_04_10_2017/model.ckpt")

    sv = tf.train.Supervisor(
        logdir="/home/dsl/all_check/face_detect/nana", summary_op=None, init_fn=restore
    )

    with sv.managed_session() as sess:
        for step in range(1000000000):

            images, true_box, true_label = next(gen)
            loct, conft = np_utils.get_loc_conf(
                true_box, true_label, batch_size=config.batch_size, cfg=cfg
            )
            feed_dict = {img: images, loc: loct, conf: conft}

            ls = sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)
                print(ls)


def detect():
    config.batch_size = 1
    ig = tf.placeholder(shape=(1, 448, 448, 3), dtype=tf.float32)
    pred_loc, pred_confs, vbs = mobile.nana_mobile(ig, config)
    box, score, pp = predict(ig, pred_loc, pred_confs, vbs, cfg)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/home/dsl/all_check/face_detect/nana/model.ckpt-6725")
        for ip in glob.glob(
            "/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/originalPics/2002/07/19/big/*.jpg"
        ):
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(
                img, min_dim=448, max_dim=448
            )

            img = (org.astype(np.float32) - np.array([123.7, 116.8, 103.9])) / 255
            img = np.expand_dims(img, axis=0)
            t = time.time()
            bx, sc, p = sess.run([box, score, pp], feed_dict={ig: img})
            print(time.time() - t)
            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s] > 0.5:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])

            # visual.display_instances(org,np.asarray(bxx)*300)
            visual.display_instances_title(
                org,
                np.asarray(bxx) * 448,
                class_ids=np.asarray(cls),
                class_names=["face"],
                scores=scores,
            )


detect()
