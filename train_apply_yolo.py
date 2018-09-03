from models import iv2_mult_chan_add,iv2_mult_chan,retinanet
import tensorflow as tf
import time
import config
from data_set.data_loader import q
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
    #ig = AddCoords(x_dim=512,y_dim=512)(img)
    anchors_num = sum(
        [config.Config['feature_maps'][s] ** 2 * config.Config['aspect_num'][s] for s in range(5)])
    loc = tf.placeholder(shape=[config.batch_size, anchors_num, 4], dtype=tf.float32)
    conf = tf.placeholder(shape=[config.batch_size, anchors_num], dtype=tf.float32)
    pred_loc, pred_confs, vbs = retinanet.model(img,config)
    train_tensors = get_loss(conf, loc, pred_loc, pred_confs,config)
    gen = data_gen.get_batch_inception(batch_size=config.batch_size,image_size=config.Config['min_dim'],max_detect=50)

    global_step = slim.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=40000,
        decay_rate=0.7,
        staircase=True)

    tf.summary.scalar('lr', lr)
    sum_op = tf.summary.merge_all()

    optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)
    vbs = []
    for s in slim.get_variables():
        print(s.name)
        if 'resnet_v2_50' in s.name and 'Momentum' not in s.name:
            print(s.name)
            vbs.append(s)

    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, config.check_dir)


    sv = tf.train.Supervisor(logdir=config.save_dir, summary_op=None, init_fn=restore)

    with sv.managed_session() as sess:
        for step in range(200000):
            print('       '+' '.join(['*']*(step%10)))
            images, true_box, true_label = q.get()

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

def detect():
    config.batch_size = 1
    imgs = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)
    #ig = AddCoords(x_dim=512, y_dim=512)(imgs)
    pred_loc, pred_confs, vbs = retinanet.model(imgs,config)
    box,score,pp = predict(imgs,pred_loc, pred_confs, vbs,config.Config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/resnet50_pasc/model.ckpt-199863')
        for ip in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/*.jpg'):
            print(ip)
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(img, min_dim=512, max_dim=512)

            #img = (org/ 255.0-0.5)*2
            img = org - [123.15, 115.90, 103.06]
            img = np.expand_dims(img, axis=0)
            t = time.time()
            bx,sc,p= sess.run([box,score,pp],feed_dict={imgs:img})
            print(time.time()-t)
            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s]>0.3:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])
            if len(bxx) > 0:
                #visual.display_instances(org,np.asarray(bxx)*300)
                visual.display_instances_title(org,np.asarray(bxx)*512,class_ids=np.asarray(cls),class_names=config.VOC_CLASSES,scores=scores)
def video():
    config.batch_size = 1
    ig = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)
    pred_loc, pred_confs, vbs = retinanet.model(ig,config)
    box,score,pp = predict(ig,pred_loc, pred_confs, vbs,config.Config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/resnet50/model.ckpt-18756')
        cap = cv2.VideoCapture('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/jijing.mp4')
        #cap = cv2.VideoCapture(0)
        cap.set(3, 320 * 3)
        cap.set(4, 320 * 3)
        t1 = time.time()
        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(img, min_dim=config.Config['min_dim'],
                                                                   max_dim=config.Config['min_dim'])

            img = org - [123.15, 115.90, 103.06]
            img = np.expand_dims(img, axis=0)
            t = time.time()

            bx, sc, p = sess.run([box, score, pp], feed_dict={ig: img})

            fps = int(1 / (time.time() - t) * 10) / 10.0

            cv2.putText(frame, 'fps:' + str(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)

            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s] > 0.4:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])
            if len(bxx) > 0:
                finbox = utils.revert_image(scale, padding, config.Config['min_dim'], np.asarray(bxx))
                for ix, s in enumerate(finbox):
                    cv2.rectangle(frame, pt1=(s[0], s[1]), pt2=(s[2], s[3]), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, config.VOC_CLASSES[cls[ix]] + '_' + str(scores[ix])[0:4], (s[0], s[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)

            cv2.imshow('fram', frame)

            if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                break
        print('ss')

        cap.release()
        cv2.destroyAllWindows()


detect()