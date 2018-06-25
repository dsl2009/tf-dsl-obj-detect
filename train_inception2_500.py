from models import inception_500_dsl,inception_500_ince
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
    anchors_num = sum(
        [config.Config['feature_maps'][s] ** 2 * config.Config['aspect_num'][s] for s in range(6)])

    loc = tf.placeholder(shape=[config.batch_size, anchors_num, 4], dtype=tf.float32)
    conf = tf.placeholder(shape=[config.batch_size, anchors_num], dtype=tf.float32)

    pred_loc, pred_confs, vbs = inception_500_ince.inception_v2_ssd(img,config)


    train_tensors, sum_op = get_loss(conf, loc, pred_loc, pred_confs,config)

    gen = data_gen.get_batch_inception(batch_size=config.batch_size,image_size=config.Config['min_dim'],max_detect=50)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)

    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, '/home/dsl/all_check/inception_v2.ckpt')

    sv = tf.train.Supervisor(logdir='/home/dsl/all_check/face_detect/voc-1', summary_op=None, init_fn=restore)

    with sv.managed_session() as sess:
        for step in range(1000000000):

            images, true_box, true_label = q.get()

            loct, conft = np_utils.get_loc_conf(true_box, true_label, batch_size=config.batch_size,cfg=config.Config)
            feed_dict = {img: images, loc: loct,
                         conf: conft}

            ls = sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)
                print(ls)


def detect():
    config.batch_size = 1
    ig = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)
    pred_loc, pred_confs, vbs = inception_500_dsl.inception_v2_ssd(ig,config)
    box,score,pp = predict(ig,pred_loc, pred_confs, vbs,config.Config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/tx/model.ckpt-29518')
        for ip in glob.glob('/home/dsl/ssd300_pascalVOC_pred_04.png'):
            print(ip)
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(img, min_dim=512, max_dim=512)

            img = (org/ 255.0-0.5)*2
            img = np.expand_dims(img, axis=0)
            t = time.time()
            bx,sc,p= sess.run([box,score,pp],feed_dict={ig:img})
            print(time.time()-t)
            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s]>0.5:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])
            if len(bxx) > 0:
                #visual.display_instances(org,np.asarray(bxx)*300)
                visual.display_instances_title(org,np.asarray(bxx)*512,class_ids=np.asarray(cls),class_names=config.VOC_CLASSES,scores=scores)

def video():
    config.batch_size = 1
    ig = tf.placeholder(shape=(1, config.Config['min_dim'], config.Config['min_dim'], 3), dtype=tf.float32)
    pred_loc, pred_confs, vbs = inception_500_dsl.inception_v2_ssd(ig, config)
    box, score, pp = predict(ig, pred_loc, pred_confs, vbs, config.Config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/voc-aug/model.ckpt-91518')
        cap = cv2.VideoCapture('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/jijing.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        out = cv2.VideoWriter('output.mpg', fourcc, 20.0, (1920, 1080))
        #cap = cv2.VideoCapture(0)
        cap.set(3,320*3)
        cap.set(4,320*3)
        t1 = time.time()
        while True:
            ret ,frame = cap.read()
            if not ret:
                break
            if time.time() -t1 >240:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(img, min_dim=config.Config['min_dim'], max_dim=config.Config['min_dim'])

            img = (org / 255.0 - 0.5) * 2
            img = np.expand_dims(img, axis=0)
            t = time.time()

            bx, sc, p = sess.run([box, score, pp], feed_dict={ig: img})

            fps = int(1/(time.time() - t)*10)/10.0

            cv2.putText(frame,  'fps:' + str(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)

            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s] > 0.4:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])
            if len(bxx)>0:
                finbox = utils.revert_image(scale,padding,config.Config['min_dim'],np.asarray(bxx))
                for ix,s in enumerate(finbox):
                    cv2.rectangle(frame,pt1=(s[0],s[1]),pt2=(s[2],s[3]),color=(0,255,0),thickness=2)
                    cv2.putText(frame, config.VOC_CLASSES[cls[ix]]+'_'+str(scores[ix])[0:4], (s[0], s[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            out.write(frame)
            cv2.imshow('fram',frame)

            if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                break
        print('ss')
        out.release()
        cap.release()
        cv2.destroyAllWindows()
detect()