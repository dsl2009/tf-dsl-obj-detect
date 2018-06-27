import data_gen
import tensorflow as tf
import config
from ssd_mask_rcnn import mask_model
from models import mask_iv2
from tensorflow.contrib import slim
import np_utils
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import time
from data_set import shapes
import utils
import numpy as np
import visual
from matplotlib import pyplot as plt
import cv2
def train():
    img = tf.placeholder(shape=[config.batch_size, config.Config['min_dim'], config.Config['min_dim'], 3], dtype=tf.float32)
    anchors_num = sum(
        [config.Config['feature_maps'][s] ** 2 * config.Config['aspect_num'][s] for s in range(6)])

    input_loc_t = tf.placeholder(shape=[config.batch_size, anchors_num, 4], dtype=tf.float32)
    input_conf_t = tf.placeholder(shape=[config.batch_size, anchors_num], dtype=tf.float32)
    input_gt_mask = tf.placeholder(shape=[config.batch_size, 28,28,100],dtype=tf.int32)
    input_gt_box = tf.placeholder(shape=[config.batch_size, 100, 4],dtype=tf.float32)
    input_mask_index = tf.placeholder(shape=[config.batch_size, anchors_num],dtype=tf.int32)

    gen = data_gen.get_coco(batch_size=config.batch_size, max_detect=100,image_size=512)

    input_gt_mask_trans = tf.transpose(input_gt_mask,[0,3,1,2])
    pred_loc, pred_confs, mask_fp, vbs = mask_iv2.inception_v2_ssd(img, config)

    target_mask = mask_model.get_target_mask(input_gt_box, input_gt_mask_trans, input_mask_index,config)

    train_tensors = mask_model.get_loss(input_conf_t, input_loc_t, pred_loc, pred_confs, target_mask, mask_fp, config)
    global_step = get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=10000,
        decay_rate=0.7,
        staircase=True)
    tf.summary.scalar('lr', lr)
    sum_op = tf.summary.merge_all()

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)

    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, '/home/dsl/all_check/inception_v2.ckpt')

    sv = tf.train.Supervisor(logdir='/home/dsl/all_check/face_detect/coco', summary_op=None, init_fn=restore)

    with sv.managed_session() as sess:
        for step in range(1000000000):

            data_images, data_true_box, data_true_label, data_true_mask = next(gen)

            data_loct, data_conft,data_mask_index = np_utils.get_loc_conf_mask(data_true_box, data_true_label,
                                                                batch_size=config.batch_size,cfg=config.Config)
            feed_dict = {img: data_images, input_loc_t: data_loct,
                         input_conf_t: data_conft,input_gt_mask:data_true_mask,input_gt_box:data_true_box,
                         input_mask_index:data_mask_index
                         }


            t = time.time()
            ls, step = sess.run([train_op, global_step], feed_dict=feed_dict)

            if step % 10 == 0:
                tt = time.time() - t

                print('step:'+str(step)+
                      ' '+'class_loss:'+str(ls[0])+
                      ' '+'loc_loss:'+str(ls[1])+
                      ' '+'mask_loss:'+str(ls[2])+
                      ' '+'timestp:'+str(tt))
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)

def detect():
    config.batch_size = 1
    ig = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)
    pred_loc, pred_confs, mask_fp, vbs = mask_iv2.inception_v2_ssd(ig, config)
    box,score,pp,masks = mask_model.predict(pred_loc, pred_confs,mask_fp, config.Config)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/mask3/model.ckpt-19234')
        for ip in range(100):
            img, cls_id, _, _ = shapes.get_image()

            org, window, scale, padding, crop = utils.resize_image(img, min_dim=512, max_dim=512)

            img = (org/ 255.0-0.5)*2
            img = np.expand_dims(img, axis=0)

            t = time.time()
            bx,sc,p,msks= sess.run([box,score,pp,masks],feed_dict={ig:img})
            print(time.time()-t)
            bxx = []
            cls = []
            scores = []
            imgs = []
            for s in range(len(p)):
                if sc[s]>0.5:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])
                    zer_ig = np.zeros(shape=[512,512],dtype=np.float32)

                    bxes = np.asarray(bx[s]*512,np.int32)
                    igs = cv2.resize(msks[s,:,:],dsize=(bxes[2]-bxes[0],bxes[3]-bxes[1]))
                    igs = np.asarray(igs*255,np.int)

                    #igs[np.where(igs > 0.5)] = 1
                    #igs[np.where(igs <= 0.2)] = 0
                    zer_ig[bxes[1]:bxes[3],bxes[0]:bxes[2]] = igs
                    imgs.append(zer_ig)
                    #plt.imshow(zer_ig)
                    #plt.show()

            imgs = np.asarray(imgs)

            imgs = np.sum(imgs,axis=0)
            imgs = imgs*255.0
            imgs[np.where(igs > 255)] = 255
            #imgs[np.where(igs <= 100)] = 0
            imgs = np.asarray(imgs,np.int)

            #imgs = np.clip(imgs,0,1)
            plt.imshow(imgs,cmap='gray')
            plt.show()

            if len(bxx) > 0:
                #visual.display_instances(org,np.asarray(bxx)*300)
                visual.display_instances_title(org,np.asarray(bxx)*512,class_ids=np.asarray(cls),
                                               class_names=["square", "circle", "triangle"],scores=scores)

train()