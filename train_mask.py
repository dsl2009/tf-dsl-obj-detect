import data_gen
import tensorflow as tf
import config
from ssd_mask_rcnn import mask_model
from models import mask_iv2
from tensorflow.contrib import slim
import np_utils
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import time
def train():
    img = tf.placeholder(shape=[config.batch_size, config.Config['min_dim'], config.Config['min_dim'], 3], dtype=tf.float32)
    anchors_num = sum(
        [config.Config['feature_maps'][s] ** 2 * config.Config['aspect_num'][s] for s in range(6)])

    input_loc_t = tf.placeholder(shape=[config.batch_size, anchors_num, 4], dtype=tf.float32)
    input_conf_t = tf.placeholder(shape=[config.batch_size, anchors_num], dtype=tf.float32)
    input_gt_mask = tf.placeholder(shape=[config.batch_size, 28,28,50],dtype=tf.int32)
    input_gt_box = tf.placeholder(shape=[config.batch_size, 50, 4],dtype=tf.float32)
    input_mask_index = tf.placeholder(shape=[config.batch_size, anchors_num],dtype=tf.int32)

    gen = data_gen.get_batch_shapes(batch_size=config.batch_size, image_size=512)

    input_gt_mask_trans = tf.transpose(input_gt_mask,[0,3,1,2])
    pred_loc, pred_confs, mask_fp, vbs = mask_iv2.inception_v2_ssd(img, config)

    target_mask = mask_model.get_target_mask(input_gt_box, input_gt_mask_trans, input_mask_index,config)

    train_tensors = mask_model.get_loss(input_conf_t, input_loc_t, pred_loc, pred_confs, target_mask, mask_fp, config)
    global_step = get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=20000,
        decay_rate=0.7,
        staircase=True)
    tf.summary.scalar('lr', lr)
    sum_op = tf.summary.merge_all()

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)

    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, '/home/dsl/all_check/inception_v2.ckpt')

    sv = tf.train.Supervisor(logdir='/home/dsl/all_check/face_detect/mask', summary_op=None, init_fn=restore)

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

train()