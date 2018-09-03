#coding=utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import utils
import data_gen
import cv2
import numpy as np
from util import gen_chors
import visual
import time
import glob
import np_utils
#tf.enable_eager_execution()
def model(inputs,cfg):
    source = []
    with tf.variable_scope('vgg_16',default_name=None, values=[inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d,  slim.max_pool2d]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            cov_43 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(cov_43, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, kernel_size=3,stride=1, scope='pool5')
            vbs = slim.get_trainable_variables()
            utils.count_parm()
            #vbs =None
            cv6 = slim.conv2d(net,1024,kernel_size=3,rate=6,activation_fn=slim.nn.relu,scope='conv6')
            cv7 = slim.conv2d(cv6,1024,kernel_size=1,activation_fn=slim.nn.relu,scope='conv7')

            s = utils.normalize_to_target(cov_43, target_norm_value=2.0, dim=1)



            cv8 = slim.conv2d(cv7,256,kernel_size=1,stride=1,scope='conv8_0')
            cv8 = slim.conv2d(cv8, 512, kernel_size=3, stride=2, scope='conv8_1')

            cv9 = slim.conv2d(cv8, 128, kernel_size=1, stride=1, scope='conv9_0')
            cv9 = slim.conv2d(cv9, 256, kernel_size=3, stride=2, scope='conv9_1')

            cv10 = slim.conv2d(cv9, 128, kernel_size=1, stride=1, scope='conv10_0')
            cv10 = slim.conv2d(cv10, 256, kernel_size=3, stride=2, scope='conv10_1')

            cv11 = slim.conv2d(cv10, 128, kernel_size=1, stride=1, scope='conv11_0')
            cv11 = slim.conv2d(cv11, 256, kernel_size=3, stride=2, scope='conv11_1')
            source = [s,cv7,cv8,cv9,cv10,cv11]
            conf = []
            loc = []
            for cv,num in zip(source,cfg.Config['aspect_num']):

                loc.append(slim.conv2d(cv,num*4,kernel_size=3,stride=1,activation_fn=None))

                conf.append(slim.conv2d(cv, num*cfg.Config['num_classes'], kernel_size=3, stride=1, activation_fn=None))
            print(loc)
        loc = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, 4)) for o in loc], axis=1)
        conf = tf.concat([tf.reshape(o, shape=(cfg.batch_size, -1, cfg.Config['num_classes'])) for o in conf], axis=1)
        return loc,conf,vbs

def loss_clc(truth_box,gt_lables):
    true_box, non_ze = utils.trim_zeros_graph(truth_box)
    gt_lables = tf.boolean_mask(gt_lables, non_ze)
    priors = utils.get_prio_box()
    priors_xywh = tf.concat((priors[:, :2] - priors[:, 2:] / 2,
               priors[:, :2] + priors[:, 2:] / 2), axis=1)
    ops = utils.overlaps_graph(true_box,priors_xywh)
    # 获取prior box 最佳匹配index size = trueth
    best_prior_idx = tf.argmax(ops,axis=1)
    # 获取truth box 最佳匹配index size = prior
    best_truth_idx = tf.argmax(ops,axis=0)
    best_truth_overlab = tf.reduce_max(ops,axis=0)
    matches = tf.gather(truth_box,best_truth_idx)
    conf = tf.gather(gt_lables,best_truth_idx)
    conf = tf.cast(conf,tf.float32)
    zer = tf.zeros(shape=(tf.shape(conf)),dtype=tf.float32)
    conf = tf.where(tf.less(best_truth_overlab,0.5),zer,conf)
    loc = utils.encode_box(matched=matches,prios=priors)
    return conf,loc
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss
def log_sum(x):
    mx = tf.reduce_max(x)
    data = tf.log(tf.reduce_sum(tf.exp(x - mx), axis=1)) + mx
    return tf.reshape(data, (-1, 1))
def soft_focal_loss(logits,labels,number_cls=20):
    labels = tf.one_hot(labels,number_cls)
    loss = tf.reduce_sum(labels*(-(1 - tf.nn.softmax(logits))**1*tf.log(tf.nn.softmax(logits))),axis=1)
    return loss

def get_aver_loss(logits,labels,number_cls=20):
    #total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logists,labels=tf.one_hot(labels,number_cls))
    total_loss = soft_focal_loss(logits=logits,labels=labels,number_cls=number_cls)
    tt = []
    for s  in range(number_cls):
        ix = tf.where(tf.equal(labels,s))
        ls = tf.gather(total_loss,ix)
        ls = tf.keras.backend.switch(tf.cast(tf.size(ix)>0,tf.bool), tf.reduce_mean(ls),tf.constant(0.0))
        tt.append(ls)
    ls = tf.stack(tt)
    ix = tf.where(tf.greater(ls,0))
    ls = tf.gather(ls,ix)
    return tf.reduce_mean(ls)





def get_loss(conf_t,loc_t,pred_loc, pred_confs,cfg):


   # conf_t, loc_t = utils.batch_slice(inputs=[truth_box, gt_lables], graph_fn=loss_clc, batch_size=batch_size)

    conf_t = tf.reshape(conf_t,shape=(-1,))
    loc_t = tf.reshape(loc_t,shape=(-1,4))

    positive_roi_ix = tf.where(conf_t > 0)[:, 0]

    positive_roi_class_ids = tf.cast(
        tf.gather(conf_t, positive_roi_ix), tf.int64)

    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    pred_loc = tf.reshape(pred_loc,shape=(-1,4))
    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(loc_t, positive_roi_ix)
    pred_bbox = tf.gather(pred_loc, positive_roi_ix)

    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                                   smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                                   tf.constant(0.0))
    loss_l = tf.reduce_sum(loss)

    pred_conf = tf.reshape(pred_confs,shape=(-1, cfg.Config['num_classes']))

    conf_t_tm = tf.cast(conf_t,tf.int32)
    conf_t_tm  = tf.reshape(conf_t_tm ,shape=(-1,))

    index = tf.stack([tf.range(tf.shape(conf_t_tm)[0]),conf_t_tm],axis=1)


    loss_c = log_sum(pred_conf) - tf.expand_dims(tf.gather_nd(pred_conf, index),-1)

    loss_c = tf.reshape(loss_c,shape=(cfg.batch_size,-1))
    conf_t = tf.reshape(conf_t, shape=(cfg.batch_size, -1))

    zeros = tf.zeros(shape=tf.shape(loss_c),dtype=tf.float32)
    loss_c = tf.where(tf.greater(conf_t,0),zeros,loss_c)


    pos_num = tf.reduce_sum(tf.cast(tf.greater(conf_t,0),dtype=tf.int32),axis=1)
    ne_num = pos_num*3



    los = []
    for s in range(cfg.batch_size):
        loss_tt = loss_c[s,:]
        value,ne_index = tf.nn.top_k(loss_tt,k=ne_num[s])
        pos_ix = tf.where(conf_t[s,:] > 0)[:,0]
        pos_ix = tf.cast(pos_ix,tf.int32)

        ix = tf.concat([pos_ix,ne_index],axis=0)

        label = tf.gather(conf_t[s,:],ix)
        label = tf.cast(label,tf.int32)
        lb,_,ct = tf.unique_with_counts(label)
        tf.summary.histogram('lbs', values=lb)
        tf.summary.histogram('ct', values=ct)

        logits = tf.gather(pred_confs[s,:],ix)
        '''
        label = tf.one_hot(label, depth=cfg.Config['num_classes'])
        ls = tf.keras.backend.switch(tf.size(label) > 0,
                                     tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits),
                                 tf.constant(0.0))

        ls = tf.reduce_sum(ls)
        '''
        #label = tf.one_hot(label, depth=cfg.Config['num_classes'])
        ls = tf.keras.backend.switch(tf.size(label) > 0,
                                     soft_focal_loss(labels=label, logits=logits,number_cls=cfg.Config['num_classes']),
                                     tf.constant(0.0))
        #ls = get_aver_loss(logists=logits,labels=label,number_cls=cfg.Config['num_classes'])
        ls = tf.reduce_sum(ls)
        los.append(ls)

    num = tf.reduce_sum(pos_num)
    num = tf.cast(num,dtype=tf.float32)

    final_loss_c = tf.keras.backend.switch(num > 0,
                                           tf.reduce_sum(los) / num,
                                 tf.constant(0.0))


    '''
    final_loss_c = tf.keras.backend.switch(num > 0,
                                        tf.reduce_mean(los),
                              tf.constant(0.0))
    '''

    final_loss_l = tf.keras.backend.switch(num > 0,
                                           loss_l / num,
                                           tf.constant(0.0))
    final_loss_c = final_loss_c

    tf.losses.add_loss(final_loss_c)


    tf.losses.add_loss(final_loss_l)

    total_loss = tf.losses.get_losses()
    tf.summary.scalar(name='class_loss',tensor=final_loss_c)
    tf.summary.scalar(name='loc_loss', tensor=final_loss_l)


    train_tensors = tf.identity(total_loss, 'ss')


    return train_tensors


def eger(cfg):
    gen = data_gen.get_batch(batch_size=cfg.batch_size)


    images, true_box, true_label = next(gen)
    print(true_label)
    loct, conft = np_utils.get_loc_conf(true_box, true_label, batch_size=cfg.batch_size)
    get_loss(images, conft, loct)

def predict(ig,pred_loc, pred_confs, vbs,cfg):

    #priors = utils.get_prio_box(cfg=cfg)
    #priors = np_utils.get_prio_box_new(cfg=cfg)
    priors = gen_chors.gen_ssd_anchors()
    print(priors.shape)
    box = utils.decode_box(prios=priors, pred_loc=pred_loc[0])
    props = slim.nn.softmax(pred_confs[0])
    pp = props[:,1:]

    cls = tf.argmax(pp, axis=1)
    pp = tf.reduce_max(pp,axis=1)


    ix = tf.where(tf.greater(pp,0.1))[:,0]
    score = tf.gather(pp,ix)
    box = tf.gather(box,ix)
    cls = tf.gather(cls, ix)
    box = tf.clip_by_value(box,clip_value_min=0.0,clip_value_max=1.0)

    keep = tf.image.non_max_suppression(
        scores=score,
        boxes=box,
        iou_threshold=0.4,
        max_output_size=50
    )
    return tf.gather(box,keep),tf.gather(score,keep),tf.gather(cls,keep)




def detect(x):
    ig = tf.placeholder(shape=(1, 300, 300, 3), dtype=tf.float32)
    box,score,pp = predict(ig)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/model.ckpt-23940')
        for ip in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/originalPics/2002/07/19/big/*.jpg'):
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(img, min_dim=300, max_dim=300)

            img = (org.astype(np.float32) - np.array([123.7, 116.8, 103.9])) / 255
            img = np.expand_dims(img, axis=0)
            t = time.time()
            bx,sc,p= sess.run([box,score,pp],feed_dict={ig:img})
            print(time.time()-t)
            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s]>0.4:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])

            #visual.display_instances(org,np.asarray(bxx)*300)
            visual.display_instances_title(org,np.asarray(bxx)*300,class_ids=np.asarray(cls),class_names=['face'],scores=scores)

def video():
    ig = tf.placeholder(shape=(1, 300, 300, 3), dtype=tf.float32)
    box, score, pp = predict(ig)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/model.ckpt-20512')
        cap = cv2.VideoCapture('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/tt.mp4')
        cap.set(3,320*3)
        cap.set(4,320*3)
        while True:
            ret ,frame = cap.read()


            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = utils.resize_image(img, min_dim=300, max_dim=300)

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
            if len(bxx)>0:
                finbox = utils.revert_image(scale,padding,300,np.asarray(bxx))
                for s in finbox:
                    cv2.rectangle(frame,pt1=(s[0],s[1]),pt2=(s[2],s[3]),color=(0,255,0),thickness=2)
            cv2.imshow('fram',frame)
            if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()






def train(cfg):
    img = tf.placeholder(shape=[cfg.batch_size, 300, 300, 3], dtype=tf.float32)
    #boxs = tf.placeholder(shape=[batch_size, 50, 4], dtype=tf.float32)
    #label = tf.placeholder(shape=[batch_size, 50], dtype=tf.int32)
    loc = tf.placeholder(shape=[cfg.batch_size, 7512,4], dtype=tf.float32)
    conf =  tf.placeholder(shape=[cfg.batch_size, 7512], dtype=tf.float32)

    pred_loc, pred_confs, vbs = model(img)

    train_tensors,sum_op = get_loss(conf,loc,pred_loc, pred_confs)

    gen = data_gen.get_batch(batch_size=cfg.batch_size)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)


    saver = tf.train.Saver(vbs)
    def restore(sess):
        saver.restore(sess,'/home/dsl/all_check/vgg_16.ckpt')

    sv = tf.train.Supervisor(logdir='/home/dsl/all_check/face_detect', summary_op=None, init_fn=restore)

    with sv.managed_session() as sess:
        for step in range(1000000000):

            images, true_box, true_label = next(gen)
            loct,conft = np_utils.get_loc_conf(true_box,true_label,batch_size=cfg.batch_size)
            feed_dict = {img: images, loc: loct,
                         conf: conft}

            ls = sess.run(train_op, feed_dict=feed_dict)
            if step%10==0:
                summaries = sess.run(sum_op,feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)
                print(ls)
#train()
#tf.enable_eager_execution()
#eger()

#detect('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2007/JPEGImages/000133.jpg')
#video()