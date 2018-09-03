from keras.applications import densenet
import keras
import tensorflow as tf
import keras.layers as KL
from tensorflow.contrib import slim


def get_model(img):

    model = densenet.DenseNet121(include_top=False, input_tensor=img, weights=None, pooling=None)
    layer_out_put = [model.get_layer(name='conv{}_block{}_concat'.format(idx + 3, block_num)).output for idx, block_num
                     in
                     enumerate(
                         [12, 24, 16])]
    return layer_out_put

def detect_layer(img):
    c3, c4, c5 = get_model(img)
    P5 = KL.Conv2D(256,1,strides=1,padding='SAME')(c5)
    P5_UP = KL.UpSampling2D()(P5)
    P5 = KL.Conv2D(256, 3, strides=1, padding='SAME')(P5)
    P4 = KL.Conv2D(256, 1, strides=1, padding='SAME')(c4)
    P4 = P4+P5_UP
    P4_UP = KL.UpSampling2D()(P4)
    P4 = KL.Conv2D(256, 3, strides=1, padding='SAME')(P4)
    P3 = KL.Conv2D(256, 1, strides=1, padding='SAME')(c3)
    P3 = P3 + P4_UP
    P3 = KL.Conv2D(256, 3, strides=1, padding='SAME')(P3)
    P6 = KL.Conv2D(256,3,strides=2,padding='SAME')(c5)
    P7 = KL.Activation('relu',name='act')(P6)
    P7 = KL.Conv2D(256, 3, strides=2, padding='SAME')(P7)

    return [P3, P4, P5, P6, P7]
def smooth_l1_loss(y_true, y_pred):

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


    final_loss_l = tf.keras.backend.switch(num > 0,
                                           loss_l / num,
                                           tf.constant(0.0))

    return final_loss_c, final_loss_l

def classfy_model(feature_map, config):
    for s  in range(4):

    with tf.variable_scope('classfy',reuse=tf.AUTO_REUSE):
        with slim.arg_scope(parm_args.mul_channel_arg_scope()):
            feature_map = slim.repeat(feature_map,4,slim.conv2d,num_outputs=256,kernel_size=3,stride=1,scope='classfy_repeat')
        out_puts = slim.conv2d(feature_map, config.Config['num_classes'] * 9, kernel_size=3, stride=1,scope='classfy_conv',
                               weights_initializer=tf.initializers.zeros,activation_fn=None)
        out_puts = tf.reshape(out_puts,shape=(config.batch_size,-1, config.Config['num_classes']))
        #out_puts = slim.nn.sigmoid(out_puts)
    return out_puts

def regression_model(feature_map, config):
    with tf.variable_scope('regression', reuse=tf.AUTO_REUSE):
        with slim.arg_scope(parm_args.mul_channel_arg_scope()):
            feature_map = slim.repeat(feature_map, 4, slim.conv2d, num_outputs=256, kernel_size=3, stride=1,scope='regression_repeat')
        out_puts = slim.conv2d(feature_map, 4 * 9, kernel_size=3, stride=1,scope='regression',activation_fn=None)
        out_puts = tf.reshape(out_puts, shape=(config.batch_size,-1, 4))

    return out_puts

def model(img,cfg):
    fpns = detect_layer(img)
    logits = []
    boxes = []
    for fp in fpns:
        logits.append(classfy_model(fp))
        boxes.append(regression_model(fp))
    logits = tf.concat(logits, axis=1)
    boxes = tf.concat(boxes, axis=1)
    return boxes,logits


def main(config):
    img = keras.Input(shape=[512, 512, 3], dtype=tf.float32)
    anchors_num = sum(
        [config.Config['feature_maps'][s] ** 2 * config.Config['aspect_num'][s] for s in range(3)])
    loc = KL.Input(shape=[anchors_num, 4], dtype=tf.float32)
    conf = KL.Input(shape=[anchors_num], dtype=tf.float32)
    pred_loc, pred_confs = model(img, config)

    cls_loss, loc_loss = get_loss(conf, loc, pred_loc, pred_confs,config)
    ip = [img,loc,conf]
    out = [cls_loss, loc_loss]
    md = keras.models.Model(ip,out)
    opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)

    md.compile(optimizer=opt,loss=[out])

import config as cfg

main(cfg)

