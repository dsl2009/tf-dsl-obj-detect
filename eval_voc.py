from models import iv2_mult_chan,iv2_mult_chan_add
import tensorflow as tf
import eval_utils
import config
from model import predict
import os
import xml.etree.cElementTree as ET
import glob
import cv2
import utils
import numpy as np
import time
import visual
from config import VOC_CLASSES
class_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

def parse_rec(filename,height, width):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    box = []
    cls = []
    for obj in tree.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if  difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1

            cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
            bndbox.append(cur_pt)
        box.append(bndbox)
        label_idx = class_ind[name]
        cls.append(label_idx)

    return box,cls


def load_gt():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2007/'
    dts = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
    with open(dts) as f:
        for s in f.readlines():
            img_id = s.replace('\n','')

            img_path = os.path.join(rt, 'JPEGImages', img_id + '.jpg')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape

            box, cls = parse_rec(os.path.join(rt, 'Annotations', img_id + '.xml'),height, width)



def detect():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2007/'
    dts = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    config.batch_size = 1
    ig = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)
    pred_loc, pred_confs, vbs = iv2_mult_chan_add.gen_box_cai(ig,config)
    box,score,pp = predict(ig,pred_loc, pred_confs, vbs,config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/loss_change/model.ckpt-208983')
        with open(dts) as f:
            ct = 1
            total_aps = []
            for s in f.readlines():
                img_id = s.replace('\n', '')

                img_path = os.path.join(rt, 'JPEGImages', img_id + '.jpg')
                img = cv2.imread(img_path)

                height, width, channels = img.shape

                gt_box, gt_cls = parse_rec(os.path.join(rt, 'Annotations', img_id + '.xml'), height, width)


                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                org, window, scale, padding, crop = utils.resize_image(img, min_dim=512, max_dim=512)

                img = (org/ 255.0-0.5)*2
                img = np.expand_dims(img, axis=0)
                t = time.time()
                bx,sc,p= sess.run([box,score,pp],feed_dict={ig:img})

                bxx = []
                cls = []
                scores = []
                for kk in range(len(p)):
                    if sc[kk]>0.5:
                        bxx.append(bx[kk])
                        cls.append(p[kk])
                        scores.append(sc[kk])


                if len(cls)>0:
                    finbox = utils.revert_image(scale, padding, config.Config['min_dim'], np.asarray(bxx))
                    finbox = np.asarray(finbox,np.float32)
                    finbox[:, 0] = finbox[:, 0]*1.0/width
                    finbox[:, 1] = finbox[:, 1]*1.0/height
                    finbox[:, 2] = finbox[:, 2]*1.0/width
                    finbox[:, 3] = finbox[:, 3]*1.0/height


                    mAP, precisions, recalls, overlaps = eval_utils.compute_ap(gt_boxes=np.asarray(gt_box),
                                        gt_class_ids=np.asarray(gt_cls),
                                        pred_boxes=finbox,
                                        pred_class_ids=np.asarray(cls),
                                        pred_scores=np.asarray(scores))
                    print(mAP,precisions, recalls, overlaps)
                    total_aps.append(mAP)

                    print(sum(total_aps)/len(total_aps))
                ct = ct + 1
                #visual.display_instances_title(org, np.asarray(bxx) * 512, class_ids=np.asarray(cls),
                                            #class_names=config.VOC_CLASSES, scores=scores)

detect()