import os
import sys
import time
import numpy as np
import imgaug
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import utils
from matplotlib import pyplot as plt
import visual
from pycocotools import mask as maskUtils


def annToRLE(ann, height, width):

    segm = ann['segmentation']
    if isinstance(segm, list):

        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask( ann, height, width):

    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m




def get_image(coco,map_source_class_id,class_ids,i):
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=False))
    img_url = os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/coco/raw-data/train2014',
                           coco.imgs[i]["file_name"])
    instance_masks = []
    cls_ids = []
    for annotation in annotations:
        class_id = map_source_class_id[annotation['category_id']]

        m = annToMask(annotation, coco.imgs[i]["height"],
                      coco.imgs[i]["width"])

        if m.max() < 1:
            continue

        instance_masks.append(m)
        cls_ids.append(class_id)

    img = cv2.imread(img_url)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    cls_ids = np.asarray(cls_ids)
    original_shape = img.shape

    image, window, scale, padding, crop = utils.resize_image(img,min_dim=512,max_dim=512)
    mask = np.asarray(instance_masks)

    mask = np.transpose(mask,axes=[1,2,0])
    mask = utils.resize_mask(mask, scale, padding, crop)
    boxes = utils.extract_bboxes(mask)
    mask = utils.minimize_mask(boxes, mask, mini_shape=(28, 28))
    boxes = boxes / 512.0
    return image,cls_ids,boxes,mask










