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




def get_image(coco,map_source_class_id,class_ids,i,mask_shape,image_size):
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

    image, window, scale, padding, crop = utils.resize_image(img,min_dim=image_size,max_dim=image_size)
    mask = np.asarray(instance_masks)

    mask = np.transpose(mask,axes=[1,2,0])
    mask = utils.resize_mask(mask, scale, padding, crop)
    image,mask = aug(image,mask)

    boxes = utils.extract_bboxes(mask)
    mask = utils.minimize_mask(boxes, mask, mini_shape=(mask_shape, mask_shape))
    boxes = boxes*1.0 / image_size
    return image,cls_ids,boxes,mask



def aug(image,mask):
    augmentation = imgaug.augmenters.Fliplr(0.5)

    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]

    def hook(images, augmenter, parents, default):

        return (augmenter.__class__.__name__ in MASK_AUGMENTERS)


    image_shape = image.shape
    mask_shape = mask.shape

    det = augmentation.to_deterministic()
    image = det.augment_image(image)

    mask = det.augment_image(mask.astype(np.uint8),
                             hooks=imgaug.HooksImages(activator=hook))

    assert image.shape == image_shape, "Augmentation shouldn't change image size"
    assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

    mask = mask.astype(np.bool)
    return image,mask





