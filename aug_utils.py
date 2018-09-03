from imgaug import augmenters as igg
import cv2
import imgaug
from matplotlib import pyplot as plt
import skimage
import tensorflow as tf

import numpy as np
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def fliplr_left_right(img, box):
    img = igg.Fliplr(1.0).augment_image(img)

    x1,x2 = box[:,0],box[:,2]
    nx1 = 1-x2
    nx2 = 1-x1
    box[:,0] = nx1
    box[:,2] = nx2
    return img,box

def fliplr_up_down(img, box):
    img = igg.Flipud(1.0).augment_image(img)
    y1, y2 = box[:, 1], box[:, 3]
    ny1 = 1 - y2
    ny2 = 1 - y1
    box[:, 1] = ny1
    box[:, 3] = ny2
    return img,box