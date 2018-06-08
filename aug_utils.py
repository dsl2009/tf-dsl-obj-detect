from imgaug import augmenters as igg
import cv2
import imgaug
from matplotlib import pyplot as plt


def fliplr_left_right(img, box):
    img = igg.Fliplr(1.0).augment_image(img)
    x1,x2 = box[:,0],box[:,2]
    nx1 = 1-x2
    nx2 = 1-x1
    box[:,0] = nx1
    box[:,2] = nx2
    return img,box


