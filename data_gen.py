from data_voc import VOCDetection
from matplotlib import pyplot as plt
from data_set import face
import aug_utils
import random
import numpy as np
import visual
data_set = VOCDetection('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit')
from data_set.augmentations import SSDAugmentation
from data_set.shapes import get_image
#data_set = face.Face(root='/home/dsl/PycharmProjects/tf-ssd/data_set/face1.json',image_size=512)
from pycocotools.coco import COCO
from data_set.coco import get_image as get_coco_image


def get_batch(batch_size,is_shuff = True,max_detect = 50,image_size=300):
    length = data_set.len()
    idx = list(range(length))
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
                print(idx)

            img, box, lab = data_set.pull_item(idx[index])
            #img = img/255.0
            #img = img - 0.5
            #img = img * 2.0
            if random.randint(0,1)==1:
                img, box = aug_utils.fliplr_left_right(img,box)
            img = (img.astype(np.float32) - np.array([123.7, 116.8, 103.9]))/255
            if b== 0:

                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                index=index+1

                b=b+1

            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                index = index + 1
                b = b + 1


            if b>=batch_size:
                yield [images,boxs,label]
                b = 0
            if index>= length:
                index = 0

def get_batch_inception(batch_size,is_shuff = True,max_detect = 50,image_size=300):
    length = data_set.len()
    idx = list(range(length))
    b = 0
    index = 0
    aug = SSDAugmentation(image_size)
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
                print(idx)

            img, box, lab = data_set.pull_item(idx[index])
            if True:

                if random.randint(0,1)==1:
                   img, box = aug_utils.fliplr_left_right(img,box)
                img = img/255.0
                img = img - 0.5
                img = img * 2.0

            else:
                img, box, lab = aug(img,box,lab)
                img = ((img + [104, 117, 123])/255-0.5)*2.0


            #img = (img.astype(np.float32) - np.array([123.7, 116.8, 103.9]))/255
            if b== 0:

                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                index=index+1

                b=b+1

            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                index = index + 1
                b = b + 1


            if b>=batch_size:
                yield [images,boxs,label]
                b = 0
            if index>= length:
                index = 0

def get_batch_shapes(batch_size,is_shuff = True,max_detect = 50,image_size=512):

    b = 0
    while True:
        if True:
            img, lab, box, mask = get_image()
            img = img/255.0
            img = img - 0.5
            img = img * 2.0
            #img = (img.astype(np.float32) - np.array([123.7, 116.8, 103.9]))/255
            if b== 0:
                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                masks = np.zeros(shape=[batch_size, 56, 56,max_detect], dtype=np.int)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                masks[b, :, :,:box.shape[0]] = mask
                b=b+1

            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                masks[b, :, :, :box.shape[0]] = mask
                b = b + 1


            if b>=batch_size:
                yield [images,boxs,label,masks]
                b = 0



def get_coco(batch_size,is_shuff = True,max_detect = 50,image_size=512,mask_shape=28):
    coco = COCO('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/coco/raw-data/annotations'
                '/instances_train2014.json')
    class_ids = sorted(coco.getCatIds())

    image_ids = list(coco.imgs.keys())
    map_source_class_id = dict(zip(class_ids, range(len(class_ids))))
    length = len(image_ids)
    idx = list(range(length))
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img, lab, box, mask = get_coco_image(coco,map_source_class_id,class_ids,image_ids[idx[index]],mask_shape,image_size)
            except:
                index = index+1
                print(index)
                continue
            img = img/255.0
            img = img - 0.5
            img = img * 2.0
            #img = (img.astype(np.float32) - np.array([123.7, 116.8, 103.9]))/255
            if b== 0:
                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                masks = np.zeros(shape=[batch_size, mask_shape, mask_shape,max_detect], dtype=np.int)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                masks[b, :, :,:box.shape[0]] = mask
                b=b+1
                index = index + 1

            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                masks[b, :, :, :box.shape[0]] = mask
                b = b + 1
                index = index + 1

            if b>=batch_size:
                yield [images,boxs,label,masks]
                b = 0

            if index>= length:
                index = 0


def tt():
    gen = get_coco(batch_size=8,max_detect=50)
    for s in range(10):
        next(gen)
