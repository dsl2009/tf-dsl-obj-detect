from data_voc import VOCDetection
from matplotlib import pyplot as plt
from data_set import face
import aug_utils
import random
import numpy as np
import visual
data_set = VOCDetection('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit')


#data_set = face.Face(root='data_set/face1.json',image_size=512)




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
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
                print(idx)

            img, box, lab = data_set.pull_item(idx[index])
            if random.randint(0,1)==1:
                img, box = aug_utils.fliplr_left_right(img,box)

            img = img/255.0
            img = img - 0.5
            img = img * 2.0

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

def tt():
    for s in range(1000):
        img, box, lab = data_set.pull_item(s)
        print(box)
        visual.display_instances(img, box * 512)

