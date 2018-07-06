import os
import cv2
import glob
import shutil
import random
import numpy as np
import json
import visual
import utils
from matplotlib import pyplot as plt
def gen_train_val(root='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/mask_rcnn/tt'):
    train_file = os.path.join(root,'train.txt')
    val_file = os.path.join(root,'val.txt')
    f_train = open(train_file, 'w')
    f_val = open(val_file, 'w')
    for img in glob.glob(os.path.join(root,'*.png')):
        na = img.split('/')[-1].split('.')[0]
        js = glob.glob(os.path.join(root,na+'*.json'))
        if len(js)==0:
            continue
        shutil.move(js[0],os.path.join(root,na+'.json'))
        if random.randint(1, 10) > 10:
            f_val.write(img.split('/')[-1].split('.')[0] + '\n')
            f_val.flush()
        else:
            f_train.write(img.split('/')[-1].split('.')[0] + '\n')
            f_train.flush()

class Tree(object):
    def __init__(self,root,image_size,mask_pool_size):
        self.root = root
        self.image_size = image_size
        self.mask_pool_size = mask_pool_size
        self.ids = open(os.path.join(root,'train.txt')).readlines()
        self.index = range(len(self.ids))

    def pull_item(self,ind):

        img_dr = os.path.join(self.root, self.ids[ind].replace('\n', '') + '.png')
        json_dr = os.path.join(self.root, self.ids[ind].replace('\n', '') + '.json')

        dts = json.loads(open(json_dr).read())

        total = len(dts)

        ig = cv2.imread(img_dr)
        ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        shape = ig.shape[0:2]

        msk = np.zeros((self.image_size, self.image_size, total))
        ids = []
        for idx, s in enumerate(dts):
            tm_img = np.zeros((shape[0], shape[1], 3))
            center = [s['x'], s['y']]
            r = s['radius']
            tm_img = cv2.circle(tm_img, center=tuple(center), radius=r, color=(255, 255, 255), thickness=-1)

            tm_img = cv2.resize(tm_img,dsize=(self.image_size,self.image_size))

            msk[:, :, idx] = tm_img[:, :, 0]
            ids.append(0)


        ig = cv2.resize(ig,dsize=(self.image_size,self.image_size))
        box = utils.extract_bboxes(msk)
        mask = utils.minimize_mask(box, msk, mini_shape=(self.mask_pool_size, self.mask_pool_size))
        box = box * 1.0 / self.image_size
        #visual.display_instances(ig,box*self.image_size)

        return ig,ids,box,mask

data_set = Tree(root='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/mask_rcnn/t1',
                    image_size=512,
                    mask_pool_size=56)
def get_image():

    x = random.randint(0,len(data_set.ids)-1)
    return data_set.pull_item(x)





