import os

import cv2
import numpy as np
import json
import glob
from utils import resize_image

def handler():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/FDDB-folds/*ellipseList.txt'
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/originalPics'
    image = dict()

    last_tile = None
    for s in glob.glob(rt):

        with open(s) as f:

            for sl in f.readlines():

                if 'img' in sl:
                    im_pth = os.path.join(image_dr, sl.replace('\n', '.jpg'))
                    last_tile = im_pth
                    image[last_tile] = []
                elif len(sl) > 10:
                    dd = [float(x) for x in sl.split(' ')[0:-2]]
                    x1 = max(int(dd[3] - dd[1]), 0)
                    y1 = max(int(dd[4] - dd[0]), 0)
                    x2 = int(dd[3] + dd[1])
                    y2 = int(dd[4] + dd[0])
                    image[last_tile].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    with open('face.json', 'w') as f:
        f.write(json.dumps(image))
        f.flush()

class AugAnnotationTransform(object):


    def __init__(self, image_size=416):

        self.image_size = image_size

    def __call__(self, img, box,label):
        image, window, scale, padding, crop = resize_image(img,min_dim=self.image_size,max_dim=self.image_size)


        box = box*scale
        box[:, 0] = box[:, 0] + padding[1][0]
        box[:, 1] = box[:, 1] +  padding[0][0]
        box[:, 2] = box[:, 2] + padding[1][1]
        box[:, 3] = box[:, 3] + padding[0][1]

        box = box/self.image_size
        return image,box,label

class Face(object):


    def __init__(self,root,image_size):
        self.data = json.loads(open(root).read())
        self.ids = list(self.data.keys())
        self.ids = self.ids[0:len(self.ids)-len(self.ids)%8]
        self.transform = AugAnnotationTransform(image_size)
        self.name = 'FACE'

    def len(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]


        img = cv2.imread(img_id)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        target = []
        for s in self.data[img_id]:
            target.append([s['x1'],s['y1'],s['x2'],s['y2'],0])

        target = np.array(target)

        img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])



        return img, boxes, labels







