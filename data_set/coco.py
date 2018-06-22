from pycocotools.coco import COCO
from pycocotools import mask

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
dataDir='/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/coco/raw-data/'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
ids = list(coco.imgToAnns.keys())





img = coco.loadImgs(ids[0])[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

annIds = coco.getAnnIds(imgIds=img['id'], catIds=ids[0], iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)
coco.showAnns(anns)
plt.show()