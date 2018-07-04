from sklearn.cluster import KMeans
from data_voc import  VOCDetection
import numpy as np
from matplotlib import pyplot as plt
data_set = VOCDetection('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit')
length = data_set.len()
data = []
for ix in range(length):
    ig, box, ids = data_set.pull_item(ix)
    box = box*512
    w = box[:,2:] - box[:,0:2]

    x = w[:,0]*w[:,1]
    for s in  np.sqrt(x):

        data.append(s)

data = np.asarray(data)
data = np.reshape(data,(-1,1))
model = KMeans(n_clusters=9).fit(data)
print(model.cluster_centers_)
