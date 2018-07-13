from data_gen import get_batch_inception
from matplotlib import pyplot as plt
import visual
import numpy as np
gen = get_batch_inception(batch_size=1,image_size=512)

img, box, cls = next(gen)
print(img)
img = img[0]
box = box[0]
img = np.asarray(img*255,np.int)
print(img)
visual.display_instances(img,box*512)