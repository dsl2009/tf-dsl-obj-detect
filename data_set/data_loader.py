from multiprocessing import Process,Queue
from data_gen import get_batch_inception
import config
import time
gen = get_batch_inception(config.batch_size,image_size=config.image_size)

def new_get_data(quene):
    while True:
        org_im, box, label = next(gen)
        quene.put([org_im, box, label])

numT = 4
q = Queue(numT)
ps = []
for p in range(numT):
    ps.append(Process(target=new_get_data,args=(q,)))
for pd in ps:
    pd.start()

