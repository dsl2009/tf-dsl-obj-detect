from data_set.data_loader import q
import time
for s in range(100):
    t = time.time()
    ig, bb, ll = q.get()
    print(time.time()-t)
    time.sleep(0.025*8)