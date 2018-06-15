import os
import json
ds = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/wider_face_split/wider_face_train_bbx_gt.txt'
root = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/face_detect/WIDER_train/images/'

image = dict()

last_tile = None
for s in open(ds).readlines():
    d = s.replace('\n','')
    if '.jpg' in d:
        last_tile = os.path.join(root,d)
        image[last_tile] = []


    elif len(d)>5:

        dd = [int(x) for x in d.split(' ')[0:4]]
        print(dd)
        x1 = dd[0]
        y1 = dd[1]
        x2 = dd[0]+dd[2]
        y2 = dd[3] + dd[1]
        image[last_tile].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

nmap = json.loads(open('face.json').read())
for k in nmap:
    image[k] = nmap[k]
    print(k)
    print(nmap[k])
finnal = dict()
for k in image:
    if len(image[k])<8:
        finnal[k] = image[k]


with open('face1.json', 'w') as f:
    f.write(json.dumps(finnal))
    f.flush()

