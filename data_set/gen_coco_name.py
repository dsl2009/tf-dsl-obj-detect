namess = []
with open('/home/dsl/coco_labels.txt') as f:
    names = f.readlines()
    for n in names:
        namess.append(n.replace('\n','').split(',')[-1])
print(namess)
