import numpy as np
voc_vgg_300 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [37, 16, 8, 4, 2, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
voc_vgg_500 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [64, 30, 15, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [30 , 60 ,120 ,240 ,318 ,394],
    'max_sizes': [60, 120 ,240 ,318 ,394 ,470],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc_inceptionv2_dsl_500 = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [32 , 64 ,128 ,192 ,300 ,412],
    'max_sizes': [64, 128 ,192 ,300 ,412 ,556],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
shape_inceptionv2_dsl_500 = {
    'num_classes': 4,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [32 , 64 ,128 ,192 ,300 ,412],
    'max_sizes': [64, 128 ,192 ,300 ,412 ,556],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
build_inceptionv2_dsl_512 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [32 , 64 ,128 ,192 ,300 ,412],
    'max_sizes': [64, 128 ,192 ,300 ,412 ,556],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
coco_res_512 = {
    'num_classes': 81,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [32 , 64 ,128 ,192 ,300 ,412],
    'max_sizes': [64, 128 ,192 ,300 ,412 ,556],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
coco_inceptionv2_dsl_800 = {
    'num_classes': 81,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [96, 48, 24, 12, 6, 3],
    'min_dim': 768,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [32 , 64 ,128 ,192 ,300 ,412],
    'max_sizes': [64, 128 ,192 ,300 ,412 ,556],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
voc_inceptionv3_dsl_500 = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [61, 30, 15, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [32 , 64 ,128 ,192 ,300 ,412],
    'max_sizes': [64, 128 ,192 ,300 ,412 ,556],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

face_inceptionv2_dsl_500 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [30 , 89 ,165 ,241 ,318 ,394],
    'max_sizes': [60, 165 ,241 ,318 ,394 ,512],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

remove_norm = {
    'num_classes': 21,
    'feature_maps': [64, 32, 16],
    'min_dim': 512,
    'steps': [8, 16, 32],
    'min_sizes': [10 , 50 ,165 ],
    'max_sizes': [50, 165 ,446 ],
    'aspect_ratios': [[2], [2], [2]],
    'aspect_num':[9,9,9],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


voc_nana_mobile_448 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [28, 14, 7, 4, 2, 1],
    'min_dim': 448,
    'steps': [16, 32, 64, 112, 224, 448],
    'min_sizes': [30 , 89 ,165 ,241 ,318 ,394],
    'max_sizes': [60, 165 ,241 ,318 ,394 ,470],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
FACE_CLASSES = ['face']
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



MAX_GT = 100
batch_size = 8
image_size = 512
mask_pool_shape = 28
norm_value = 8.0
mask_weight_loss = 2.0
mask_train = 50
flag = 2
local_voc_dir = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit'
server_voc_dir = '/data_set/data/VOCdevkit'

local_coco_dir = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/coco/raw-data/train2014'
local_coco_ann = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/coco/raw-data/annotations/instances_train2014.json'

server_coco_dir = '/data_set/data/train2014'
server_coco_ann = '/data_set/data/annotations/instances_train2014.json'

local_check = '/home/dsl/all_check/inception_v2.ckpt'
server_check = '/data_set/check/inception_v2.ckpt'

local_save = '/home/dsl/all_check/face_detect/voc_ssd_yolo1'
server_save = '/data_set/check/voc_ssd_yolo'


if flag == 1:
    save_dir = local_save
    check_dir = local_check
    voc_dir = local_voc_dir
    coco_image_dir = local_coco_dir
    annotations = local_coco_ann
    batch_size = 8
elif flag ==2:
    save_dir = server_save
    check_dir = server_check
    voc_dir = server_voc_dir
    coco_image_dir = server_coco_dir
    annotations = server_coco_ann
    batch_size = 32

Config = remove_norm


