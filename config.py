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
    'min_sizes': [30 , 89 ,165 ,241 ,318 ,394],
    'max_sizes': [60, 165 ,241 ,318 ,394 ,470],
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
    'min_sizes': [30 , 89 ,165 ,241 ,318 ,394],
    'max_sizes': [60, 165 ,241 ,318 ,394 ,470],
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
    'max_sizes': [60, 165 ,241 ,318 ,394 ,470],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_num':[4,6,6,6,4,4],
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
MAX_GT = 100
batch_size = 8
Config = voc_inceptionv2_dsl_500


