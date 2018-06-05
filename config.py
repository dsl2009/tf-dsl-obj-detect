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
MAX_GT = 100