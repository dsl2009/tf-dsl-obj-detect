import numpy as np
import config
import math
from itertools import product as product
import data_gen
from util import gen_chors
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[:,2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:,:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """


    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]

    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]))  # [A,B]

    union = area_a + area_b - inter
    return inter / union  # [A,B]

def get_prio_box(cfg = config.voc_vgg_300):
    mean = []
    for k, f in enumerate(cfg['feature_maps']):
        for i, j in product(range(f), repeat=2):
            f_k = cfg['min_dim'] / cfg['steps'][k]
            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = cfg['min_sizes'][k] / cfg['min_dim']
            mean.append([cx, cy, s_k, s_k])


            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (cfg['max_sizes'][k] / cfg['min_dim']))
            mean.append([cx, cy, s_k_prime, s_k_prime])

            # rest of aspect ratios
            for ar in cfg['aspect_ratios'][k]:
                mean.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                mean.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])

                # back to torch land
    out = np.asarray(mean,dtype=np.float32)

    out = np.clip(out,a_min=0.0,a_max=1.0)
    return out


def get_prio_box_new(cfg=config.voc_vgg_300):
    mean = []
    for k, f in enumerate(cfg['feature_maps']):
        for i, j in product(range(f), repeat=2):
            f_k = cfg['min_dim'] / cfg['steps'][k]

            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = cfg['min_sizes'][k] / cfg['min_dim']
            mean.append([cx, cy, s_k, s_k])

            mean.append([cx, cy, s_k * 1.2, s_k * 1.2])
            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (cfg['max_sizes'][k] / cfg['min_dim']))
            mean.append([cx, cy, s_k_prime, s_k_prime])

            # rest of aspect ratios
            for ar in cfg['aspect_ratios'][k]:
                mean.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                mean.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])

                mean.append([cx, cy, s_k * math.sqrt(ar) * 1.2, s_k / math.sqrt(ar) * 1.2])
                mean.append([cx, cy, s_k / math.sqrt(ar) * 1.2, s_k * math.sqrt(ar) * 1.2])

                # back to torch land
    out = np.asarray(mean, dtype=np.float32)

    out = np.clip(out, a_min=0.0, a_max=1.0)
    return out

def over_laps(boxa,boxb):
    A = boxa.shape[0]
    B = boxb.shape[0]
    b_box = np.expand_dims(boxb,0)
    a = np.repeat(boxa, repeats=B, axis=0)
    b = np.reshape(np.repeat(b_box, repeats=A, axis=0), newshape=(-1, 4))
    d = jaccard_numpy(a, b)
    return np.reshape(d,newshape=(A,B))

def pt_from(boxes):
    xy_min = boxes[:, :2] - boxes[:, 2:] / 2
    xy_max = boxes[:, :2] + boxes[:, 2:] / 2
    return np.hstack([xy_min,xy_max])

def encode(matched, priors, variances):


    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]

    g_wh = np.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return np.hstack([g_cxcy, g_wh])  # [num_priors,4]


def get_loc_conf(true_box, true_label,batch_size = 4,cfg = config.voc_vgg_300):
    #pri = get_prio_box_new(cfg = cfg)
    pri = gen_chors.gen_ssd_anchors()
    num_priors = pri.shape[0]
    loc_t = np.zeros([batch_size, num_priors, 4])
    conf_t = np.zeros([batch_size, num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]


        ix = np.sum(true_box_tm, axis=1)
        true_box_tm = true_box_tm[np.where(ix > 0)]
        labels = labels[np.where(ix > 0)]





        ops = over_laps(true_box_tm, pt_from(pri))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)


        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)

        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j




        matches = true_box_tm[best_true_idx]


        conf = labels[best_true_idx] + 1


        conf[best_true < 0.5] = 0




        loc = encode(matches, pri, variances=[0.1, 0.2])

        loc_t[s] = loc
        conf_t[s] = conf
    return loc_t,conf_t





def get_loc_conf_mask(true_box, true_label,batch_size = 4,cfg = config.voc_vgg_300):

    pri = get_prio_box(cfg = cfg)
    num_priors = pri.shape[0]
    loc_t = np.zeros([batch_size, num_priors, 4])
    conf_t = np.zeros([batch_size, num_priors])
    mask_index = np.zeros([batch_size,num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]



        ix = np.sum(true_box_tm, axis=1)
        true_box_tm = true_box_tm[np.where(ix > 0)]
        labels = labels[np.where(ix > 0)]

        mask_ix = np.asarray(np.arange(0, labels.shape[0]))



        ops = over_laps(true_box_tm, pt_from(pri))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)


        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)

        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j

        matches = true_box_tm[best_true_idx]
        mask_t = mask_ix[best_true_idx]+1

        conf = labels[best_true_idx] + 1


        conf[best_true < 0.5] = 0

        mask_t[best_true < 0.5] = 0


        loc = encode(matches, pri, variances=[0.1, 0.2])

        loc_t[s] = loc
        conf_t[s] = conf
        mask_index[s] = mask_t
    return loc_t,conf_t,mask_index











