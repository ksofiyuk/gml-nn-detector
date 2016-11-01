# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

"""Config system.

This file specifies default config options for detector. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# RPN options
#
__C.RPN = edict()
__C.RPN.ANCHOR_BASE_SIZE = 16
__C.RPN.ANCHOR_RATIOS = (0.5, 1, 2)
__C.RPN.ANCHOR_SCALES = (2 ** np.arange(3, 6)).tolist()
__C.RPN.ANCHOR_SHIFT_NUM_XY = [(1,1)]


#
# Model options
#
__C.MODEL = edict()
__C.MODEL.NAME = "unnamed"
__C.MODEL.WEIGHTS_PATH = ''
__C.MODEL.PRETRAINED_MODEL_CONFIG = ''
__C.MODEL.ATTACHED_NETS = []


#
# Training options
#

__C.TRAIN = edict()

__C.TRAIN.SOLVER = edict()
__C.TRAIN.SOLVER.IMS_PER_BATCH = 2
__C.TRAIN.SOLVER.TRAIN_ITERS = 0
__C.TRAIN.SOLVER.BASE_LR = 0.001

__C.TRAIN.SOLVER.LR_POLICY = edict()
__C.TRAIN.SOLVER.LR_POLICY.TYPE = "fixed"
__C.TRAIN.SOLVER.LR_POLICY.GAMMA = 0.1
__C.TRAIN.SOLVER.LR_POLICY.STEPS = []

__C.TRAIN.SOLVER.WEIGHT_DECAY = 0.0005
__C.TRAIN.SOLVER.MOMENTUM = 0.9

__C.TRAIN.SOLVER.DISPLAY = edict()
__C.TRAIN.SOLVER.DISPLAY.PERIOD = 20
__C.TRAIN.SOLVER.DISPLAY.AVERAGE_LOSS = 100


__C.TRAIN.DATASETS = []

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

__C.TRAIN.GENERATED_FRACTION = 0.0

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5


__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'selective_search'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = False
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

__C.TRAIN.RPN_PERIODIC_TN = 0
__C.TRAIN.RPN_LINEAR_TNF_K = -1.0
__C.TRAIN.RPN_LINEAR_START_TNF = 0.1
__C.TRAIN.RPN_TOP_NEGATIVE_FRACTION = 0.0
__C.TRAIN.RPN_SQUARE_TARGETS = False

__C.TRAIN.DATASET = 'default'

# Каждое изображение при обучении будет случайно масштабироваться относительно
# целевого размера, так что размер наименьшей стороны будет лежать
# в отрезке [(1 - SCALE_JITTERING/100)*SCALE, (1 + SCALE_JITTERING/100)*SCALE]
__C.TRAIN.SCALE_JITTERING = 0.0

__C.TRAIN.DOUBLE_GENERATE = False
__C.TRAIN.REDISTRIBUTE_CLASSES = False
__C.TRAIN.USE_LMDB = False

__C.TRAIN.ENABLE_SMART_ORDER = False
__C.TRAIN.SO_GOOD_SKIP_PROB = 0.8
__C.TRAIN.SO_TAIL_LEN = 1000
__C.TRAIN.SO_FORCE_BATCHSIZE = 200
__C.TRAIN.SO_FORCE_ROUNDS = 5

#
# Testing options
#
__C.TEST = edict()

__C.TEST.DATASETS = []
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.WITHOUT_UPSAMPLE = False
__C.TEST.DENSE_SCAN = False

__C.TEST.MAX_PER_IMAGE = 100

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3
__C.TEST.FINAL_NMS = 0.5

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Get detections only from RPN
__C.TEST.RPN_ONLY = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'selective_search'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Draw train and test net
__C.DRAW_NET = False

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models'))

# Model weights path
__C.WEIGHTS_PATH = 'default'

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0


def get_output_dir(suffix, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'exps', __C.EXP_DIR, 'output', suffix))
    if net is None:
        return path
    else:
        return osp.join(path, net.name)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
