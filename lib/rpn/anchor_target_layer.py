# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

import os
import caffe
import json
from core.config import cfg
import numpy as np
import numpy.random as npr
from .generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from core.bbox_transform import bbox_transform
from sklearn.cluster import KMeans


DEBUG = False
ANCHORS_SCALES_DEBUG = False


class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        layer_params = json.loads(self.param_str)
        self._feat_stride = layer_params['feat_stride']
        self._iters = 0
        self._periodic_tn_enabled = True
        self._square_targets = layer_params['square_targets']
        self._square_targets_ky = layer_params.get('square_targets_ky', 0.5)

        self._positive_overlap = layer_params['positive_overlap']
        self._negative_overlap = layer_params['negative_overlap']
        self._batchsize = layer_params['batchsize']
        self._dynamic_batchsize = self._batchsize == 'dynamic'

        self._max_tn_fraction = layer_params['tn_fraction']
        self._fg_fraction = layer_params['fg_fraction']
        self._name = layer_params['name']
        self._num_classes = layer_params['num_classes']
        self._ratios = layer_params['anchor_ratios']

        assert self._square_targets and len(self._ratios) == 1

        self._anchors = generate_anchors(base_size=self._feat_stride,
                                         ratios=layer_params['anchor_ratios'],
                                         scales=layer_params['anchor_scales'],
                                         shift_num_xy=layer_params['anchor_shift_num_xy'])

        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print('anchors:')
            print(self._anchors)
            print('anchor shapes:')
            print(np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            )))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))

        if ANCHORS_SCALES_DEBUG:
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)
        self._top_neg_fraction = cfg.TRAIN.RPN_LINEAR_START_TNF

        self._gt_stat = []

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print('AnchorTargetLayer: height', height, 'width', width)

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        assert bottom[0].shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].shape[-2:]

        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        gt_ignored_boxes = bottom[2].data
        im_info = bottom[3].data[0, :]

        if DEBUG:
            print('')
            print('class_distrib', gt_boxes[:, 4])
            print('im_info: ({}, {}, {})'.format(im_info[0], im_info[1], im_info[2]))
            print('height, width: ({}, {})'.format(height, width))

        all_anchors = self._get_all_anchors(bottom)
        valid_anchors, valid_anchors_inds = self._get_valid_anchors(all_anchors, im_info)
        num_total_anchors = all_anchors.shape[0]
        num_valid_anchors = valid_anchors.shape[0]

        labels, bbox_targets = self._compute_fg_labels_and_targets(valid_anchors,
                                                                   gt_boxes, gt_ignored_boxes)

        if cfg.TRAIN.RPN_LINEAR_TNF_K > 0:
            self._top_neg_fraction = min(self._top_neg_fraction + cfg.TRAIN.RPN_LINEAR_TNF_K, #0.00008
                                         self._max_tn_fraction)
        else:
            self._top_neg_fraction = self._max_tn_fraction

        if cfg.TRAIN.RPN_PERIODIC_TN > 0 and self._iters % cfg.TRAIN.RPN_PERIODIC_TN == 0:
            self._periodic_tn_enabled = not self._periodic_tn_enabled
            if self._periodic_tn_enabled:
                print('Switch on top negatives with fraction %.6f' % self._top_neg_fraction)
            else:
                print('Switch off top negatives')

        if not self._periodic_tn_enabled:
            self._top_neg_fraction = 0

        if self._iters % 250 == 0:
            print('Current TNF: %.6f' % self._top_neg_fraction)

        if self._dynamic_batchsize:
            num_bg = int(round(np.sum(labels >= 1) * (1 / self._fg_fraction - 1)))
            num_bg = max(num_bg, 4)
        else:
            num_bg = self._batchsize - np.sum(labels >= 1)

        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            keep_first = int(np.floor(num_bg * self._top_neg_fraction))

            if keep_first > 0:
                # scores are (1, A, H, W) format
                # transpose to (1, H, W, A)
                # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)

                scores = np.zeros((self._num_anchors * bottom[5].shape[2] * bottom[5].shape[3], 1))

                for class_id in range(1, self._num_classes + 1):
                    indx_from = class_id * self._num_anchors
                    indx_to = indx_from + self._num_anchors
                    tmp = bottom[5].data[:, indx_from:indx_to, :, :]
                    tmp = tmp.transpose((0, 2, 3, 1)).reshape((-1, 1))
                    scores += tmp

                scores = scores[valid_anchors_inds]
                order = scores[bg_inds].ravel().argsort()[::-1]
                bg_inds = bg_inds[order]

            disable_inds = npr.choice(
                bg_inds[keep_first:], size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

        bbox_inside_weights = np.zeros((num_valid_anchors, 4), dtype=np.float32)
        bbox_inside_weights[labels >= 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((num_valid_anchors, 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels >= 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels >= 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels >= 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels >= 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels >= 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print('means:')
            print(means)
            print('stdevs:')
            print(stds)

        # map up to original set of anchors
        labels = _unmap(labels, num_total_anchors, valid_anchors_inds, fill=-1)
        bbox_targets = _unmap(bbox_targets, num_total_anchors, valid_anchors_inds, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, num_total_anchors, valid_anchors_inds, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, num_total_anchors, valid_anchors_inds, fill=0)

        if DEBUG:
            print(self._name + ': num_positive', np.sum(labels >= 1))
            print(self._name + ': num_negative', np.sum(labels == 0))

        if ANCHORS_SCALES_DEBUG:
            self._fg_sum += np.sum(labels >= 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            if self._iters % 100 == 0:
                print(self._name + ': num_positive avg', self._fg_sum / self._count)
                print(self._name + ': num_negative avg', self._bg_sum / self._count)

        num_anchors = self._num_anchors
        # labels
        # print(np.unique(labels))
        # labels[labels >= 1] = 1
        labels = labels.reshape((1, height, width, num_anchors)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, num_anchors * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, num_anchors * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, num_anchors * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, num_anchors * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

        self._iters += 1

    def _get_all_anchors(self, bottom):
        # 1. Generate proposals from bbox deltas and shifted anchors

        height, width = bottom[0].shape[-2:]
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]

        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        if DEBUG:
            print('num_total_anchors', all_anchors.shape[0])

        return all_anchors

    def _get_valid_anchors(self, anchors, im_info):
        # only keep anchors inside the image
        valid_anchors_inds = np.where(
            (anchors[:, 0] >= -self._allowed_border) &
            (anchors[:, 1] >= -self._allowed_border) &
            (anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        # keep only inside anchors
        valid_anchors = anchors[valid_anchors_inds, :]

        if DEBUG:
            print('num_valid_anchors', len(valid_anchors_inds))

        return valid_anchors, valid_anchors_inds

    def _compute_fg_labels_and_targets(self, valid_anchors, gt_boxes, ignored_boxes):
        num_valid_anchors = valid_anchors.shape[0]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((num_valid_anchors,), dtype=np.float32)
        labels.fill(-1)

        if gt_boxes.shape[0]:
            boxes = _square_boxes(gt_boxes, self._ratios[0], self._square_targets_ky) \
                if self._square_targets else gt_boxes

            # overlaps.shape = (num_valid_anchors, num_gt_boxes)
            overlaps = bbox_overlaps(
                np.ascontiguousarray(valid_anchors, dtype=np.float),
                np.ascontiguousarray(boxes, dtype=np.float))

            argmax_overlaps = overlaps.argmax(axis=1)

            if ANCHORS_SCALES_DEBUG:
                gt_max_overlaps = overlaps.max(axis=0)
                gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                gt_anchors_count = np.sum(overlaps >= self._positive_overlap, axis=0)
                for w, h, overlap, cnt in zip(gt_w, gt_h, gt_max_overlaps, gt_anchors_count):
                    self._gt_stat.append([w, h, overlap, cnt])

                if self._iters % 250 == 249:
                    self._print_gt_stat()

            max_overlaps = overlaps[np.arange(num_valid_anchors), argmax_overlaps]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < self._negative_overlap] = 0

            obj_classes = gt_boxes[argmax_overlaps, 4]
            overlap_mask = max_overlaps >= self._positive_overlap
            labels[overlap_mask] = obj_classes[overlap_mask]

            if DEBUG:
                print(self._name + ': max max_overlap',
                      np.max(max_overlaps) if 'max_overlaps' in locals() else 0)

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < self._negative_overlap] = 0
        else:
            labels.fill(0)

        # ignored label
        if len(ignored_boxes):
            boxes = _square_boxes(ignored_boxes, self._ratios[0], self._square_targets_ky) \
                if self._square_targets else ignored_boxes
            ignored_overlaps = bbox_overlaps(
                np.ascontiguousarray(valid_anchors, dtype=np.float),
                np.ascontiguousarray(boxes, dtype=np.float))

            iargmax_overlaps = ignored_overlaps.argmax(axis=1)
            imax_overlaps = ignored_overlaps[np.arange(num_valid_anchors), iargmax_overlaps]
            labels[imax_overlaps > 0.3] = -1

        # subsample positive labels if we have too many
        fg_inds = np.where(labels >= 1)[0]
        if self._dynamic_batchsize:
            num_fg = min(max(len(fg_inds), 2), 256)
        else:
            num_fg = int(self._fg_fraction * self._batchsize)

        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((num_valid_anchors, 4), dtype=np.float32)
        if gt_boxes.shape[0]:
            bbox_targets = _compute_targets(valid_anchors, gt_boxes[argmax_overlaps, :])

        return labels, bbox_targets

    def _print_gt_stat(self):
        gt_stat = np.array(self._gt_stat)
        features = gt_stat[:, :2]

        kmeans = KMeans(n_clusters=8)
        labels = kmeans.fit_predict(features)
        num_clusters = max(labels) + 1

        clusters = []
        for i in range(num_clusters):
            gt_objects = gt_stat[labels == i, :]
            min_w = gt_objects[:, 0].min()
            min_h = gt_objects[:, 1].min()
            max_w = gt_objects[:, 0].max()
            max_h = gt_objects[:, 1].max()
            mean_w = gt_objects[:, 0].mean()
            mean_h = gt_objects[:, 1].mean()
            num_objects = gt_objects.shape[0]
            mean_overlap = gt_objects[:, 2].mean()
            mean_cnt = gt_objects[:, 3].mean()

            clusters.append([i, num_objects, mean_w, mean_h, min_w, max_w, mean_overlap, mean_cnt])

        clusters = sorted(clusters, key=lambda x: x[2])

        print('\n' + '-' * 80 + '\n' + self._name + '\n' + '-' * 80)
        for i, num_objects, mean_w, mean_h, min_w, max_w, mean_overlap, mean_cnt in clusters:
            # if mean_overlap < 0.4:
                # continue
            print('')
            print('Cluster', i, 'num:', num_objects)
            print('mean_w:', mean_w, 'mean_h:', mean_h, 'min_w', min_w, 'max_w', max_w)
            print('mean_overlap:', mean_overlap, 'mean_cnt:', mean_cnt)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def _square_boxes(boxes, ratio, ky=0.5):
    if boxes.shape[0] == 0:
        return boxes

    ret = boxes.copy()
    gt_sz = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]) / ratio)
    gt_cntr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    gt_cntr_y = ky * (boxes[:, 1] + boxes[:, 3])
    ret[:, 0] = gt_cntr_x - gt_sz * 0.5
    ret[:, 1] = gt_cntr_y - gt_sz * 0.5 * ratio
    ret[:, 2] = gt_cntr_x + gt_sz * 0.5
    ret[:, 3] = gt_cntr_y + gt_sz * 0.5 * ratio
    return ret
