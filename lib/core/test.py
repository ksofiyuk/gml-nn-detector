# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2015 Microsoft, 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Konstantin Sofiyuk
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from core.config import cfg, get_output_dir
from core.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from core.nms_wrapper import nms
import pickle
from utils.blob import im_list_to_blob
import os
import re
import json

from datasets.collections import ImagesCollection
from datasets.iterators import DirectIterator
from core.detector_model import DetectorModel


def _get_image_blob(sample, target_size):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = sample.bgr_data.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > sample.max_size:
        im_scale = float(sample.max_size) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(sample, target_size, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(sample, target_size)

    return blobs, im_scale_factors


def fixed_scale_detect(net, model, sample, target_size, boxes=None):
    blobs, im_scales = _get_blobs(sample, target_size, boxes)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)
    rois, scores = model.extract_detections(net)

    assert len(im_scales) == 1, "Only single-image batch implemented"

    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    # if cfg.TEST.BBOX_REG:
    #     # Apply bounding-box regression deltas
    #
    #     if cfg.TEST.RPN_ONLY:
    #         pred_boxes = bbox_transform_inv(boxes, np.zeros((len(boxes), 8), dtype=np.float32))
    #     else:
    #         box_deltas = blobs_out['bbox_pred']
    #         pred_boxes = bbox_transform_inv(boxes, box_deltas)
    #         pred_boxes = clip_boxes(pred_boxes, im.shape)
    # else:
    #     # Simply repeat the boxes, once for each class
    #     pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    plt.switch_backend('Qt5Agg')

    im = im[:, :, (2, 1, 0)]

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 1:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def dense_scan_image(net, im, block_h, block_w, oratio):

    shift_h = int(block_h * 0.85)
    shift_w = int(block_w * 0.85)

    max_target_size = max(block_h, block_w)

    cur_x, cur_y = 0, 0
    scores, boxes = None, None
    while cur_y < im.shape[0]:
        cur_x = 0

        while cur_x < im.shape[1]:
            end_x = min(im.shape[1], cur_x + block_w)
            end_y = min(im.shape[0], cur_y + block_h)
            start_x = end_x - block_w
            start_y = end_y - block_h

            print(start_y, end_y, start_x, end_x)
            sub_im = im[start_y:end_y, start_x:end_x, :]
            tscores, tboxes = fixed_scale_detect(net, sub_im, max_target_size, None)

            tboxes[:, 4] += start_x
            tboxes[:, 6] += start_x
            tboxes[:, 5] += start_y
            tboxes[:, 7] += start_y

            if scores is not None:
                scores = np.vstack((scores, tscores))
                boxes = np.vstack((boxes, tboxes))
            else:
                scores, boxes = tscores, tboxes

            cur_x += shift_w
        cur_y += shift_h
    return scores, boxes


def im_detect(net, model, sample):
    max_target_size = max(sample.scales)

    im = sample.bgr_data

    scores, boxes = None, None
    if cfg.TEST.WITHOUT_UPSAMPLE and np.min(im.shape[:2]) < max_target_size:
        target_size = np.min(im.shape[:2])
        scores, boxes = fixed_scale_detect(net, model, sample, target_size)

    elif cfg.TEST.DENSE_SCAN and np.min(im.shape[:2]) > 1.5 * max_target_size:
        max_size = min(sample.max_size, np.max(im.shape[:2]))

        if im.shape[0] > im.shape[1]:
            block_h, block_w = max_size, max_target_size
        else:
            block_h, block_w = max_target_size, max_size

        shift_h = int(block_h * 0.90)
        shift_w = int(block_w * 0.90)

        print(im.shape, shift_h, shift_w)
        cur_x, cur_y = 0, 0
        while cur_y < im.shape[0]:
            cur_x = 0

            while cur_x < im.shape[1]:
                end_x = min(im.shape[1], cur_x + block_w)
                end_y = min(im.shape[0], cur_y + block_h)
                start_x = end_x - block_w
                start_y = end_y - block_h

                print(start_y, end_y, start_x, end_x)
                sub_im = im[start_y:end_y, start_x:end_x, :]
                tscores, tboxes = fixed_scale_detect(net, model, sub_im, max_target_size)

                tboxes[:, 4] += start_x
                tboxes[:, 6] += start_x
                tboxes[:, 5] += start_y
                tboxes[:, 7] += start_y

                if scores is not None:
                    scores = np.vstack((scores, tscores))
                    boxes = np.vstack((boxes, tboxes))
                else:
                    scores, boxes = tscores, tboxes

                cur_x += shift_w
            cur_y += shift_h

    for target_size in sample.scales:
        if cfg.TEST.WITHOUT_UPSAMPLE and np.min(im.shape[:2]) < target_size:
            continue

        tscores, tboxes = fixed_scale_detect(net, model, sample, target_size)
        if scores is not None:
            scores = np.vstack((scores, tscores))
            boxes = np.vstack((boxes, tboxes))
        else:
            scores, boxes = tscores, tboxes

    return scores, boxes


def to_json_format(detections, object_class):
    bboxes = []
    for det in detections:
        bbox = {'x': int(det[0]), 'y': int(det[1]),
                'w': int(det[2]-det[0]), 'h': int(det[3]-det[1]),
                'score': float(det[4]), 'class': object_class}
        bboxes.append(bbox)
    return bboxes


def test_image_collection(net, model, image_collection):
    max_per_image = cfg.TEST.MAX_PER_IMAGE
    SCORE_THRESH = 0.05

    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    all_detections = {}

    for indx, sample in enumerate(DirectIterator(image_collection)):
        image_basename = os.path.basename(sample.id)

        _t['im_detect'].tic()
        scores, boxes = im_detect(net, model, sample)
        _t['im_detect'].toc()

        _t['misc'].tic()

        json_detections = []
        for j in range(1, scores.shape[2]):
            inds = np.where(scores[:, j] > SCORE_THRESH)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]

            detections = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

            keep = nms(detections, cfg.TEST.FINAL_NMS)
            detections = detections[keep]

            json_detections.append(to_json_format(detections, j))

        all_detections[image_basename] = json_detections

        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(indx + 1, len(image_collection),
                      _t['im_detect'].average_time, _t['misc'].average_time))

    return all_detections


def test_net(weights_path, output_dir):
    model = DetectorModel(cfg.MODEL)

    fd, test_prototxt = model.create_temp_test_prototxt()
    net = caffe.Net(test_prototxt, weights_path, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(weights_path))[0]
    os.close(fd)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for indx, dataset in enumerate(cfg.TEST.DATASETS):
        image_collection = ImagesCollection(dataset)

        print("# %d/%d dataset %s: %d images" %
              (indx + 1, len(cfg.TEST.DATASETS), dataset.PATH, len(image_collection)))

        detections = test_image_collection(net, model, image_collection)

        output_path = os.path.join(output_dir, dataset.OUTPUT_FILE)
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=4)

        print('Output detections file: %s\n' % output_path)
