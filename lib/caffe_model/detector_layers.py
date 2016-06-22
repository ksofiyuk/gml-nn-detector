# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------

from caffe_model.layers import PythonLayer


class AnchorTargetLayer(PythonLayer):
    def __init__(self, name, anchors_params, layer_params, feat_stride=None):

        self._layer_params = {'feat_stride': feat_stride}
        self._layer_params.update(anchors_params)
        self._layer_params.update(layer_params)
        super(AnchorTargetLayer, self).__init__(name, 'rpn.anchor_target_layer.AnchorTargetLayer',
                                                self._layer_params, 6, 4)
        self._dynamic_params = ['stride']

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()

        self._layer_params['feat_stride'] = value
        self.update_layer_params(self._layer_params)

    def slots_out_names(self):
        return ['labels', 'bbox_targets', 'inside_weights', 'outside_weights']


class ProposalLayer(PythonLayer):
    def __init__(self, name, anchors_params, min_size=16,
                 pre_nms_topN=12000, post_nms_topN=2000, nms_thresh=0.7,
                 feat_stride=None):

        self._layer_params = {'feat_stride': feat_stride, 'min_size': min_size,
                              'pre_nms_topN': pre_nms_topN,
                              'post_nms_topN': post_nms_topN,
                              'nms_thresh': nms_thresh}

        self._layer_params.update(anchors_params)
        super(ProposalLayer, self).__init__(name, 'rpn.proposal_layer.ProposalLayer',
                                            self._layer_params, 3, 2)
        self._dynamic_params = ['stride']

    def set_dynamic_param(self, param, value):
        if param not in self._dynamic_params:
            raise ValueError()

        self._layer_params['feat_stride'] = value
        self.update_layer_params(self._layer_params)

    def slots_out_names(self):
        return ['rois', 'rois_scores']