# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------
from detector_nets.rpn import RPN


def create_network(config):
    if config.TYPE == "RPN":
        return RPN(config)
    else:
        raise NotImplementedError(config.TYPE)