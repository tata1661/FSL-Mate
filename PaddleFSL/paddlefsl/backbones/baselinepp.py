# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
from paddle.nn.utils import weight_norm
import paddle


class distLinear(nn.Layer):
    """
    classifer of baselinepp
    Args:
        indim: dim of the input feature
        outdim: dim of the output
    """

    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.L = nn.Linear(indim, outdim, bias_attr=False)
        # split the weight update component to direction and nor
        weight_norm(self.L, 'weight', dim=0)
        # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        self.scale_factor = 2

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_norm = paddle.norm(x, p=2, axis=1).unsqueeze(1).expand_as(x)
        x_normalized = x.divide(x_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)
        return scores


class Baselinepp(nn.Layer):
    """
    Baselinepp Module for memory network of MatchingNet.
    Args:
        encoder: backbone of the baselinepp
        num_class:how many class need to classify
        encoder_findim: the dim of encoder output
    """

    def __init__(self, encoder, num_class, encoder_findim):
        super(Baselinepp, self).__init__()
        self.encoder = encoder
        self.encoder_findim = encoder_findim
        self.num_class = num_class
        self.classifier = distLinear(encoder_findim, num_class)
        self.num_class = num_class

    def forward(self, x):
        out = self.encoder.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def resetclassifier(self):
        self.classifier = distLinear(self.encoder_findim, self.num_class)
