# Copyright 2021 PaddleFSL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import math

__all__ = ['ConvBlock', 'Conv', 'RCConv1D']


class ConvBlock(nn.Layer):
    """
    Implementation of a convolution block: conv2d-BatchNorm-ReLU-pooling

    Args:
        in_channels(int): channel number of the input.
        out_channels(int): channel number of the output.
        kernel_size(Tuple, optional): convolutional kernel size, default (3, 3).
        pooling_size(Tuple, optional): pooling kernel size, default None. If None, this layer will not do max-pooling.
        stride_size(Tuple, optional): convolutional stride size, default (1, 1).

    Examples:
        ..code-block:: python

            train_input = paddle.ones(shape=(1, 1, 28, 28), dtype='float32')
            conv = paddle.nn.Sequential(
                ConvBlock(in_channels=1, out_channels=32, pooling_size=(2, 2)),
                ConvBlock(in_channels=32, out_channels=64, pooling_size=(2, 2)),
                paddle.nn.Flatten(),
                paddle.nn.Linear(64 * 7 * 7, 2)
            )
            print(conv(train_input)) # A paddle.Tensor with shape=[1, 2]

    """

    init_weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
    init_bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0))
    norm_weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Uniform())

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 pooling_size=None,
                 stride_size=(1, 1)):
        super(ConvBlock, self).__init__()
        # Conv2d layer
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride_size,
                              padding=1,
                              weight_attr=self.init_weight_attr,
                              bias_attr=self.init_bias_attr)
        # BatchNorm layer
        self.norm = nn.BatchNorm2D(out_channels,
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   weight_attr=self.norm_weight_attr)
        # ReLU activation
        self.relu = nn.ReLU()
        # Max-pooling
        self.pool = nn.MaxPool2D(kernel_size=pooling_size) if pooling_size is not None else lambda x: x

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.norm(y)
        y = self.relu(y)
        return self.pool(y)


class Conv(nn.Layer):
    """
    Implementation of a CNN(Convolutional Neural Network) model.

    Args:
        input_size(Tuple): input size of the image. The tuple must have 3 items representing channels, width and
            height of the image, for example (1, 28, 28) for Omniglot images.
        output_size(int): size of the output.
        conv_channels(List, optional): channel numbers of the hidden layers, default None. If None, it will be set
            [64, 64, 64, 64]. Please note that there is a final classifier after all hidden layers, so the last
            item of hidden_sizes is not the output size.
        kernel_size(Tuple, optional): convolutional kernel size, default (3, 3).
        pooling(bool, optional): whether to do max-pooling after each convolutional layer, default True. If False,
            the model use stride convolutions instead of max-pooling. If True, the pooling kernel size will be (2, 2).

    Examples:
        ..code-block:: python

            train_input = paddle.ones(shape=(1, 1, 28, 28), dtype='float32')
            conv = Conv(input_size=(1, 28, 28), output_size=2)
            print(conv(train_input)) # A paddle.Tensor with shape=[1, 2]

    """
    init_weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
    init_bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0))

    def __init__(self,
                 input_size,
                 output_size,
                 conv_channels=None,
                 kernel_size=(3, 3),
                 pooling=True):
        super(Conv, self).__init__()
        # Convolution layers
        conv_channels = [64, 64, 64, 64] if conv_channels is None else conv_channels
        if len(input_size) != 3:
            raise ValueError('The input_size must have 3 items representing channels, width and height of the image.')
        in_channels, out_channels = input_size[0], conv_channels[0]
        pooling_size = (2, 2) if pooling else None
        stride_size = (2, 2) if not pooling else (1, 1)
        self.conv = nn.Sequential(
            ('conv0', ConvBlock(in_channels, out_channels, kernel_size, pooling_size, stride_size))
        )
        for i in range(1, len(conv_channels)):
            in_channels, out_channels = out_channels, conv_channels[i]
            self.conv.add_sublayer(name='conv' + str(i),
                                   sublayer=ConvBlock(in_channels, out_channels,
                                                      kernel_size, pooling_size, stride_size))
        # Output layer
        if pooling:
            features = (input_size[1] >> len(conv_channels)) * (input_size[2] >> len(conv_channels))
        else:
            width, height = input_size[1], input_size[2]
            for i in range(len(conv_channels)):
                width, height = math.ceil(width / 2), math.ceil(height / 2)
            features = width * height
        self.feature_size = features * conv_channels[-1]
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, output_size,
                      weight_attr=self.init_weight_attr, bias_attr=self.init_bias_attr)
        )

    def forward(self, inputs):
        y = self.conv(inputs)
        return self.output(y)


class RCConv1D(nn.Layer):

    def __init__(self,
                 max_len=100,
                 embedding_size=60,
                 hidden_size=230,
                 output_size=None):
        super(RCConv1D, self).__init__()
        self.conv1d = nn.Conv1D(
            in_channels=embedding_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        self.activate = nn.ReLU()
        self.pooling = nn.MaxPool1D(max_len)
        if output_size is None:
            self.output = nn.Flatten()
        else:
            self.output = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_size, out_features=output_size)
            )
        self.hidden_size = hidden_size

    def forward(self, inputs):
        # inputs: shape: [batch_size, max_len, embedding_size]
        # In few shot learning, batch_size = ways * shots (or ways * query_num)
        # In relation classification, embedding_size = word_embedding_dimension + 2 * position_embedding_dimension
        inputs = paddle.transpose(inputs, [0, 2, 1])
        output = self.conv1d(inputs)
        output = self.activate(output)
        output = self.pooling(output)
        return self.output(output)
