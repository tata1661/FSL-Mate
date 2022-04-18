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
from .conv import ConvBlock

__all__ = ['ConvEmbedModel', 'ConvRelationModel']


def _reshape_input(prototypes, query_embeddings):
    batch_list = []
    for query_embedding in query_embeddings:
        for prototype in prototypes:
            batch_list.append(paddle.concat([prototype, query_embedding], axis=0))
    relation_input = paddle.stack(batch_list, axis=0)
    return relation_input


def _reshape_output(relation_score, ways):
    relation_score = paddle.squeeze(relation_score, axis=1)
    relation_score = [relation_score[i: i + ways] for i in range(0, len(relation_score), ways)]
    output = paddle.stack(relation_score)
    return output


class ConvEmbedModel(nn.Layer):
    """
    Convolutional Embedding Module in RelationNet.
    In embedding module of the RelationNet, each convolutional block contains a 64-filter 3 ×3 convolution,
    a batch normalisation and a ReLU nonlinearity layer respectively. The first two blocks also contain a
    2 ×2 max-pooling layer while the latter two do not[1].

    Refs:
        1.Sung, Flood, et al. 2018. "Learning to compare: Relation network for few-shot learning." CVPR.

    Args:
        input_size(Tuple): input size of the image. The tuple must have 3 items representing channels, width and
            height of the image, for example (1, 28, 28) for Omniglot images.
        num_filters(int, optional): number of filters in the hidden convolutional layers, default 64.

    Examples:
        ..code-block:: python

            train_input = paddle.ones(shape=(1, 1, 28, 28), dtype='float32')
            embed_model = ConvEmbedModel(input_size=(1, 28, 28), num_filters=64)
            print(embed_model.output_size)  # (64, 7, 7)
            print(embed_model(train_input))  # Tensor of shape [1, 64, 7, 7]

    """

    def __init__(self, input_size, num_filters=64):
        super(ConvEmbedModel, self).__init__()
        self.conv0 = ConvBlock(in_channels=input_size[0], out_channels=num_filters, pooling_size=(2, 2))
        self.conv1 = ConvBlock(in_channels=num_filters, out_channels=num_filters, pooling_size=(2, 2))
        self.conv2 = ConvBlock(in_channels=num_filters, out_channels=num_filters)
        self.conv3 = ConvBlock(in_channels=num_filters, out_channels=num_filters)
        self.output_size = (num_filters, input_size[1] >> 2, input_size[2] >> 2)

    def forward(self, inputs):
        output = self.conv0(inputs)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        return output


class ConvRelationModel(paddle.nn.Layer):
    """
    Relation Module for convolution embedding module of relationNet.
    The relation module consists of two convolutional blocks and two fully-connected layers. Each of
    convolutional block is a 3 × 3 convolution with 64 filters followed by batch normalisation, ReLU
    non-linearity and 2 × 2 max-pooling.

    Args:
        input_size(Tuple, optional): input size of the representation, usually it is the output size of the
            embedding model of the RelationNet, default (64, 21, 21).
        output_size(int, optional): output size of the model, usually the way number, default 5.
        num_filters(int, optional): number of filters in the hidden convolutional layers, default 64.

    Examples:
        ..code-block:: python

            prototypes = paddle.ones(shape=(5, 64, 21, 21), dtype='float32')  # 5: 5 ways
            query_embeddings = paddle.ones(shape=(10, 64, 21, 21), dtype='float32')  # 10: batch size 10
            embed_model = ConvRelationModel()
            print(embed_model(prototypes, query_embeddings))  # Tensor of shape [10, 5]

    """
    init_weight_attr = paddle.framework.ParamAttr(initializer=nn.initializer.TruncatedNormal(mean=0.0, std=0.01))
    init_bias_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Constant(value=0.0))

    def __init__(self, input_size=(64, 21, 21), output_size=5, num_filters=64):
        super(ConvRelationModel, self).__init__()
        self.conv0 = ConvBlock(in_channels=num_filters * 2, out_channels=num_filters, pooling_size=(2, 2))
        self.conv1 = ConvBlock(in_channels=num_filters, out_channels=num_filters, pooling_size=(2, 2))
        input_size = input_size[0] * (input_size[1] >> 2) * (input_size[2] >> 2)
        self.fc0 = paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(in_features=input_size, out_features=8,
                             weight_attr=self.init_weight_attr, bias_attr=self.init_bias_attr),
            paddle.nn.ReLU()
        )
        self.fc1 = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=8, out_features=1,
                             weight_attr=self.init_weight_attr, bias_attr=self.init_bias_attr),
            paddle.nn.Sigmoid()
        )
        self.output_size = output_size

    def forward(self, prototypes, query_embeddings):
        """
        Forward calculation of the model.

        Args:
            prototypes: tensor of shape (ways, representation_shape), for example (5, 64, 21, 21).
                prototypes[i] represents the prototype of class i.
            query_embeddings: tensor of shape (batch_size, representation_shape), for example (10, 64, 21, 21)

        Returns:
            classification logits of shape (batch_size, ways)

        """
        ways, batch_size = prototypes.shape[0], query_embeddings.shape[0]
        assert ways == self.output_size
        relation_input = _reshape_input(prototypes, query_embeddings)
        relation_score = self.conv0(relation_input)
        relation_score = self.conv1(relation_score)
        relation_score = self.fc0(relation_score)
        relation_score = self.fc1(relation_score)
        # output = _reshape_output(relation_score, self.output_size)
        output = paddle.reshape(relation_score, shape=(batch_size, ways))
        return output
