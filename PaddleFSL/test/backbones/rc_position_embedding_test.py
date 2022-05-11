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
from paddlefsl.backbones import RCPositionEmbedding


def rc_position_embedding_test():
    max_len = 100
    embedding_dim = 50
    position_embedding_dim = 5
    rc_vector = paddle.rand(shape=[5, max_len, embedding_dim + 2], dtype='float32')
    position_embedding = RCPositionEmbedding(
        max_len=max_len,
        embedding_dim=embedding_dim,
        position_embedding_dim=position_embedding_dim
    )
    print(position_embedding(rc_vector).shape)


if __name__ == '__main__':
    rc_position_embedding_test()
