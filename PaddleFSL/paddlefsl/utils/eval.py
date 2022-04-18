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


__all__ = ['classification_acc']


def classification_acc(predict, label):
    """
    Calculate classification accuracy: correct_result / sample_number

    Args:
        predict(paddle.Tensor): predictions, shape (sample_number, class_number), in the form of one-hot coding.
        label(paddle.Tensor): labels, shape (sample_number), in the form of continuous coding.

    Returns:
        accuracy(float): classification accuracy.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            predictions = paddle.to_tensor([[0.1, 0.9], [0.8, 0.2]], dtype='float32')
            labels = paddle.to_tensor([0, 0], dtype='int64')
            accuracy = utils.classification_acc(predictions, labels) # accuracy: 0.5

    """
    correct = 0
    for i in range(predict.shape[0]):
        if paddle.argmax(predict[i]) == int(label[i]):
            correct += 1
    return float(correct) / predict.shape[0]