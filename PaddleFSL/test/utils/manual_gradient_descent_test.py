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
import paddlefsl.utils as utils


def manual_gradient_descent_test():
    train_data = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]], dtype='float32')
    train_label = paddle.to_tensor([0, 1], dtype='int64')
    test_data = paddle.to_tensor([[0.98, 0.98]], dtype='float32')
    model = paddle.nn.Linear(2, 2)
    loss_fn = paddle.nn.CrossEntropyLoss()
    for epoch in range(100):
        predict = model(train_data)
        loss = loss_fn(predict, train_label)
        utils.manual_gradient_descent(model, 0.1, loss)
    print(model(test_data))  # Tensor of shape [1, 2]
    pass

if __name__ == '__main__':
    manual_gradient_descent_test()
