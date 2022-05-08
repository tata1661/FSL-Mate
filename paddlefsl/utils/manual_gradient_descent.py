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


__all__ = ['manual_gradient_descent']


def manual_gradient_descent(model, lr, loss, approximate=True, memo=None):
    """
    Do a step of gradient descent on the model manually, instead of using loss.backward() and opt.step().

    Args:
        model(paddle.nn.Layer): model to be updated.
        lr(float): learning rate used in gradient descent.
        loss(paddle.Tensor): loss tensor used to calculate gradient.
        approximate(bool, optional): whether to perform first order approximate in MAML, default True. Here, to retain
            second order gradient is to set retain_graph and create_graph True when calculating gradients.
        memo(Set, optional): memo of the parameters and buffers, default None. If the parameter or buffer to be updated
            is in memo, new parameter or buffer will be set the value in memo instead of updating again.
            Usually it will be set None when calling this function from outside.

    Returns:
        None.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            train_data = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]], dtype='float32')
            train_label = paddle.to_tensor([0, 1], dtype='int64')
            test_data = paddle.to_tensor([[0.98, 0.98]], dtype='float32')
            model = paddle.nn.Linear(2, 2)
            loss_fn = paddle.nn.CrossEntropyLoss()
            for epoch in range(100):
                predict = model(train_data)
                loss = loss_fn(predict, train_label)
                utils.manual_gradient_descent(model, 0.1, loss)
            print(model(test_data))
            # The second item of the result should be larger than the first which means the model predicts "1".

    """
    # Maps original data_ptr to the cloned tensor.
    # Useful when a model uses parameters from another model.
    memo = set() if memo is None else memo
    # Do gradient descent on parameters
    gradients = []
    if len(model.parameters()) != 0:
        gradients = paddle.grad(loss,
                                model.parameters(),
                                retain_graph=not approximate,
                                create_graph=not approximate,
                                allow_unused=True)
    update_values = [- lr * grad if grad is not None else None for grad in gradients]
    for param, update in zip(model.parameters(), update_values):
        if update is not None:
            param_ptr = id(param)
            if param_ptr not in memo:
                memo.add(param_ptr)
                param.set_value(param.add(update))
    # Do gradient descent recursively on sub-layers
    for sub_layer in model.sublayers():
        manual_gradient_descent(sub_layer, lr, loss, approximate, memo)