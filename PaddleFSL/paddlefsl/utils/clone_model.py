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
from collections import OrderedDict


__all__ = ['clone_model']


def _clone_ordered_dict(ordered_dict, memo):
    cloned = OrderedDict()
    keys_values = [(key, value) for key, value in ordered_dict.items()]
    for (key, value) in keys_values:
        value_ptr = id(value)
        if value_ptr not in memo:
            cloned[key] = memo[value_ptr] = value.clone()
        else:
            cloned[key] = memo[value_ptr]
    return cloned


def clone_model(model, memo=None):
    """
    Clone a model, return the cloned model. The cloned model and the original one do not share memory, but share the
    same computation graph.

    Args:
        model(paddle.nn.Layer): model to be cloned.
        memo(Dict, optional): memo of the parameters and buffers, default None. If the parameter or buffer to be cloned
            is in memo, new parameter or buffer will be set the value in memo instead of cloning the original ones.
            Usually it will be set None when calling this function from outside.

    Returns:
        cloned(paddle.nn.Layer): the cloned model.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            train_data = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]], dtype='float32')
            train_label = paddle.to_tensor([0, 1], dtype='int64')
            test_data = paddle.to_tensor([[0.99, 0.99]], dtype='float32')
            model = paddle.nn.Linear(2, 2)
            loss_fn, opt = paddle.nn.CrossEntropyLoss(), paddle.optimizer.Adam(parameters=model.parameters())
            for epoch in range(100):
                predict = model(train_data)
                loss = loss_fn(predict, train_label)
                loss.backward()
                opt.step()
            print(model(test_data))
            model_cloned = utils.clone_model(model)
            print(model_cloned(test_data))
            # model_cloned(test_data) should be the same as model(test_data), and the second item of the result
            # should be larger than the first which means the model predicts "1".

    """
    # Maps original data_ptr to the cloned tensor.
    # Useful when a model uses parameters from another model.
    memo = {} if memo is None else memo
    # Create a copy of the model.
    cloned = model.__new__(type(model))
    cloned.__dict__ = model.__dict__.copy()
    # Clone parameters of the current layer.
    cloned._parameters = _clone_ordered_dict(model._parameters, memo)
    # Clone buffers of the current layer.
    cloned._buffers = _clone_ordered_dict(model._buffers, memo)
    # Recursively clone each sub-layer.
    cloned._sub_layers = OrderedDict()
    for layer_key in model._sub_layers:
        sub_layer = model._sub_layers[layer_key]
        cloned._sub_layers[layer_key] = clone_model(sub_layer, memo)
    return cloned