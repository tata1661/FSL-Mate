import paddle
from collections import OrderedDict

__all__ = ['classification_acc', 'clone_model', 'gradient_descent']


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


def gradient_descent(model, lr, loss, approximate=True, memo=None):
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
                utils.gradient_descent(model, 0.1, loss)
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
        gradient_descent(sub_layer, lr, loss, approximate, memo)
