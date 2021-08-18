import paddle
import paddlefsl.utils as utils


def classification_acc_test():
    predictions = paddle.to_tensor([[0.1, 0.9], [0.8, 0.2]], dtype='float32')
    labels = paddle.to_tensor([0, 0], dtype='int64')
    accuracy = utils.classification_acc(predictions, labels)
    print(accuracy)  # 0.5


def clone_model_test():
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
    print(model(test_data))  # Tensor of shape [1, 2]
    model_cloned = utils.clone_model(model)
    print(model_cloned(test_data))  # Tensor of shape [1, 2]


def gradient_descent_test():
    train_data = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]], dtype='float32')
    train_label = paddle.to_tensor([0, 1], dtype='int64')
    test_data = paddle.to_tensor([[0.98, 0.98]], dtype='float32')
    model = paddle.nn.Linear(2, 2)
    loss_fn = paddle.nn.CrossEntropyLoss()
    for epoch in range(100):
        predict = model(train_data)
        loss = loss_fn(predict, train_label)
        utils.gradient_descent(model, 0.1, loss)
    print(model(test_data))  # Tensor of shape [1, 2]
    pass


if __name__ == '__main__':
    classification_acc_test()
    clone_model_test()
    gradient_descent_test()
