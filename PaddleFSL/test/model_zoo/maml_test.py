import paddle
from paddlefsl.model_zoo import maml


def inner_adapt_test():
    img1, label1 = paddle.ones(shape=(1, 1, 2, 2), dtype='float32'), paddle.to_tensor([[0]], dtype='int64')
    img2, label2 = paddle.zeros(shape=(1, 1, 2, 2), dtype='float32'), paddle.to_tensor([[1]], dtype='int64')
    model = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(4, 2)
    )
    loss_fn = paddle.nn.CrossEntropyLoss()
    data = ((img1, label1), (img2, label2))
    maml.inner_adapt(model, data, loss_fn, 0.4)


if __name__ == '__main__':
    inner_adapt_test()
