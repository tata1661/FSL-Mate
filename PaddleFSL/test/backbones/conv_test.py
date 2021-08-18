import paddle
from paddlefsl.backbones import ConvBlock, Conv, RCConv1D


def conv_block_test():
    train_input = paddle.ones(shape=(1, 1, 28, 28), dtype='float32')
    conv = paddle.nn.Sequential(
        ConvBlock(in_channels=1, out_channels=32, pooling_size=(2, 2)),
        ConvBlock(in_channels=32, out_channels=64, pooling_size=(2, 2)),
        paddle.nn.Flatten(),
        paddle.nn.Linear(64 * 7 * 7, 2)
    )
    print(conv(train_input))  # Tensor of shape [1, 2]


def conv_test():
    train_input = paddle.ones(shape=(1, 1, 28, 28), dtype='float32')
    conv = Conv(input_size=(1, 28, 28), output_size=2)
    print(conv(train_input))  # Tensor of shape [1, 2]


def rc_conv1d_test():
    max_len, embedding_size = 100, 60
    train_input = paddle.rand(shape=[5, max_len, embedding_size], dtype='float32')
    conv1d = RCConv1D(max_len=max_len, embedding_size=embedding_size)
    print(conv1d(train_input).shape)


if __name__ == '__main__':
    conv_block_test()
    conv_test()
    rc_conv1d_test()
