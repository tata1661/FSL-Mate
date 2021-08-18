import paddle
import paddle.nn as nn


__all__ = ['LinearBlock', 'MLP']


class LinearBlock(nn.Layer):
    """
    Implementation of a linear block: linear-BatchNorm-ReLU

    Args:
        input_size(int): size of the input
        output_size(int): size of the output

    Examples:
        ..code-block:: python

            train_input = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]])
            mlp = paddle.nn.Sequential(
                LinearBlock(2, 5),
                LinearBlock(5, 1)
            )
            print(mlp(train_input)) # A paddle.Tensor with shape=[2, 1]

    """

    init_weight_attr = paddle.framework.ParamAttr(initializer=nn.initializer.TruncatedNormal(mean=0.0, std=0.01))
    init_bias_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Constant(value=0.0))

    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()
        # Linear layer
        self.linear = nn.Linear(input_size,
                                output_size,
                                weight_attr=self.init_weight_attr,
                                bias_attr=self.init_bias_attr)
        # Batch-normalization layer
        self.norm = nn.BatchNorm1D(output_size, momentum=0.999, epsilon=1e-3)
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x)
        return self.relu(x)


class MLP(nn.Layer):
    """
    Implementation of MLP(Multi-Layer Perceptron) model.

    Args:
        input_size(int or tuple): size of the input, int for a vector and tuple for an image. As MLP flattens the
            input, users can also set input_size to the absolut size of the image. For example, if the input is a
            28 * 28, 3-channel image, both (28, 28, 3) and 28 * 28 * 3 is the legal input_size.
        output_size(int): size of the output.
        hidden_sizes(List, optional): hidden sizes of each hidden layer, default None. If None, it will be set
            [256, 128, 64, 64]. Please note that there is a final classifier after all hidden layers, so the last
            item of hidden_sizes is not the output size.

    Examples:
        ..code-block:: python

            import paddle
            from paddlefsl.vision.backbones import MLP
            train_input = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0]])
            mlp = MLP(input_size=2, output_size=1, hidden_sizes=[5])
            print(mlp(train_input)) # A paddle.Tensor with shape=[2, 1]

    """

    init_weight_attr = paddle.framework.ParamAttr(initializer=nn.initializer.TruncatedNormal(mean=0.0, std=0.01))
    init_bias_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Constant(value=0.0))

    def __init__(self, input_size, output_size, hidden_sizes=None):
        super(MLP, self).__init__()
        # Flatten layer
        self.flatten = nn.Flatten()
        # Hidden layers
        if type(input_size) is not int:
            hidden_in = 1
            for size in input_size:
                hidden_in *= size
        else:
            hidden_in = input_size
        hidden_sizes = [256, 128, 64, 64] if hidden_sizes is None else hidden_sizes
        hidden_out = hidden_sizes[0]
        self.hidden = nn.Sequential(
            ('hidden0', LinearBlock(hidden_in, hidden_out))
        )
        for i in range(1, len(hidden_sizes)):
            hidden_in, hidden_out = hidden_out, hidden_sizes[i]
            self.hidden.add_sublayer(name='hidden' + str(i),
                                     sublayer=LinearBlock(hidden_in, hidden_out))
        # Output layer
        self.output = paddle.nn.Linear(hidden_sizes[-1],
                                       output_size,
                                       weight_attr=self.init_weight_attr,
                                       bias_attr=self.init_bias_attr)

    def forward(self, inputs):
        y = self.flatten(inputs)
        y = self.hidden(y)
        return self.output(y)
