import paddle
import paddle.nn as nn
from paddle.nn.utils import weight_norm


class distLinear(nn.Layer):
    """
    Distance-based classifier.

    Args:
        indim(int, optional):  dimensions of the input, default 1024.
        outdim(int, optional): dimensions of the output, default 64.

    Examples:
        ..code-block:: python
            train_input = paddle.ones(shape=(25, 1024), dtype='float32')  # 5: 5 ways
            classifier = distLinear(indim=1024, outdim=64)
            print(classifier(train_input))  # Tensor of shape [25, 64]

    """
    def __init__(self, indim=1024, outdim=64):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias_attr=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            weight_norm(self.L, 'weight', dim=0)

        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_norm = paddle.norm(x, p=2, axis=1).unsqueeze(1).expand_as(x)
        x_normalized = x.divide(x_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)
        return scores