"""ANIL Meta Learner"""
from __future__ import annotations

from paddle.nn import Layer

from paddlefsl.utils.manual_gradient_descent import manual_gradient_descent
from paddlefsl.utils.clone_model import clone_model
from paddlefsl.metaopt.base_learner import BaseLearner


class ANILLearner(BaseLearner):
    """ANIL Meta Learner"""
    def __init__(self, feature_model: Layer, head_layer: Layer, learning_rate: float, approximate: bool = True) -> None:
        """The constructor of ANILearner

        Args:
            module (Layer): the model to be trained
            optimizer (Optimizer): the optimizer to be used
        """
        super().__init__(head_layer)
        self.feature_model = feature_model
        self.learning_rate = learning_rate
        self.approximate = approximate

    def clone(self) -> ANILLearner:
        """get the cloned model and keep the computation gragh

        Returns:
            ANILearner: the cloned model
        """
        cloned_head_layer = clone_model(self.module)
        return ANILLearner(
            feature_model=self.feature_model,
            head_layer=cloned_head_layer,
            learning_rate=self.learning_rate,
            approximate=self.approximate
        )

    def adapt(self, loss) -> None:
        """adapt the gradient descent to the module based on the loss

        Args:
            loss (Tensor): the loss of head layer
        """
        manual_gradient_descent(
            model=self.module,
            lr=self.learning_rate,
            loss=loss,
            approximate=self.approximate
        )
    
    def forward(self, inputs):
        """forward the feature model and the head layer"""
        y = self.feature_model(inputs)
        y = self.module(y)
        return y