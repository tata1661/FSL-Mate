"""ANIL Meta Learner"""
from __future__ import annotations

import paddle
from paddle.nn import Layer
from paddle.optimizer import Optimizer

from paddlefsl.utils import gradient_descent
from .base_learner import BaseLearner


class ANILearner(BaseLearner):
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

    def clone(self) -> ANILearner:
        """get the cloned model and keep the computation gragh

        Returns:
            ANILearner: the cloned model
        """
        cloned_head_layer = clone_model(self.module)

        return ANILearner(
            feature_model=self.feature_model,
            head_layer=cloned_head_layer,
            learning_rate=self.learning_rate,
            approximate=self.approximate
        )

    def adapt(self, train_loss) -> None:
        gradient_descent(
            model=self.module,
            lr=self.learning_rate,
            loss=train_loss,
            approximate=self.approximate
        )