"""MAML Meta Learner"""
from __future__ import annotations

import paddle
from paddle.nn import Layer
from paddle.optimizer import Optimizer

from paddlefsl.utils import gradient_descent
from .base_learner import BaseLearner
from ..utils.model import clone_model


class MAMLLearner(BaseLearner):
    """MAML Meta Learner"""
    def __init__(self, module: Layer, learning_rate: float, approximate: bool = True) -> None:
        """The constructor of MAMLLearner

        Args:
            module (Layer): the model to be trained
            optimizer (Optimizer): the optimizer to be used
        """
        super().__init__(self, module)

        self.learning_rate = learning_rate
        self.approximate = approximate
    
    def clone(self,) -> Layer:
        """get the cloned model and keep the computation gragh

        Returns:
            Layer: the cloned model
        """
        cloned_module = clone_model(self.module)

        return MAMLLearner(
            module=cloned_module,
            learning_rate=self.learning_rate,
            approximate=self.approximate
        )

    def adapt(self, loss: Tensor) -> None:
        """adapt the gradient descent to the module based on the loss

        Args:
            loss (Tensor): _description_
        """
        gradient_descent(
            self.module,
            lr=self.learning_rate,
            loss=loss,
            approximate=self.approximate
        )
