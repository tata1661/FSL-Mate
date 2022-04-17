"""ANIL Meta Learner"""
from __future__ import annotations

import paddle
from paddle.nn import Layer
from paddle.optimizer import Optimizer

from paddlefsl.utils import gradient_descent
from .base_learner import BaseLearner


class ANILearner(BaseLearner):
    """ANIL Meta Learner"""
    def __init__(self, module: Layer, optimizer: Optimizer, learning_rate: float, approximate: bool = True) -> None:
        """The constructor of ANILearner

        Args:
            module (Layer): the model to be trained
            optimizer (Optimizer): the optimizer to be used
        """
        super().__init__(module, optimizer)
        self.learning_rate = learning_rate
        self.approximate = approximate

    def adapt(self, train_loss) -> None:
        gradient_descent(
            model=self.cloned_module,
            lr=self.learning_rate,
            loss=train_loss,
            approximate=self.approximate
        )

    def step(self) -> None:
        """run step in the meta learner"""
        self.optimizer.step()