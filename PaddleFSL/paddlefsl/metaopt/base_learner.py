"""Base Learner"""
from __future__ import annotations
from abc import ABC, abstractmethod

from paddle.nn import Layer
from paddle.optimizer import Optimizer
from ..utils.model import clone_model

class BaseLearner(ABC):
    """Abstract Base Learner Class"""
    def __init__(self, module: Layer, optimizer: Optimizer) -> None:
        """The constructor of BaseLearner

        Args:
            module (Layer): the model to be trained
        """
        super().__init__()
        self._source_module = module
        self.cloned_module = None
        self.optimizer = optimizer

    def new_cloned_model(self,) -> Layer:
        """get the cloned model and keep the computation gragh

        Returns:
            Layer: the cloned model
        """
        self.cloned_module = clone_model(self._source_module)
        return self.cloned_module

    @abstractmethod
    def adapt(self, train_loss: Tensor) -> None:
        """Adapt the model to the current training loss

        Args:
            train_loss (Tensor): the current training loss
        """
        raise NotImplementedError


    @abstractmethod
    def step(self) -> None:
        """Perform a step of training

        Args:
            loss (float): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def clear_grad(self):
        """clear the gradient in the computation graph
        """
        self.optimizer.clear_grad()
