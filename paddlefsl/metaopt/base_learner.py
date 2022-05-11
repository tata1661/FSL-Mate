"""Base Learner"""
from __future__ import annotations
from typing import Type, TypeVar
from abc import ABC, abstractmethod
from ast import arg
from turtle import forward

from paddle.nn import Layer
from paddle.optimizer import Optimizer


Learner = TypeVar('Learner')

class BaseLearner(Layer):
    """Abstract Base Learner Class"""
    def __init__(self, module: Layer) -> None:
        """The constructor of BaseLearner

        Args:
            module (Layer): the model to be trained
        """
        super().__init__()
        self.module = module

    @abstractmethod
    def adapt(self, loss: Tensor) -> None:
        """Adapt the model to the current training loss

        Args:
            loss (Tensor): the current training loss
        """
        raise NotImplementedError
    
    def clone(self: Type[Learner]) -> Learner:
        """create cloned module and keep the computation gragh

        Args:
            self (Type[Learner]): the sub-learner

        Returns:
            Learner: the cloned model
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
