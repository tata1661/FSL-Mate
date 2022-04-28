"""MAML example for optimization"""
from __future__ import annotations
import os
from sched import scheduler
from typing import Optional, Tuple
import warnings
import math
from loguru import logger

import paddle
from paddle import nn
from paddle.optimizer import Adam, Optimizer
from paddle.optimizer.lr import LRScheduler
from paddle.nn import Layer
from paddle.metric.metrics import Accuracy
from tap import Tap
from tqdm import tqdm
from tabulate import tabulate
from mlflow.tracking import MlflowClient
from mlflow import set_tag
import numpy as np

import paddlefsl
from paddlefsl.datasets.cv_dataset import CVDataset
from paddlefsl.backbones.conv import ConvBlock
from paddlefsl.metaopt.base_learner import BaseLearner
from paddlefsl.metaopt.anil import ANILLearner

"""Data Utils for Meta Optimzations Algorithms"""
from typing import Tuple, Dict
import paddlefsl
from paddlefsl.datasets.cv_dataset import CVDataset


def load_datasets(name: str) -> Tuple[CVDataset, CVDataset, CVDataset]:
    """load CV Dataset by name, which can be omniglot, miniimagenet, or cifar10

    Args:
        name (str): the name of datasets

    Returns:
        Tuple[CVDataset, CVDataset, CVDataset]: train, dev, test dataset
    """
    if name == "omniglot":
        return (
            paddlefsl.datasets.Omniglot(mode='train', image_size=(28, 28)),
            paddlefsl.datasets.Omniglot(mode='valid', image_size=(28, 28)),
            paddlefsl.datasets.Omniglot(mode='test', image_size=(28, 28))
        )
    if name == "miniimagenet":
        return (
            paddlefsl.datasets.MiniImageNet(mode='train'),
            paddlefsl.datasets.MiniImageNet(mode='valid'),
            paddlefsl.datasets.MiniImageNet(mode='test')
        )
    if name == "cifarfs":
        return (
            paddlefsl.datasets.CifarFS(mode='train', image_size=(28, 28)),
            paddlefsl.datasets.CifarFS(mode='valid', image_size=(28, 28)),
            paddlefsl.datasets.CifarFS(mode='test', image_size=(28, 28))
        )
    if name == "fc100":
        return (
            paddlefsl.datasets.FC100(mode='train'),
            paddlefsl.datasets.FC100(mode='valid'),
            paddlefsl.datasets.FC100(mode='test')
        )
    if name == "cub":
        return (
            paddlefsl.datasets.CubFS(mode='train'),
            paddlefsl.datasets.CubFS(mode='valid'),
            paddlefsl.datasets.CubFS(mode='test')
        )
    raise ValueError(f"the dataset name: <{name}> is not supported")


class Config(Tap):
    """Alernative for Argument Parse"""
    dataset: str = 'omniglot'
    input_size: Optional[str] = None
    n_way: int = 5
    k_shot: int = 1
    meta_lr: float = 0.005
    inner_lr: float = 0.5
    epochs: int = 60000                 # also named as iterations
    test_epoch: int = 10
    eval_iters: int = 10

    meta_batch_size: int = 32
    test_batch_size: int = 10
    
    train_inner_adapt_steps: int = 1
    test_inner_adapt_steps: int = 1

    approximate: bool = True

    do_eval_step: int = 30
    do_test_step: int = 100
    save_model_iter: int = 5000
    save_model_root: str = '~/trained_models'
    test_param_file: str = 'iteration60000.params'

    device: str = 'cpu'

    tracking_uri: str = ''
    experiment_id: str = '0'
    run_id: str = '<run_id>'

    def place(self):
        """get the default device place for tensor"""
        return paddle.fluid.CUDAPlace(0) if self.device == 'gpu' else paddle.fluid.CPUPlace()

    def get_input_size(self) -> Tuple[int, int, int]:
        """get the input size based on the datasets"""
        if self.dataset in ['omniglot']:
            return (1, 28, 28)
        
        if self.dataset in ['cifarfs']:
            return (3, 32, 32)

        if self.dataset == 'miniimagenet':
            return (3, 84, 84)

        if self.dataset == 'fc100':
            return (3, 32, 32)

        if self.dataset == 'cub':
            return (3, 84, 84)

        if not self.input_size:
            return None

        return tuple(map(int, self.input_size.split(',')))

class ContextData:
    """context data to store the training and testing results"""
    def __init__(self) -> None:
        self.train_epoch = 0
        self.epoch = 0

        self.train_loss = 0
        self.train_acc = 0

        self.dev_loss = 0
        self.dev_acc = 0


class Trainer:
    """Trainer for meta training epoch"""
    def __init__(
        self,
        config: Config,
        train_dataset: CVDataset,
        dev_dataset: CVDataset,
        test_dataset: CVDataset,
        learner: BaseLearner,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: Layer,

    ) -> None:
        self.config = config
        
        logger.info(self.config)

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
        self.criterion = criterion
        self.learner = learner

        self.train_bar = tqdm(total=self.config.epochs, desc='Train Progress')
        self.context = ContextData()

        self.metric = Accuracy()

        self.scheduler = scheduler
        self.optimizer = optimizer

        self._set_device()
        warnings.filterwarnings("ignore")

        if self.config.tracking_uri:
            self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
            
            run = self.client.create_run(self.config.experiment_id)
            self.config.run_id = run.info.run_id

            for key, value in self.config.as_dict().items():
                self.client.log_param(
                    self.config.run_id,
                    key=key,
                    value=value
                )

            self.client.log_param(self.config.run_id, 'learner', value=self.learner.__class__.__name__)
            set_tag("mlflow.runName", self.learner.__class__.__name__)

    def _set_device(self):
        paddle.device.set_device(self.config.device)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

    def on_train_epoch_end(self):
        """handle the end of one epoch"""
        self.context.epoch += 1

        self.train_bar.update()

        bar_info = f'Epoch: {self.context.epoch}/{self.config.epochs} \t dev-loss: {self.context.dev_loss}  \t\t dev-acc: {self.context.dev_acc}'
        self.train_bar.set_description(bar_info)

        if self.config.tracking_uri:
            self.client.log_metric(self.config.run_id, key='train-loss', value=self.context.train_loss)
            self.client.log_metric(self.config.run_id, key='train-acc', value=self.context.train_acc)

            self.client.log_metric(self.config.run_id, key='dev-loss', value=self.context.dev_loss)
            self.client.log_metric(self.config.run_id, key='dev-acc', value=self.context.dev_acc)

    def fast_adapt(self, task, learner: BaseLearner):
        """make inner loop fast adaption based on the task

        Args:
            task (_type_): which contains the support_set & query_set data
            learner (BaseLearner): the meta optimization algriothm, which contains the model

        Returns:
            Tuples[Tensor, Tensor]: the loss and acc of the inner loop
        """
        support_set, support_set_labels = paddle.to_tensor(task.support_data, dtype='float32'), paddle.to_tensor(task.support_labels, dtype='int64')
        query_set, query_set_labels = paddle.to_tensor(task.query_data, dtype='float32'), paddle.to_tensor(task.query_labels, dtype='int64')

        # handle the fast adaption in few dataset
        for _ in range(self.config.train_inner_adapt_steps):
            logits = learner(support_set)
            loss = self.criterion(logits, support_set_labels)
            learner.adapt(loss)

        # evaluate the model on query set data
        logits = learner(query_set)
        val_loss = self.criterion(logits, query_set_labels)
        val_acc = self.metric.compute(logits, query_set_labels)
        val_acc = self.metric.update(val_acc)
        return val_loss, val_acc

    def compute_loss(self, input_data, labels, learner: BaseLearner, inner_steps: int = 1):
        """compute the loss based on the input_data and labels"""
        input_data, labels = paddle.to_tensor(input_data, dtype='float32'), paddle.to_tensor(labels, dtype='int64')

        all_loss, all_acc = 0.0, 0.0
        for _ in range(inner_steps):
            logits = learner(input_data)
            loss = self.criterion(logits, labels)
            all_loss += loss

            acc = self.metric.compute(logits, labels)
            all_acc += self.metric.update(acc)
        
        return all_loss / inner_steps, all_acc / inner_steps

    def train_epoch(self):
        """train one epoch"""
        self.context.train_loss = 0

        train_loss, train_acc = 0, 0
        val_loss, val_acc = 0, 0

        self.metric.reset()
        self.optimizer.clear_grad()
        for _ in range(self.config.meta_batch_size):
            task = self.train_dataset.sample_task_set(
                ways=self.config.n_way,
                shots=self.config.k_shot
            )
            learner = self.learner.clone()

            # inner loop
            inner_loss, inner_acc = self.compute_loss(
                task.support_data, task.support_labels,
                learner,
                inner_steps=self.config.train_inner_adapt_steps
            )
            learner.adapt(inner_loss)
            train_loss += inner_loss.numpy()[0]
            train_acc += inner_acc

            # outer loop: compute loss on the validation dataset
            val_loss_, val_acc_ = self.compute_loss(
                task.query_data,
                task.query_labels,
                learner,
                inner_steps=self.config.train_inner_adapt_steps
            )
            val_loss += val_loss_
            val_acc += val_acc_

        self.context.train_loss, self.context.train_acc = train_loss / self.config.meta_batch_size, train_acc / self.config.meta_batch_size
        self.context.dev_loss, self.context.dev_acc = val_loss.numpy()[0] / self.config.meta_batch_size, val_acc / self.config.meta_batch_size

        self.optimizer.clear_grad()
        val_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def eval(self, dataset: CVDataset, learner: BaseLearner, mode: str = 'dev'):
        """eval the model on the dataset

        Args:
            dataset (CVDataset): the dataset to evaluate
            learner (BaseLearner): the learner to evaluate the model
            mode (str): the mode to evaluate, 'dev' or 'test'
        """
        logger.info(f'start doing {mode} on the dataset ...')
        eval_bar = tqdm(total=self.config.test_epoch, desc=f'{mode} Bar')
        test_loss, test_acc = 0, []
        for _ in range(self.config.test_epoch):
            epoch_acc = 0.0
            for _ in range(self.config.test_batch_size):

                task = dataset.sample_task_set(
                    ways=self.config.n_way,
                    shots=self.config.k_shot
                )
                learner = self.learner.clone()

                # inner loop
                inner_loss, inner_acc = self.compute_loss(
                    task.support_data, task.support_labels,
                    learner,
                    inner_steps=self.config.test_inner_adapt_steps
                )
                learner.adapt(inner_loss)

                # outer loop: compute loss on the validation dataset
                _, val_acc_ = self.compute_loss(
                    task.query_data,
                    task.query_labels,
                    learner,
                    inner_steps=self.config.test_inner_adapt_steps
                )
                epoch_acc += val_acc_

            test_acc.append(epoch_acc / self.config.test_batch_size)
            eval_bar.update()
            eval_bar.set_description(
                f'acc {test_acc[-1]:.6f}'
            )
        result = [
            test_loss / self.config.test_epoch * self.config.test_batch_size,
            min(test_acc),
            sum(test_acc) / len(test_acc),
            max(test_acc)
        ]
        logger.success(test_acc)
        logger.success("\n" + tabulate([result], ['loss', 'min-acc', 'mean-acc', 'max-acc'], tablefmt="grid"))
        if self.config.tracking_uri:
            self.client.log_metric(self.config.run_id, f'{mode}-loss', result[0])
            self.client.log_metric(self.config.run_id, f'{mode}-min-acc', result[1])
            self.client.log_metric(self.config.run_id, f'{mode}-mean-acc', result[2])
            self.client.log_metric(self.config.run_id, f'{mode}-max-acc', result[3])

    def train(self):
        """handle the main train"""
        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch()
            self.on_train_epoch_end()
            if epoch % self.config.do_eval_step == 0:
                self.eval(self.dev_dataset, self.learner, 'eval')

            if epoch % self.config.do_test_step ==0:
                self.eval(self.test_dataset, self.learner, 'test')
