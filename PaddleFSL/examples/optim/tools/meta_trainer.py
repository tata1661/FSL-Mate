"""MAML example for optimization"""
from __future__ import annotations
from typing import Optional, Tuple
from loguru import logger

import paddle
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler
from paddle.nn import Layer
from paddle.metric.metrics import Accuracy
from tap import Tap
from tqdm import tqdm
import numpy as np

import paddlefsl
from paddlefsl.datasets.cv_dataset import CVDataset
from paddlefsl.metaopt.base_learner import BaseLearner
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
    dataset: str = ''
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

    def _set_device(self):
        paddle.device.set_device(self.config.device)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

    def on_train_epoch_end(self):
        """handle the end of one epoch"""
        self.context.epoch += 1

        self.train_bar.update()

        bar_info = f'Epoch: {self.context.epoch}/{self.config.epochs} \t train-loss: {self.context.train_loss}  \t\t train-acc: {self.context.train_acc}'
        self.train_bar.set_description(bar_info)

    def compute_loss(self, input_data, labels, learner: BaseLearner):
        """compute the loss based on the input_data and labels"""
        input_data, labels = paddle.to_tensor(input_data, dtype='float32'), paddle.to_tensor(labels, dtype='int64')

        logits = learner(input_data)
        loss = self.criterion(logits, labels)

        acc = self.metric.compute(logits, labels)
        acc = self.metric.update(acc)
        
        return loss, acc

    def train_epoch(self):
        """train one epoch"""
        self.learner.train()

        self.context.train_loss = 0

        train_loss, train_acc = 0, 0

        self.metric.reset()
        self.optimizer.clear_grad()
        for _ in range(self.config.meta_batch_size):
            task = self.train_dataset.sample_task_set(
                ways=self.config.n_way,
                shots=self.config.k_shot
            )
            learner = self.learner.clone()

            # inner loop
            for _ in range(self.config.train_inner_adapt_steps):
                inner_loss, _ = self.compute_loss(
                    task.support_data, task.support_labels, learner
                )
                learner.adapt(inner_loss)

            # outer loop: compute loss on the validation dataset
            loss, acc = self.compute_loss(
                task.query_data, task.query_labels, learner
            )
            train_loss += loss
            train_acc += acc

        self.optimizer.clear_grad()
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.context.train_loss, self.context.train_acc = train_loss.numpy()[0] / self.config.meta_batch_size, train_acc / self.config.meta_batch_size

    def eval(self, dataset: CVDataset, learner: BaseLearner, mode: str = 'dev'):
        """eval the model on the dataset

        Args:
            dataset (CVDataset): the dataset to evaluate
            learner (BaseLearner): the learner to evaluate the model
            mode (str): the mode to evaluate, 'dev' or 'test'
        """
        logger.info(f'start doing {mode} on the dataset ...')
        eval_bar = tqdm(total=self.config.test_epoch, desc=f'{mode} Bar')
        test_loss, test_acc = [], []
        for _ in range(self.config.test_epoch):
            val_loss, val_acc = 0.0, 0.0
            for _ in range(self.config.test_batch_size):

                task = dataset.sample_task_set(
                    ways=self.config.n_way,
                    shots=self.config.k_shot
                )
                learner = self.learner.clone()

                # inner loop
                for _ in range(self.config.test_inner_adapt_steps):
                    inner_loss, _ = self.compute_loss(
                        task.support_data, task.support_labels, learner
                    )
                    learner.adapt(inner_loss)

                # outer loop: compute loss on the validation dataset
                loss, acc = self.compute_loss(
                    task.query_data, task.query_labels, learner,
                )
                val_loss += loss.numpy()[0]
                val_acc += acc
            
            test_acc.append(val_acc / self.config.test_batch_size)
            test_loss.append(val_loss / self.config.test_batch_size)

            eval_bar.update()
            eval_bar.set_description(
                f'acc {test_acc[-1]:.6f}'
            )
        mean_loss, std_loss = np.mean(test_loss), np.std(test_loss)
        mean_acc, std_acc = np.mean(test_acc), np.std(test_acc)
        
        logger.success(f'======================Epoch: {self.context.epoch}/{self.config.epochs}-{mode}======================')
        logger.success(f'mean-loss: {mean_loss:.6f}, std-loss: {std_loss:.6f}')
        logger.success(f'mean-acc: {mean_acc:.6f}, std-acc: {std_acc:.6f}')
        logger.success('================================================')

    def train(self):
        """handle the main train"""
        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch()
            self.on_train_epoch_end()
            if epoch % self.config.do_eval_step == 0:
                self.eval(self.dev_dataset, self.learner, 'eval')

        self.eval(self.test_dataset, self.learner, 'test')
