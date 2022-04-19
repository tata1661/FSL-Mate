"""MAML example for optimization"""
from __future__ import annotations
from typing import Tuple
import warnings

import paddle
from paddle import nn
from paddle.optimizer import Adam
from paddle.nn import Layer
from paddle.metric.metrics import Accuracy
from tap import Tap
from tqdm import tqdm
from tabulate import tabulate

import paddlefsl
from paddlefsl.datasets.cv_dataset import CVDataset
from paddlefsl.metaopt.base_learner import BaseLearner
from paddlefsl.metaopt.maml import MAMLLearner


class Config(Tap):
    """Alernative for Argument Parse"""
    n_way: int = 5
    k_shot: int = 1
    meta_lr: float = 0.02
    inner_lr: float = 0.002
    epochs: int = 60000                 # also named as iterations
    test_epoch: int = 10

    meta_batch_size: int = 32
    test_batch_size: int = 10
    
    train_inner_adapt_steps: int = 5

    approximate: bool = True

    do_eval_step: int = 30
    save_model_iter: int = 5000
    save_model_root: str = '~/trained_models'
    test_param_file: str = 'iteration60000.params'

    device: str = 'cpu'

    def place(self): 
        """get the default device place for tensor"""
        return paddle.fluid.CUDAPlace(0) if self.device == 'gpu' else paddle.fluid.CPUPlace()

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

        model: Layer,
        criterion: Layer,

    ) -> None:
        self.config = config

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
        self.model = model
        self.criterion = criterion
        self.learner = MAMLLearner(
            self.model,
            learning_rate=self.config.meta_lr,
            approximate=self.config.approximate
        )

        self.train_bar = tqdm(total=self.config.epochs, desc='Train Progress')
        self.context = ContextData()

        self.metric = Accuracy()
        self.optimizer = Adam(parameters=self.model.parameters(), learning_rate=self.config.meta_lr)

        self._set_device()
        warnings.filterwarnings("ignore")

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

        # if self.context.epoch % self.config.do_eval_step == 0:
        #     self.eval(self.dev_dataset)

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

    def train_epoch(self):
        """train one epoch"""
        self.context.train_loss = 0

        self.metric.reset()
        
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
            inner_loss, inner_acc = self.fast_adapt(task, learner)
            train_loss += inner_loss.detach().clone().cpu().numpy().item()
            train_acc += inner_acc

            inner_loss.backward()

            # outer loop
            task = self.dev_dataset.sample_task_set(
                ways=self.config.n_way,
                shots=self.config.k_shot
            )
            outer_loss, outer_acc = self.fast_adapt(task, learner)
            val_loss += outer_loss.detach().clone().cpu().numpy().item()
            val_acc += outer_acc
        
        self.context.train_loss, self.context.train_acc = train_loss / self.config.meta_batch_size, train_acc / self.config.meta_batch_size
        self.context.dev_loss, self.context.dev_acc = val_loss / self.config.meta_batch_size, val_acc / self.config.meta_batch_size
        self.optimizer.step()

    def eval(self, dataset: CVDataset, learner: BaseLearner):
        """eval the model on the dataset

        Args:
            dataset (CVDataset): the dataset to evaluate
            learner (BaseLearner): the learner to evaluate the model
        """

        self.context.dev_loss = 0
        self.context.dev_acc = 0

        eval_bar = tqdm(total=self.config.test_epoch, desc='Dev Bar')
        all_metric = []
        for _ in range(self.config.test_epoch):
            test_loss, test_acc = 0, 0
            for _ in range(self.config.test_batch_size):
                learner = learner.clone()
                task = dataset.sample_task_set(ways=self.config.n_way, shots=self.config.k_shot)
                inner_loss, inner_acc = self.fast_adapt(task, learner)
                test_loss += inner_loss.detach().clone().cpu().numpy().item()
                test_acc += inner_acc
            
            test_acc = test_acc / self.config.test_batch_size
            test_loss = test_loss / self.config.test_batch_size
            eval_bar.update()
            eval_bar.set_description(
                f'\tloss {test_loss:.6f}\t\tacc {test_acc:.6f}'
            )
            all_metric.append([test_loss, test_acc])
        print(tabulate(all_metric, ['loss', 'acc'], tablefmt="grid"))

    def train(self):
        """handle the main train"""
        for epoch in range(self.config.epochs):
            self.train_epoch()
            self.on_train_epoch_end()
            if (epoch + 1) % 100 == 0:
                self.eval(self.test_dataset, self.learner)


if __name__ == '__main__':

    config = Config().parse_args(known_only=True)
    config.device = 'gpu'
    # Config: MAML, Mini-ImageNet, Conv, 5 Ways, 1 Shot
    train_dataset = paddlefsl.datasets.MiniImageNet(mode='train')
    valid_dataset = paddlefsl.datasets.MiniImageNet(mode='valid')
    test_dataset = paddlefsl.datasets.MiniImageNet(mode='test')

    model = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
    optimizer = Adam(
        learning_rate=config.meta_lr,
        parameters=model.parameters(),
    )
    learner = MAMLLearner(model,learning_rate=config.inner_lr, approximate=config.approximate)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        dev_dataset=valid_dataset,
        test_dataset=test_dataset,
        model=model,
        criterion=criterion
    )
    trainer.train()
