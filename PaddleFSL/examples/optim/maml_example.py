"""MAML example for optimization"""
from __future__ import annotations
from numpy import place

import paddle
from paddle import dtype, nn
from paddle.optimizer import Optimizer, Adam
from paddle.nn import Layer
from tap import Tap
from tqdm import tqdm, trange
import warnings

import paddlefsl
from paddlefsl import utils
from paddlefsl.datasets.cv_dataset import CVDataset
from paddlefsl.metaopt.base_learner import BaseLearner
from paddlefsl.metaopt.maml import MAMLLearner
from paddle.metric.metrics import Accuracy

class Config(Tap):
    """Alernative for Argument Parse"""
    n_way: int = 5
    k_shot: int = 1
    meta_lr: float = 0.02
    inner_lr: float = 0.002
    epochs: int = 60000                 # also named as iterations
    test_epoch: int = 10

    meta_batch_size: int = 32
    train_inner_adapt_steps: int = 5
    test_inner_adapt_steps: int = 10

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
        learner: BaseLearner,
        criterion: Layer,

    ) -> None:
        self.config = config

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.model = model.to(self.config.place())
        self.criterion = criterion
        self.learner = learner

        self.train_bar = tqdm(total=self.config.epochs, desc='Train Progress')
        self.context = ContextData()

        self.metric = Accuracy()

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

    def train_epoch(self):
        """train one epoch"""
        self.learner.clear_grad()
        self.context.train_loss = 0

        self.metric.reset()

        for _ in range(self.config.meta_batch_size):
            task = self.train_dataset.sample_task_set(
                ways=self.config.n_way,
                shots=self.config.k_shot
            )

            model = self.learner.clone()
            # model.to(device=self.config.place())
            support_set, support_set_labels = paddle.to_tensor(task.support_data, dtype=paddle.float32, place=self.config.place()), paddle.to_tensor(task.support_labels, dtype=paddle.int64, place=self.config.place())
            query_set, query_set_labels = paddle.to_tensor(task.query_data, dtype=paddle.float32, place=self.config.place()), paddle.to_tensor(task.query_labels, dtype=paddle.int64, place=self.config.place())

            # handle the fast adaption in few dataset
            for _ in range(self.config.train_inner_adapt_steps):
                logits = model(support_set)
                loss = self.criterion(logits, support_set_labels)
                self.learner.adapt(loss)    # accumulate the gradients

            # evaluate the model on query set data
            logits = model(query_set)
            loss = self.criterion(logits, query_set_labels)
            loss.backward()
            self.learner.step()

            self.context.train_loss += loss.detach().cpu().numpy().item()

            self.metric.update(
                self.metric.compute(logits, query_set_labels)
            )

        self.context.train_acc = self.metric.accumulate()

    def eval(self, dataset: CVDataset):
        """evaluate the model on the dataset"""
        self.learner.eval()
        self.model.eval()

        self.context.dev_loss = 0
        self.context.dev_acc = 0

        for _ in range(self.config.test_inner_adapt_steps):
            task = dataset.sample_task_set(
                ways=self.config.n_way,
                shots=self.config.k_shot
            )
            support_set, support_set_labels = paddle.to_tensor(task.support_data, dtype=paddle.float32), paddle.to_tensor(task.support_labels, dtype=paddle.int64)
            query_set, query_set_labels = paddle.to_tensor(task.query_data, dtype=paddle.float32), paddle.to_tensor(task.query_labels, dtype=paddle.int64)

            logits = self.model(support_set)
            loss = self.criterion(logits, support_set_labels)
            self.context.dev_loss += loss.numpy()

            logits = self.model(query_set)
            acc = paddle.fluid.layers.accuracy(input=logits, label=query_set_labels)
            self.context.dev_acc += acc.numpy()

        self.context.dev_loss /= self.config.test_inner_adapt_steps
        self.context.dev_acc /= self.config.test_inner_adapt_steps

        self.learner.train()
        self.model.train()

    def train(self):
        """handle the main train"""
        for _ in range(self.config.epochs):
            self.train_epoch()
            self.on_train_epoch_end()


if __name__ == '__main__':

    config = Config().parse_args(known_only=True)
    config.device = 'cpu'
    # Config: MAML, Mini-ImageNet, Conv, 5 Ways, 1 Shot
    train_dataset = paddlefsl.datasets.MiniImageNet(mode='train')
    valid_dataset = paddlefsl.datasets.MiniImageNet(mode='valid')
    test_dataset = paddlefsl.datasets.MiniImageNet(mode='test')

    model = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
    optimizer = Adam(
        learning_rate=config.meta_lr,
        parameters=model.parameters(),
    )
    learner = MAMLLearner(model, optimizer, learning_rate=config.inner_lr, approximate=config.approximate)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        dev_dataset=valid_dataset,
        test_dataset=test_dataset,
        model=model,
        learner=learner,
        criterion=criterion
    )
    trainer.train()
