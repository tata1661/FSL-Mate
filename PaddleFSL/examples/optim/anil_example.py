"""MAML example for optimization"""
from __future__ import annotations
from turtle import forward
from typing import Tuple
import warnings
import math

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
from paddlefsl.backbones.conv import ConvBlock
from paddlefsl.metaopt.base_learner import BaseLearner
from paddlefsl.metaopt.anil import ANILLearner


class ConvEncoder(nn.Layer):
    """
    Implementation of a CNN(Convolutional Neural Network) model.

    Args:
        input_size(Tuple): input size of the image. The tuple must have 3 items representing channels, width and
            height of the image, for example (1, 28, 28) for Omniglot images.
        output_size(int): size of the output.
        conv_channels(List, optional): channel numbers of the hidden layers, default None. If None, it will be set
            [64, 64, 64, 64]. Please note that there is a final classifier after all hidden layers, so the last
            item of hidden_sizes is not the output size.
        kernel_size(Tuple, optional): convolutional kernel size, default (3, 3).
        pooling(bool, optional): whether to do max-pooling after each convolutional layer, default True. If False,
            the model use stride convolutions instead of max-pooling. If True, the pooling kernel size will be (2, 2).

    Examples:
        ..code-block:: python

            train_input = paddle.ones(shape=(1, 1, 28, 28), dtype='float32')
            conv = Conv(input_size=(1, 28, 28), output_size=2)
            print(conv(train_input)) # A paddle.Tensor with shape=[1, 2]

    """
    init_weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
    init_bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0))

    def __init__(self,
                 input_size,
                 output_size,
                 conv_channels=None,
                 kernel_size=(3, 3),
                 pooling=True):
        super(ConvEncoder, self).__init__()
        # Convolution layers
        conv_channels = [64, 64, 64, 64] if conv_channels is None else conv_channels
        if len(input_size) != 3:
            raise ValueError('The input_size must have 3 items representing channels, width and height of the image.')
        in_channels, out_channels = input_size[0], conv_channels[0]
        pooling_size = (2, 2) if pooling else None
        stride_size = (2, 2) if not pooling else (1, 1)
        self.conv = nn.Sequential(
            ('conv0', ConvBlock(in_channels, out_channels, kernel_size, pooling_size, stride_size))
        )
        for i in range(1, len(conv_channels)):
            in_channels, out_channels = out_channels, conv_channels[i]
            self.conv.add_sublayer(name='conv' + str(i),
                                   sublayer=ConvBlock(in_channels, out_channels,
                                                      kernel_size, pooling_size, stride_size))
        # Output layer
        if pooling:
            features = (input_size[1] >> len(conv_channels)) * (input_size[2] >> len(conv_channels))
        else:
            width, height = input_size[1], input_size[2]
            for i in range(len(conv_channels)):
                width, height = math.ceil(width / 2), math.ceil(height / 2)
            features = width * height
        self.feature_size = features * conv_channels[-1]

    def forward(self, inputs):
        return self.conv(inputs)


class ConvHead(nn.Layer):
    """Classifier Head for ConvNet."""
    def __init__(self, feature_size: int, output_size: int):
        super().__init__()
        self.head_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                feature_size,
                output_size,
                weight_attr=ConvEncoder.init_weight_attr,
                bias_attr=ConvEncoder.init_bias_attr
            )
        )

    def forward(self, inputs):
        """handle forward computation."""
        return self.head_layer(inputs)


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

        feature_model: Layer,
        head_layer: Layer,

        criterion: Layer,

    ) -> None:
        self.config = config

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
        self.criterion = criterion
        self.learner = ANILLearner(
            feature_model=feature_model,
            head_layer=head_layer,
            learning_rate=self.config.meta_lr,
            approximate=self.config.approximate
        )

        self.train_bar = tqdm(total=self.config.epochs, desc='Train Progress')
        self.context = ContextData()

        self.metric = Accuracy()
        
        self.optimizer = Adam(
            parameters=list(feature_model.parameters()) + list(head_layer.parameters()),
            learning_rate=self.config.meta_lr
        )

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

    feature_model = ConvEncoder(
        input_size=(3, 84, 84),
        output_size=config.n_way,
        conv_channels=[32, 32, 32, 32]
    )
    head_layer = ConvHead(
        feature_size=feature_model.feature_size,
        output_size=config.n_way
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        dev_dataset=valid_dataset,
        test_dataset=test_dataset,
        feature_model=feature_model,
        head_layer=head_layer,
        criterion=criterion
    )
    trainer.train()
