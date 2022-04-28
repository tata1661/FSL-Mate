"""MAML example for optimization"""
from __future__ import annotations
import os
import paddle
from paddle import nn
from paddle.optimizer import Adam
import paddlefsl
from paddlefsl.metaopt.anil import ANILLearner
from examples.optim.meta_trainer import Config, Trainer, load_datasets


def init_models(config: Config):
    """Initialize models."""
    if config.dataset == 'cub':
        config.meta_lr = 0.002
        config.inner_lr = 0.01
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 10000

        if config.k_shot == 5:
            config.meta_lr = 0.003
            config.inner_lr = 0.05
            config.epochs = 10000

        feature_model = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
        feature_model.output = paddle.nn.Flatten()
        head_layer = paddle.nn.Linear(in_features=feature_model.feature_size, out_features=config.n_way,
                                    weight_attr=feature_model.init_weight_attr, bias_attr=feature_model.init_bias_attr)
    
    if config.dataset == 'cifarfs':
        config.meta_lr = 0.001
        config.inner_lr = 0.02
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 20000
        if config.k_shot == 5:
            config.meta_lr = 0.001
            config.inner_lr = 0.08

        feature_model = paddlefsl.backbones.Conv(input_size=(3, 32, 32), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
        feature_model.output = paddle.nn.Flatten()
        head_layer = paddle.nn.Linear(in_features=32, out_features=config.n_way,
                                    weight_attr=feature_model.init_weight_attr, bias_attr=feature_model.init_bias_attr)
    
    if config.dataset == 'miniimagenet':

        config.meta_lr = 0.002
        config.inner_lr = 0.05
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 30000

        feature_model = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
        feature_model.output = paddle.nn.Flatten()
        head_layer = paddle.nn.Linear(in_features=feature_model.feature_size, out_features=config.n_way,
                                    weight_attr=feature_model.init_weight_attr, bias_attr=feature_model.init_bias_attr)
    
    if config.dataset == 'omniglot':
        config.meta_lr = 0.005
        config.inner_lr = 0.5

        if config.k_shot == 5:
            config.meta_lr = 0.06
            config.inner_lr = 0.12
            config.train_inner_adapt_steps = 3
            config.test_inner_adapt_steps = 5

        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 1
        config.test_inner_adapt_steps = 3
        config.epochs = 30000

        feature_model = paddlefsl.backbones.Conv(input_size=(1, 28, 28), output_size=config.n_way, pooling=False)
        feature_model.output = paddle.nn.Flatten()
        head_layer = paddle.nn.Linear(in_features=feature_model.feature_size, out_features=config.n_way,
                                    weight_attr=feature_model.init_weight_attr, bias_attr=feature_model.init_bias_attr)    
    
    if config.dataset == 'fc100':
        config.meta_lr = 0.005
        config.inner_lr = 0.1
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 5000
        if config.k_shot == 5:
            config.meta_lr = 0.002
            config.epochs = 2000

        feature_model = paddlefsl.backbones.Conv(input_size=(3, 32, 32), output_size=config.n_way)
        feature_model.output = paddle.nn.Flatten()
        head_layer = paddle.nn.Linear(in_features=feature_model.feature_size, out_features=config.n_way,
                                    weight_attr=feature_model.init_weight_attr, bias_attr=feature_model.init_bias_attr)

    return feature_model, head_layer
        

if __name__ == '__main__':

    config = Config().parse_args(known_only=True)
    config.device = 'gpu'
    config.k_shot = 5

    # config.dataset = 'omniglot'
    # config.dataset = 'miniimagenet'
    config.dataset = 'cifarfs'
    # config.dataset = 'fc100'
    # config.dataset = 'cub'

    config.tracking_uri = os.environ.get('TRACKING_URI', None)
    config.experiment_id = os.environ.get('EXPERIMENT_ID', None)

    # Config: ANIL, Omniglot, Conv, 5 Ways, 1 Shot
    train_dataset, valid_dataset, test_dataset = load_datasets(config.dataset)
    feature_model, head_layer = init_models(config)

    criterion = nn.CrossEntropyLoss()
    learner = ANILLearner(
        feature_model=feature_model,
        head_layer=head_layer,
        learning_rate=config.inner_lr,
    )
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=config.meta_lr, T_max=config.epochs)
    optimizer = Adam(parameters=learner.parameters(), learning_rate=scheduler)
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        dev_dataset=valid_dataset,
        test_dataset=test_dataset,
        learner=learner,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion
    )
    trainer.train()
