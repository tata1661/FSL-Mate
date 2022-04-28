"""MAML example for optimization"""
import os

import paddle
from paddle import nn
from paddle.optimizer import Adam

import paddlefsl
from paddlefsl.metaopt.maml import MAMLLearner
from examples.optim.meta_trainer import Config, Trainer, load_datasets



def init_models(config: Config):
    """Initialize models."""
    if config.dataset == 'cub':
        config.meta_lr = 0.002
        config.inner_lr = 0.03
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 10000

        model = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
    
    if config.dataset == 'cifarfs':
        config.meta_lr = 0.001
        config.inner_lr = 0.03
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 30000

        model = paddlefsl.backbones.Conv(input_size=(3, 32, 32), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
        model.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=config.n_way,
                      weight_attr=model.init_weight_attr, bias_attr=model.init_bias_attr)
        )
    
    if config.dataset == 'miniimagenet':
        config.meta_lr = 0.002
        config.inner_lr = 0.03
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 60000

        model = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=config.n_way, conv_channels=[32, 32, 32, 32])
    
    if config.dataset == 'omniglot':
        config.meta_lr = 0.005
        config.inner_lr = 0.5
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 1
        config.test_inner_adapt_steps = 3
        config.epochs = 30000

        model = paddlefsl.backbones.Conv(input_size=(1, 28, 28), output_size=config.n_way, pooling=False)
    
    if config.dataset == 'fc100':
        config.meta_lr = 0.002
        config.inner_lr = 0.05
        config.test_epoch = 10
        config.meta_batch_size = 32
        config.train_inner_adapt_steps = 5
        config.test_inner_adapt_steps = 10
        config.epochs = 5000

        model = paddlefsl.backbones.Conv(input_size=(3, 32, 32), output_size=config.n_way)

    return model 


if __name__ == '__main__':

    config = Config().parse_args(known_only=True)
    config.device = 'gpu'
    # config.dataset = 'omniglot'
    # config.dataset = 'miniimagenet'
    # config.dataset = 'cifarfs'
    # config.dataset = 'fc100'
    config.dataset = 'cub'

    config.tracking_uri = os.environ.get('TRACKING_URI', None)
    config.experiment_id = os.environ.get('EXPERIMENT_ID', None)

    train_dataset, valid_dataset, test_dataset = load_datasets(config.dataset)
    model = init_models(config)

    criterion = nn.CrossEntropyLoss()
    learner = MAMLLearner(
        module=model,
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