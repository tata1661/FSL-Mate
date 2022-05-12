# Copyright 2021 PaddleFSL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddlefsl
from paddlefsl.model_zoo import baselinepp
import numpy as np
import random
import os

np.random.seed(66)
random.seed(66)
paddle.seed(66)

paddle.set_device('gpu:0')

# Config: BaselineppNet, Mini-ImageNet, Conv, 5 Ways, 1 Shot
WAYS = 5
QUERY_NUM = 16
NUM_WORKERS = 12
LOSS_FUCTION = paddle.nn.CrossEntropyLoss()
BACKBONE_FINDIM = 1600
TRAIN_OUTPUT_SIZE = 200
BACKBONE = paddle.nn.Sequential(paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=TRAIN_OUTPUT_SIZE,
                                                         conv_channels=[64, 64, 64, 64]).conv, paddle.nn.Flatten())

TRAIN_BATCHSIZE = 16
TRAIN_TRANSFORM = paddlefsl.model_zoo.baselinepp.get_baselinepp_transfer(mode='train', image_size=84)
TRAIN_DATASET = paddlefsl.datasets.MiniImageNetNoreseize(mode='train', transform=TRAIN_TRANSFORM)
TRAIN_LOADER = paddle.io.DataLoader(TRAIN_DATASET, batch_size=TRAIN_BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS)

BASELINEPP_MODEL_TRAIN = paddlefsl.backbones.Baselinepp(BACKBONE, TRAIN_OUTPUT_SIZE, BACKBONE_FINDIM)
LR_TRAIN = 0.001
OPTIMIZER_TRAIN = paddle.optimizer.Adam(learning_rate=LR_TRAIN, parameters=BASELINEPP_MODEL_TRAIN.parameters())
TRAIN_EPOCHS = 60
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 5

SAVE_MODEL_ROOT = '../trained_models'
TEST_PARAM_FILE = 'epoch60.params'
train_dir, BASELINEPP_MODEL_TRAIN = baselinepp.baselinepp_training(train_loader=TRAIN_LOADER,
                                                                   baselineppmodel=BASELINEPP_MODEL_TRAIN,
                                                                   lossfuc=LOSS_FUCTION,
                                                                   lr=LR_TRAIN,
                                                                   optimizer=OPTIMIZER_TRAIN,
                                                                   epochs=TRAIN_EPOCHS,
                                                                   ways=WAYS,
                                                                   report_epoch=REPORT_EPOCH,
                                                                   lr_step_epoch=LR_STEP_EPOCH,
                                                                   save_model_epoch=SAVE_MODEL_EPOCH,
                                                                   save_model_root=SAVE_MODEL_ROOT)
print(train_dir)

TEST_EPOCHS = 600
EPISODES = 100
INNERBATCH_SIZE = 4
TEST_OUTPUT_SIZE = 5
TEST_TRANSFORM = paddlefsl.model_zoo.baselinepp.get_baselinepp_transfer(mode='test', image_size=84)
TEST_DATASET = paddlefsl.datasets.MiniImageNetNoreseize(mode='test', transform=TEST_TRANSFORM)

BASELINEPP_MODEL_TEST = paddlefsl.backbones.Baselinepp(BACKBONE, TEST_OUTPUT_SIZE, BACKBONE_FINDIM)
BASELINEPP_MODEL_TRAIN.set_state_dict(paddle.load(os.path.join(train_dir, TEST_PARAM_FILE)))

BASELINEPP_MODEL_TEST.encoder.set_state_dict(BASELINEPP_MODEL_TRAIN.encoder.state_dict())

TEST_OPTIMIZER_DICT ={'optimizer':paddle.optimizer.Momentum,'param':{'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.01}}
# TEST:5way 1shot 16query
# change SHOTS to 5 for 5way 5shot 16query test
SHOTS = 1
BASELINEPP_MODEL_TEST = baselinepp.baselinepp_testing(
    test_dataset=TEST_DATASET,
    baselineppmodel=BASELINEPP_MODEL_TEST,
    lossfuc=LOSS_FUCTION,
    optimizerdict=TEST_OPTIMIZER_DICT,
    epochs=TEST_EPOCHS,
    episodes=EPISODES,
    innerbatch_size=INNERBATCH_SIZE,
    ways=WAYS,
    shots=SHOTS,
    query_num=QUERY_NUM)



