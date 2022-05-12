# Copyright 2022 PaddleFSL Authors
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
from paddlefsl.model_zoo import baseline

# Set computing device
paddle.set_device('gpu:0')

# Config: MatchingNet, Mini-ImageNet, Conv, 5 Ways, 5 Shots
TRAIN_DATASET = paddlefsl.datasets.MiniImageNet(mode='train')
VALID_DATASET = paddlefsl.datasets.MiniImageNet(mode='valid')
TEST_DATASET = paddlefsl.datasets.MiniImageNet(mode='test')
WAYS = 5
SHOTS = 5
QUERY_NUM = 16
NUM_CLASS = 200
MODEL = paddlefsl.backbones.Conv(input_size=(3, 84, 84), output_size=WAYS, conv_channels=[64, 64, 64, 64])
MODEL.output = paddle.nn.Flatten()
CLASSIFIER = paddlefsl.backbones.distLinear(MODEL.feature_size, NUM_CLASS)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=MODEL.parameters() + CLASSIFIER.parameters())
EPOCHS = 100
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'

train_dir, MODEL, CLASSIFIER = baseline.meta_training(train_dataset=TRAIN_DATASET,
                                                      valid_dataset=VALID_DATASET,
                                                      model=MODEL,
                                                      classifier=CLASSIFIER,
                                                      lr=LR,
                                                      optimizer=OPTIMIZER,
                                                      epochs=EPOCHS,
                                                      episodes=EPISODES,
                                                      ways=WAYS,
                                                      shots=SHOTS,
                                                      query_num=QUERY_NUM,
                                                      report_epoch=REPORT_EPOCH,
                                                      lr_step_epoch=LR_STEP_EPOCH,
                                                      save_model_epoch=SAVE_MODEL_EPOCH,
                                                      save_model_root=SAVE_MODEL_ROOT)
print(train_dir)

TEST_EPOCHS = 100
EPISODES = 600
CLASSIFIER_TEST = paddlefsl.backbones.distLinear(MODEL.feature_size, WAYS)
OPTIMIZER = paddle.optimizer.Momentum(learning_rate=0.01,
                                      momentum=0.9,
                                      parameters=CLASSIFIER_TEST.parameters(),
                                      weight_decay=0.001)
baseline.meta_testing(model=MODEL,
                      classifier=CLASSIFIER_TEST,
                      test_dataset=TEST_DATASET,
                      optimizer=OPTIMIZER,
                      epochs=TEST_EPOCHS,
                      episodes=EPISODES,
                      ways=WAYS,
                      shots=SHOTS,
                      query_num=QUERY_NUM)
