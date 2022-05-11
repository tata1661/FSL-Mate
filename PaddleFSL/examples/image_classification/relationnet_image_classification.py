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
from paddlefsl.model_zoo import relationnet


# Set computing device
paddle.set_device('gpu:1')


# Config: RelationNet, FC100, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.FC100(mode='train')
VALID_DATASET = paddlefsl.datasets.FC100(mode='valid')
TEST_DATASET = paddlefsl.datasets.FC100(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 15
EMBEDDING_MODEL = paddlefsl.backbones.ConvEmbedModel(input_size=(3, 32, 32))
RELATION_MODEL = paddlefsl.backbones.ConvRelationModel(input_size=EMBEDDING_MODEL.output_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=EMBEDDING_MODEL.parameters() + RELATION_MODEL.parameters())
EPOCHS = 30
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 10
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch30'




train_dir, EMBEDDING_MODEL, RELATION_MODEL = relationnet.meta_training(train_dataset=TRAIN_DATASET,
                                                                       valid_dataset=VALID_DATASET,
                                                                       embedding_model=EMBEDDING_MODEL,
                                                                       relation_model=RELATION_MODEL,
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
relationnet.meta_testing(embedding_model=EMBEDDING_MODEL,
                         relation_model=RELATION_MODEL,
                         test_dataset=TEST_DATASET,
                         epochs=TEST_EPOCHS,
                         episodes=EPISODES,
                         ways=WAYS,
                         shots=SHOTS,
                         query_num=QUERY_NUM)