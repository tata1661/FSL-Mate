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
from paddlefsl.model_zoo import gnn

# Set computing device
paddle.set_device('gpu:0')

# Config: GNN, Mini-ImageNet, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.MiniImageNet(mode='train')
VALID_DATASET = paddlefsl.datasets.MiniImageNet(mode='valid')
TEST_DATASET = paddlefsl.datasets.MiniImageNet(mode='test')
WAYS = 5
SHOTS = 1
QUERY_NUM = 16
hidden_size = 128
FEATURE_MODEL = paddlefsl.backbones.EmbeddingImagenet(emb_size=hidden_size)
GNN_MODEL = paddlefsl.backbones.gnn_iclr.GNN(WAYS, hidden_size)
LR = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.5)
OPTIMIZER = paddle.optimizer.Adam(learning_rate=LR,
                                  parameters=FEATURE_MODEL.parameters() + GNN_MODEL.parameters())

EPOCHS = 100
TEST_EPOCHS = 10
EPISODES = 1000
REPORT_EPOCH = 1
LR_STEP_EPOCH = 10
SAVE_MODEL_EPOCH = 20
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'epoch100.params'

train_dir, FEATURE_MODEL, GNN_MODEL = gnn.meta_training(train_dataset=TRAIN_DATASET,
                                                        valid_dataset=VALID_DATASET,
                                                        feature_model=FEATURE_MODEL,
                                                        gnn_model=GNN_MODEL,
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
gnn.meta_testing(feature_model=FEATURE_MODEL,
                 gnn_model=GNN_MODEL,
                 test_dataset=TEST_DATASET,
                 epochs=TEST_EPOCHS,
                 episodes=EPISODES,
                 ways=WAYS,
                 shots=SHOTS,
                 query_num=QUERY_NUM)
