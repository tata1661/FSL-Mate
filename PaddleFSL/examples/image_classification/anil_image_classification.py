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
from paddlefsl.model_zoo import anil

# Set computing device
paddle.set_device('gpu:0')


# """ ---------------------------------------------------------------------------------
# Config: ANIL, Omniglot, Conv, 5 Ways, 1 Shot
TRAIN_DATASET = paddlefsl.datasets.Omniglot(mode='train', image_size=(28, 28))
VALID_DATASET = paddlefsl.datasets.Omniglot(mode='valid', image_size=(28, 28))
TEST_DATASET = paddlefsl.datasets.Omniglot(mode='test', image_size=(28, 28))
WAYS = 5
SHOTS = 1
FEATURE_MODEL = paddlefsl.backbones.Conv(input_size=(1, 28, 28), output_size=WAYS, pooling=False)
FEATURE_MODEL.output = paddle.nn.Flatten()
HEAD_LAYER = paddle.nn.Linear(in_features=FEATURE_MODEL.feature_size, out_features=WAYS,
                              weight_attr=FEATURE_MODEL.init_weight_attr, bias_attr=FEATURE_MODEL.init_bias_attr)
META_LR = 0.005
INNER_LR = 0.5
ITERATIONS = 30000
TEST_EPOCH = 10
META_BATCH_SIZE = 32
TRAIN_INNER_ADAPT_STEPS = 1
TEST_INNER_ADAPT_STEPS = 3
APPROXIMATE = True
REPORT_ITER = 10
SAVE_MODEL_ITER = 5000
SAVE_MODEL_ROOT = '~/trained_models'
TEST_PARAM_FILE = 'iteration30000'
# ----------------------------------------------------------------------------------"""


train_dir, FEATURE_MODEL, HEAD_LAYER = anil.meta_training(train_dataset=TRAIN_DATASET,
                                                          valid_dataset=VALID_DATASET,
                                                          feature_model=FEATURE_MODEL,
                                                          head_layer=HEAD_LAYER,
                                                          meta_lr=META_LR,
                                                          inner_lr=INNER_LR,
                                                          iterations=ITERATIONS,
                                                          meta_batch_size=META_BATCH_SIZE,
                                                          ways=WAYS,
                                                          shots=SHOTS,
                                                          inner_adapt_steps=TRAIN_INNER_ADAPT_STEPS,
                                                          approximate=APPROXIMATE,
                                                          report_iter=REPORT_ITER,
                                                          save_model_iter=SAVE_MODEL_ITER,
                                                          save_model_root=SAVE_MODEL_ROOT)
print(train_dir)
anil.meta_testing(feature_model=FEATURE_MODEL,
                      head_layer=HEAD_LAYER,
                      test_dataset=TEST_DATASET,
                      test_epoch=TEST_EPOCH,
                      test_batch_size=META_BATCH_SIZE,
                      ways=WAYS,
                      shots=SHOTS,
                      inner_lr=INNER_LR,
                      inner_adapt_steps=TEST_INNER_ADAPT_STEPS,
                      approximate=APPROXIMATE)

