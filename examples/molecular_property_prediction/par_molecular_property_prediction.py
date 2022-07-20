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
from paddlefsl.model_zoo import par
import paddlefsl.datasets as datasets
from utils import get_args, Logger
import os

# Set computing device
paddle.set_device('gpu:0')


# """ ---------------------------------------------------------------------------------
# Config: PAR, tox21, GIN, 2 Ways, 10 Shot
METHOD = 'par'
DATASET = 'tox21'
TRAIN_DATASET = datasets.mol_dataset.load_dataset(dataset = DATASET, type = 'train')
TEST_DATASET = datasets.mol_dataset.load_dataset(dataset = DATASET, type = 'test')
SHOT = 10
N_QUERY = 16
META_LR = 0.001
WEIGHT_DECAY = 5e-5
INNER_LR = 0.05
EPOCHS = 1000
EVAL_STEPS = 10
SEED = 1
SAVE_MODEL_ITER = 2000
# ----------------------------------------------------------------------------------"""
args = get_args(root_dir = os.path.abspath(os.path.dirname(__file__)),
                n_shot = SHOT,
                n_query = N_QUERY,
                meta_lr = META_LR,
                weight_decay = WEIGHT_DECAY,
                inner_lr = INNER_LR,
                epochs = EPOCHS,
                eval_steps = EVAL_STEPS,
                seed = SEED,
                save_model_iter = SAVE_MODEL_ITER,
                method = METHOD)
LOGGER = Logger(args.trial_path + '/results.txt')
par.run(args, 
        train_dataset = TRAIN_DATASET,
        test_dataset = TEST_DATASET,
        logger = LOGGER)
