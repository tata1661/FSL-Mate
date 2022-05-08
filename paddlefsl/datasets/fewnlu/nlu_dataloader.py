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

import os
import json
import numpy as np

import paddle
from paddlenlp.utils.log import logger


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def convert_example(example,
                    preprocess,
                    lan):
    
    if lan == 'zh':
        items = preprocess(example)
        try:
            src_ids, token_type_ids, mask_positions, mask_lm_labels = items
        except:
            src_ids, token_type_ids, mask_positions, mask_lm_labels, candidate_labels_ids = items
            return src_ids, token_type_ids, mask_positions, mask_lm_labels, candidate_labels_ids
    else:
        # Preprocessing the input
        input_features, mask_positions = preprocess.get_input_features(example, True)
        
        src_ids, token_type_ids, mask_lm_labels = input_features.input_ids, \
        input_features.token_type_ids, input_features.mlm_labels
        mask_positions = [mask_positions]
    
    return src_ids, token_type_ids, mask_positions, mask_lm_labels
        
