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

import json
from abc import ABC
from typing import List


class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(self,
                 device: str = None,
                 per_gpu_train_batch_size: int = 8,
                 n_gpu: int = 1,
                 num_train_epochs: int = 3,
                 max_steps: int = -1,
                 gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0,
                 learning_rate: float = 5e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 max_grad_norm: float = 1,
                 alpha: float = 0.9999):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param alpha: the alpha parameter for auxiliary language modeling
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(self,
                 device: str = None,
                 n_gpu: int = 1,
                 per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = None):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics



class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self,
                 model_type: str,
                 model_name_or_path: str,
                 task_name: str,
                 max_seq_length: int,
                 label_list: List[str],
                 pattern_id: int = 0,
                 cache_dir: str = None,
                 output_dir=None,
                 embed_size=128,
                 prompt_encoder_type="lstm",
                 eval_every_step=20):

        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.embed_size = embed_size
        self.prompt_encoder_type = prompt_encoder_type
        self.eval_every_step = eval_every_step
