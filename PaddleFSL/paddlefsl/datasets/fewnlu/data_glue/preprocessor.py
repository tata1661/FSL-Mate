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

from abc import ABC, abstractmethod
from paddlefsl.datasets.fewnlu.data_glue.utils import InputFeatures, InputExample
from paddlefsl.datasets.data_glue.task_pvps import PVP, PVPS


class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:`InputExample` into a :class:`InputFeatures` object so that it can be
    processed by the model being used.
    """

    def __init__(self, wrapper, task_name, pattern_id: int = 0):
        """
        Create a new preprocessor.

        :param wrapper: the wrapper for the language model to use
        :param task_name: the name of the task
        :param pattern_id: the id of the PVP to be used
        :param verbalizer_file: path to a file containing a verbalizer that overrides the default verbalizer
        """
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id)  # type: PVP
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class MLMPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""

    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:

        input_ids, token_type_ids, block_flag = self.pvp.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        block_flag = block_flag + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length
        assert len(block_flag) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100
        logits = example.logits if example.logits else [-1]

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             label=label,
                             mlm_labels=mlm_labels,
                             logits=logits,
                             idx=example.idx,
                             block_flag=block_flag)
