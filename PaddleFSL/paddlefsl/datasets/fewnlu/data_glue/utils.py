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

import copy
import json
import pickle
import random
import string
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
from transformers import PreTrainedTokenizer, GPT2Tokenizer


class LogitsList:
    """A list of logits obtained from a finetuned PET model"""

    def __init__(self, score: float, logits: List[List[float]]):
        """
        Create a new LogitsList.
        :param score: the corresponding PET model's score on the training set
        :param logits: the list of logits, where ``logits[i][j]`` is the score for label ``j`` at example ``i``
        """
        self.score = score
        self.logits = logits

    def __repr__(self):
        return 'LogitsList(score={}, logits[:2]={})'.format(self.score, self.logits[:2])

    def save(self, path: str) -> None:
        """Save this list to a file."""
        with open(path, 'w') as fh:
            fh.write(str(self.score) + '\n')
            for example_logits in self.logits:
                fh.write(' '.join(str(logit) for logit in example_logits) + '\n')

    @staticmethod
    def load(path: str, with_score: bool = True) -> 'LogitsList':
        """Load a list from a file"""
        score = -1
        logits = []
        with open(path, 'r') as fh:
            for line_idx, line in enumerate(fh.readlines()):
                line = line.rstrip('\n')
                if line_idx == 0 and with_score:
                    score = float(line)
                else:
                    logits.append([float(x) for x in line.split()])
        return LogitsList(score=score, logits=logits)


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta: Optional[Dict] = None, idx=-1):
        """
        Create a new InputExample.
        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(self, input_ids, attention_mask, token_type_ids, label, mlm_labels=None, logits=None,
                 meta: Optional[Dict] = None, idx=-1, block_flag=None):
        """
        Create new InputFeatures.
        :param input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param mlm_labels: an optional sequence of labels used for auxiliary language modeling
        :param logits: an optional sequence of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.mlm_labels = mlm_labels
        self.logits = logits
        self.idx = idx
        self.block_flag = block_flag
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'mlm_labels        = {self.mlm_labels}\n' + \
               f'logits            = {self.logits}\n' + \
               f'label             = {self.label}\n' + \
               f'block_flag         = {self.block_flag}'

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PLMInputFeatures(InputFeatures):
    """A set of numeric input features for a model pretrained with a permuted language modeling objective."""

    def __init__(self, *_, perm_mask, target_mapping, **kwargs):
        super().__init__(**kwargs)
        self.perm_mask = perm_mask
        self.target_mapping = target_mapping

    def pretty_print(self, tokenizer):
        return super().pretty_print(tokenizer) + '\n' + \
               f'perm_mask         = {self.perm_mask}\n' + \
               f'target_mapping    = {self.target_mapping}'



def get_verbalization_ids(word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization
    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, **kwargs)['input_ids'][1:-1]
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id





