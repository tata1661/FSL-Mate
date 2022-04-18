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

from . import NLPDataset
import paddlefsl.utils as utils
from paddlefsl.task_sampler import TaskSet
import json
import random
from tqdm import tqdm

__all__ = ['FewRel']


class FewRel(NLPDataset):
    """
    Implementation of FewRel1.0 dataset.
    FewRel is a large-scale few-shot relation extraction dataset, which contains more than one hundred
    relations and tens of thousands of annotated instances cross different domains.[1]

    Refs:
        1. Xu Han et al., 2018. "FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset
            with State-of-the-Art Evaluation." EMNLP.

    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        root(str, optional): root directory of the subdirectory '/${module_name}' which contains data files,
            can be set None. If None, it will be set default root: '<path to>/paddlefsl/../raw_data'.
        max_len(int, optional): maximum token number of a single sentence/text, can be set None. Default None.
            If not None, this code will pad or truncate every sentence/text to fit the max_len.
        backend(str, optional): specify the type of the dataset information. Should be one of ['numpy', 'text'].
            Default 'numpy'. If 'numpy', this code will initialize the texts into vectors using vector_initializer.
        vector_initializer(paddlefsl.backbones.RCInitVector, optional): if backend is 'numpy', users will have to
            specify a vector initializer to initialize the texts into vectors. Default None.

    Examples:
        ..code-block:: python

            from paddlefsl.datasets import FewRel
            from paddlefsl.backbones import RCInitVector

            init_vector = RCInitVector()

            train_dataset = FewRel(mode='train', max_len=100, vector_initializer=init_vector)
            valid_dataset = FewRel(mode='valid', backend='text')
            print(train_dataset[0])
            print(len(valid_dataset))
            taskset = train_dataset.sample_task_set(ways=5, shots=1)
            print(taskset.support_data.shape)
            data, label = train_dataset[10]
            label_name = train_dataset.get_label_name(label)
            print(label, label_name)

    """
    file_name = {
        'label_names': 'pid2name.json',
        'train': 'train_wiki.json',
        'valid': 'val_wiki.json'
    }
    file_url = {
        'label_names': 'https://raw.githubusercontent.com/thunlp/FewRel/master/data/pid2name.json',
        'train': 'https://raw.githubusercontent.com/thunlp/FewRel/master/data/train_wiki.json',
        'valid': 'https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json'
    }
    file_md5 = {
        'label_names': '0fe4c5a197c83668d1c7ef255cba5474',
        'train': '5e663e9c3f1bfbdb2de72696e9504fd7',
        'valid': '3f25573428c0332cb64b367a275ab0c7'
    }

    def __init__(self,
                 mode,
                 root=None,
                 max_len=None,
                 backend='numpy',
                 vector_initializer=None):
        super(FewRel, self).__init__(
            module_name='few-rel',
            mode=mode,
            root=root,
            max_len=max_len,
            backend=backend,
            vector_initializer=vector_initializer
        )

        # Check exist and download the data file
        self._check_and_download_file()
        self.label_names, self._label_data_dict = self._load_data()

        # Transfer label-data dictionary into data list and label list
        self._data_list, self._label_list = self._transfer_list()

    def __getitem__(self, idx):
        return self._data_list[idx], self._label_list[idx]

    def __len__(self):
        return len(self._data_list)

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        if not hasattr(self.vector_initializer, '__call__'):
            raise ValueError("When calling 'sample_task_set', 'vector_initializer' is required "
                             "to be paddlefsl.backbones.RCInitVector to generate word vector numpy array")
        query_num = shots if query_num is None else query_num
        # result: List[ (str(label name), List[np.ndarray(image)]) ]
        result = []
        labels = random.sample(self._label_data_dict.keys(), ways)
        for label in labels:
            data_list = self._label_data_dict[label]
            index = random.sample(range(len(data_list)), shots + query_num)
            result.append((label, [data_list[i] for i in index]))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def get_label_name(self, label):
        name = self.label_names[label]
        return name

    def _check_and_download_file(self):
        # Label pid to name file
        utils.download_url(self.root, self.file_url['label_names'], self.file_md5['label_names'])
        # Data file
        utils.download_url(self.root, self.file_url[self.mode], self.file_md5[self.mode])

    def _load_data(self):
        with open(self.root + '/' + self.file_name['label_names'], 'r') as label_names_file:
            # label_names: dict {pid: list[relation, describe]}
            label_names = json.load(label_names_file)
        with open(self.root + '/' + self.file_name[self.mode], 'r') as data_file:
            data = json.load(data_file)
        if self.backend == 'text':
            return label_names, data
        print('Initializing relation classification word vectors...')
        label_data_dict = {}
        for label, texts in tqdm(data.items()):
            vectors_list = []
            for text in texts:
                # text: {'tokens': list of tokens, 'h': ['head words', 'head id', [head position list] ]}
                tokens, head_position, tail_position = text['tokens'], text['h'][2][0], text['t'][2][0]
                vectors_list.append(self.vector_initializer(tokens, head_position, tail_position, self.max_len))
            label_data_dict[label] = vectors_list
        print('Word vector initialization done.')
        return label_names, label_data_dict

    def _transfer_list(self):
        data_list, label_list = [], []
        for label, data in self._label_data_dict.items():
            data_list.extend(data)
            label_list += [label] * len(data)
        return data_list, label_list
