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

import paddlefsl.utils as utils
from paddle.io import Dataset

__all__ = ['NLPDataset']


class NLPDataset(Dataset):
    """
    Parent class of implementation of several few-shot relation classification datasets.

    Args:
        module_name(str): name of the implemented datasets, for example 'few_rel'.
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
        (Please see paddlefsl.datasets.few_rel)

    """

    def __init__(self,
                 module_name,
                 mode,
                 root=None,
                 max_len=None,
                 backend='numpy',
                 vector_initializer=None):
        super(NLPDataset, self).__init__()

        # self.root: ${root}/${module_name}
        self.root = utils.process_root(root, module_name)

        # self.mode: str, for example 'train' or 'valid'
        self.mode = mode

        # self.max_len: max length of each sentence
        self.max_len = max_len

        # self.backend: str, one of ['numpy', 'text']. 'numpy' for initialized vector, 'text' for tokens, head and tail
        if backend not in ['numpy', 'text']:
            raise ValueError("'backend' should be one of ['numpy', 'text'], got " + str(backend))
        self.backend = backend

        # self.vector_initializer: callable
        if self.backend == 'numpy' and not hasattr(vector_initializer, '__call__'):
            raise ValueError("When 'backend' is 'numpy', 'vector_initializer' is required "
                             "to be paddlefsl.backbones.RCInitVector to generate word vector numpy array")
        self.vector_initializer = vector_initializer

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class {}".
                                  format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class {}".
                                  format('__len__', self.__class__.__name__))

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        """
        Randomly Sample a task set of N ways and K shots according to meta-learning task.
        :param ways: int, number of classes of the task, default 5.
        :param shots: int, number of samples of each class when training, default 5.
        :param query_num: int, number of query points of each class, default None.
        :return: paddlefsl.vision.task_sampler.TaskSet.
        """
        raise NotImplementedError("'{}' not implement in class {}".
                                  format('sample_task_set', self.__class__.__name__))
