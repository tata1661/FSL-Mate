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

__all__ = ['CVDataset']


class CVDataset(Dataset):
    """
    Parent class of implementation of several few-shot image classification datasets.

    Args:
        module_name(str): name of the implemented datasets, for example 'omniglot' or 'mini-imagenet'.
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        image_size(tuple, optional): size of the single image, for example (28, 28), default None.
        root(str, optional): root directory of the subdirectory '/${module_name}' which contains data files,
            can be set None. If None, it will be set default root: '<path to>/paddlefsl/../raw_data'.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'numpy'}. Default 'numpy'.

    Examples:
        ..code-block:: python

            from paddlefsl.datasets import CVDataset

            class MyDataset(CVDataset):
                def __init__(self,
                    mode,
                    image_size=None,
                    root=None,
                    transform=None,
                    backend='numpy'):
                    super(Omniglot, self).__init__('my_dataset', mode, root, image_size, transform, backend)
                    # Do your own initialize.

                def __getitem__(self, idx):
                    # Implement your own ged_item

                def __len__(self):
                    # Implement your own length

                def sample_task_set(self, ways=5, shots=5):
                    # Implement your own sample_task_set

    """

    def __init__(self,
                 module_name,
                 mode,
                 root=None,
                 image_size=None,
                 transform=None,
                 backend='numpy'):
        super(CVDataset, self).__init__()

        # self.root: ${root}/${module_name}
        self.root = utils.process_root(root, module_name)

        # self.mode: str, one of ['train', 'valid', 'test']
        if mode not in ['train', 'valid', 'test']:
            raise ValueError("'backend' should be one of ['train', 'valid', 'test'], got '"
                             + mode + "' instead.\n")
        self.mode = mode

        self.image_size = image_size

        # self.backend: one of ['pil', 'numpy']
        if backend != 'pil' and backend != 'numpy':
            raise ValueError("'backend' should be 'pil' or 'numpy', got '"
                             + backend + "' instead.\n")
        self.backend = backend

        self.transform = transform

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
