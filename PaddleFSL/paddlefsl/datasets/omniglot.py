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

from . import CVDataset
from paddlefsl.task_sampler import TaskSet
import paddlefsl.utils as utils
from paddle.io import Dataset
from PIL import Image
import numpy as np
import random


__all__ = ['Omniglot']


class Omniglot(CVDataset):
    """
    Implementation of `Omniglot <https://github.com/brendenlake/omniglot>`_ datasets
    The Omniglot datasets was introduced by Lake et al., 2015[1].
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    The original datasets has 'background' and 'evaluation' parts, which is defined by the authors.
    This class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017[2]. We also provide a default random splitting.

    Refs:
        1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
        2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        image_size(tuple, optional): size of the single image, for example (28, 28), default None. If None, this
            code will return the original image size.
        root(str, optional): root directory of the subdirectory '/omniglot/' which contains .zip data files,
            can be set None. If None, it will be set default root: '<path to>/paddlefsl/../raw_data/'.
            This code will check whether subdirectory '/omniglot/' exist under root. If it exists and it
            contains the correct .zip files, the code will use the existing file. Otherwise it will
            automatically create '/omniglot/' subdirectory and download .zip files.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned: PIL.Image or numpy.ndarray.
            Should be one of {'pil', 'numpy'}. Default 'numpy'. If 'numpy', this code will transform the matrix by
            / 255.0 and reverse black and white.

    Returns:
        Dataset: instance of Omniglot datasets

    Examples:
        ..code-block:: python

            from paddlefsl.vision.datasets import Omniglot
            training_set = Omniglot(mode='train', image_size=(28, 28))
            image, label = training_set[15]
            print(image) # An 28 * 28 numpy array
            print(label) # Name of the alphabet class, for example '/images_background/Japanese_(katakana)/character44'
            print(len(training_set)) # 22000
            print(training_set.class_num(mode='self')) # 1100
            print(training_set.class_num(mode='total')) # 1623
            training_set.split_task(shuffle=True, train_class_num=1200)
            print(training_set.class_num()) # 1200
            task = training_set.sample_task_set(ways=5, shots=5)
            print(task.support_images.shape) # (25, 1, 28, 28)
            print(task.query_images.shape) # (25, 1, 28, 28)

    """
    background_url = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
    evaluation_url = 'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    background_md5 = '68d2efa1b9178cc56df9314c21c6e718'
    evaluation_md5 = '6b91aef0f799c5bb55b94e3f2daec811'
    background_folder = '/images_background/'
    evaluation_folder = '/images_evaluation/'

    def __init__(self,
                 mode,
                 image_size=None,
                 root=None,
                 transform=None,
                 backend='numpy'):
        # Set self.mode, self.root, self.transform and self.backend in parent class
        super(Omniglot, self).__init__('omniglot', mode, root, image_size, transform, backend)

        # Check exist and download the data file
        self._check_and_download_file()

        # Load the full datasets as a list of label-image names tuple
        # self._full_labels_image_names: List[ (str(label name), List[image names]) ]
        self._full_labels_image_names = self._load_image_label()

        # Split the datasets into 'train', 'validate' and 'test' and get the set of self.mode
        # self._labels_image_names: ListList[ (str(label name for self.mode), List[image names for self.mode]) ]
        self._labels_image_names = []
        self.split_task()

    def __getitem__(self, idx):
        label_idx, image_idx = int(idx / 20), idx % 20
        label_name = self._labels_image_names[label_idx][0]
        image_name = self._labels_image_names[label_idx][1][image_idx]
        image = self._get_image_by_name(label_name, image_name)
        return image, label_name

    def __len__(self):
        return len(self._labels_image_names) * 20

    def class_num(self, mode='self'):
        """
        Class number. Total class number includes both background and evaluation set. Self class number depend on the
        way to split training, validation and test set.
        :param mode: str, return total class number or self class number, should be one of ['self', 'total']
        :return: int, total class number of this datasets
        """
        if mode == 'self':
            return len(self._labels_image_names)
        elif mode == 'total':
            return len(self._full_labels_image_names)
        else:
            raise ValueError('\'mode\' of class_num should be \'self\' or \'total\'.')

    def split_task(self,
                   shuffle=False,
                   train_class_num=1100,
                   valid_class_num=100):
        """
        To split the datasets into training set, validating set and testing set.
        Each set contains different classes, according to meta-learning tasks.
        :param shuffle: bool, whether to shuffle all the datasets classes, default False.
        :param train_class_num: int, number of classes used as training task, default 1100.
        :param valid_class_num: int, number of classes used as validating task, default 100.
                The rest of the classes will be used as testing task, default 423
        :return: List[ (str(label name for self.mode), List[image names for self.mode]) ]
        """
        n_total = self.class_num(mode='total')
        if train_class_num <= 0 or valid_class_num <= 0 or train_class_num + valid_class_num >= n_total:
            print("Invalid splitting. Use default splitting instead.")
            train_class_num, valid_class_num = 1100, 100
        if shuffle:
            random.shuffle(self._full_labels_image_names)
        _train_labels_images = self._full_labels_image_names[:train_class_num]
        _valid_labels_images = self._full_labels_image_names[train_class_num:train_class_num + valid_class_num]
        _test_labels_images = self._full_labels_image_names[train_class_num + valid_class_num:]
        if self.mode == 'train':
            self._labels_image_names = _train_labels_images
        elif self.mode == 'valid':
            self._labels_image_names = _valid_labels_images
        else:
            self._labels_image_names = _test_labels_images

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        query_num = shots if query_num is None else query_num
        sample = random.sample(self._labels_image_names, ways)
        # result: List[ (str(label name), List[np.ndarray(image)]) ]
        result = []
        for (label_name, image_names) in sample:
            rotate_degree = 90 * random.randint(0, 3)
            image_list = [self._get_image_by_name(label_name, image_name, rotate_degree=rotate_degree, backend='numpy')
                          for image_name in random.sample(image_names, shots + query_num)]
            result.append((label_name, image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def _check_and_download_file(self):
        background_file = utils.download_url(self.root, self.background_url, self.background_md5)
        utils.decompress(background_file)
        evaluation_file = utils.download_url(self.root, self.evaluation_url, md5=self.evaluation_md5)
        utils.decompress(evaluation_file)

    def _load_image_label(self):
        background_alphabets = [self.background_folder + d
                                for d in utils.list_dir(self.root + self.background_folder)]
        evaluation_alphabets = [self.evaluation_folder + d
                                for d in utils.list_dir(self.root + self.evaluation_folder)]
        alphabets = background_alphabets + evaluation_alphabets
        labels = sum([[a + '/' + c for c in utils.list_dir(self.root + '/' + a)]
                      for a in alphabets], [])
        labels_images = [(label, [image for image in utils.list_files(self.root + '/' + label, '.png')])
                         for label in labels]
        return labels_images

    def _get_image_by_name(self, label_name, image_name, rotate_degree=0, backend=None):
        backend = self.backend if backend is None else backend
        image = Image.open(self.root + label_name + '/' + image_name, mode='r').convert('L')
        if self.image_size is not None:
            image = image.resize(self.image_size, Image.ANTIALIAS)
        image = image.rotate(rotate_degree)
        if backend == 'numpy':
            image = np.array(image) / 255.0
            image = np.ones_like(image) - image
            image = np.expand_dims(image, axis=0)
        if self.transform is not None:
            image = self.transform(image)
        return image
