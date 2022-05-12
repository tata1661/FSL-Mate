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
from PIL import Image
import numpy as np
import random
import csv
import os

__all__ = ['MiniImageNetNoreseize']


class MiniImageNetNoreseize(CVDataset):
    """
    Download URL: 'https://pan.baidu.com/s/1qwzFa3XeB_92-uj2d0anJw          password:g4t9'
    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        root(str, optional): root directory of the .zip file, can be set None. If None, it will be set
            default root: '<path to>/paddlefsl/../raw_data/'. This code will check whether root contains .zip file.
            If not, error occurs to inform the user to download file manually.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'numpy'}. Default 'pil'.

    Returns:
        Dataset: instance of Mini-ImageNet dataset.

    Examples:
        ..code-block:: python

            # Before using this dataset, download file and place it under 'root'.
            from paddlefsl.datasets import MiniImageNetNoreseize

            training_set = paddlefsl.datasets.MiniImageNetNoreseize(mode='train', backend='pil',
                                                        transform=paddle.vision.transforms.RandomResizedCrop(84),
                                                        root='your path ')
            image, label_dig, label_str = training_set[0]
            print(image, label_dig, label_str)  # Numpy array and its label
            print(len(training_set))  # 38400
            task = training_set.sample_task_set(ways=5, shots=5)
            print(task.support_data.shape)  # (25, 3, 84, 84)
            print(task.query_data.shape)  # (25, 3, 84, 84)

    """

    tar_url = 'https://pan.baidu.com/s/1qwzFa3XeB_92-uj2d0anJw          password:g4t9'
    tar_name = 'mini_imagenet_noresize.tar.gz'
    file_name = {'train': 'train.csv',
                 'valid': 'valid.csv',
                 'test': 'test.csv'}

    def __init__(self,
                 mode,
                 root=None,
                 transform=None,
                 backend='pil'):
        # Set self.mode, self.root, self.transform and self.backend in parent class
        super(MiniImageNetNoreseize, self).__init__('mini-imagenet-noresize', mode, root, None, transform, backend)

        # Check exist and download the data file
        self._check_files()
        # Load data from  file
        # image_names: name of images,type = ndarray ,such as [n0207436700000001.jpg,n0207436700000003.jpg...]
        # image_labels_str: label of images,type = ndarray ,such as [n02074367,n02074367,n02074367...]
        self._image_names, self._image_labels_str = self._load_data()
        self.root = root
        if mode == 'train':
            self._numclass = 64
        elif mode == 'valid':
            self._numclass = 16
        elif mode == 'test':
            self._numclass = 20
        # label of images , such as [0,0,0,0,....,1,1,1....,32,32,32..]
        self._image_labels_dig = np.arange(self._numclass).repeat(600)
        self._idxmatrix = np.arange(self._numclass * 600).reshape(self._numclass, 600)

    def __getitem__(self, idx):
        # load a image from 'self.root/images/ '
        image = Image.open(os.path.join(self.root, 'mini-imagenet-noresize','images', self._image_names[idx]))
        if self.backend == 'numpy':
            image = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
        if self.transform is not None:
            image = self.transform(image)
        return image, self._image_labels_dig[idx], self._image_labels_str[idx]

    def __len__(self):
        return self._image_names.shape[0]

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        query_num = shots if query_num is None else query_num
        sample = random.sample(list(self._idxmatrix), ways)
        # result: List[ (str(label name), List[image1,image2...]) ]
        result = []
        for label in sample:
            image_list = [Image.open(os.path.join(self.root,'mini-imagenet-noresize','images', self._image_names[i])) for i in
                          random.sample(list(self._idxmatrix[int(label[0] / 600)]), shots + query_num)]
            if self.transform is not None:
                image_list = [self.transform(image) for image in image_list]
            result.append((self._image_labels_str[label[0]], image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def _check_files(self):
        if utils.check_file_exist(self.root, self.file_name[self.mode]):
            print("Using downloaded and verified mini-imagenet_noresize files.")
        elif utils.check_file_exist(self.root + '/..', self.tar_name):
            print("Using downloaded and verified .tar.gz file. Decompressing...")
            utils.decompress(self.root + '/../' + self.tar_name, self.root)
            print("Decompress finished.")
        else:
            error_info = "Data files not found. Please download our mini-ImageNet file manually from\n" \
                         + self.tar_url + "\nand place the .tar.gz file under 'root' which you can set, " \
                         + "or under default root: " + utils.DATA_HOME
            raise RuntimeError(error_info)

    def _load_data(self):
        datafile = self.root + '/' + self.mode + '.csv'
        with open(datafile) as f:
            # column 1 : list[imagenames] such as [n0207436700000001.jpg,n0207436700000003.jpg...]
            # column 2 : list[imagelabels] such as [n02074367,n02074367,n02074367...]
            names_labels = [row for row in csv.reader(f)]
        names_labels = np.array(names_labels)
        image_names = names_labels[:, 0]
        image_labels_str = names_labels[:, 1]
        return image_names, image_labels_str

