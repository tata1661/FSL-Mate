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
import pickle as pkl


__all__ = ['FC100']


class FC100(CVDataset):
    """
    Implementation of `FC100` dataset.
    The CifarFS(Few-shot CIFAR100) dataset was introduced by Oreshkin B et al., 2018[1].
    Just like CifarFS, FC100 also origin from Cifar100 dataset, and it also consists of 100 image classes, each
    containing 600 samples of size 32*32, RGB format. It was split into 60 classes for training, 20 classes for
    validating and 20 classes for testing.
    Users will have to manually confirm and download .zip file and place it properly.

    Download URL: https://drive.google.com/file/d/18SPp-RLOL-nxxoHtkU8-n8OspDjMfhAH/view

    Refs:
        1. Oreshkin B, López P R, Lacoste A. et al. 2018. “Tadam: Task dependent adaptive metric for improved
        few-shot learning” NIPS.

    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        root(str, optional): root directory of the .zip file, can be set None. If None, it will be set
            default root: '<path to>/paddlefsl/../raw_data/'. This code will check whether root contains .zip file.
            If not, error occurs to inform the user to download file manually.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned: PIL.Image or numpy.ndarray.
            Should be one of {'pil', 'numpy'}. Default 'numpy'.

    Returns:
        Dataset: instance of FC100 datasets

    Examples:
        ..code-block:: python

            from paddlefsl.vision.datasets import FC100

            train_dataset = FC100(mode='train')
            image, label = train_dataset[3001]
            print(image, label)  # numpy.ndarray of size (3, 32, 32), 70
            print(len(train_dataset))  # 36000
            task = train_dataset.sample_task_set()
            print(task.support_images)  # numpy.ndarray of size (25, 3, 32, 32)
            print(task.query_images.shape)  # (25, 3, 32, 32)

    """
    zip_url = 'https://drive.google.com/file/d/18SPp-RLOL-nxxoHtkU8-n8OspDjMfhAH/view'
    zip_name = 'FC100.zip'
    zip_md5 = 'db78f3657ff1a06cfe3dd74421a98d5d'
    file_name = {'train': 'FC100_train.pickle',
                 'valid': 'FC100_val.pickle',
                 'test': 'FC100_test.pickle'}
    file_md5 = {'train': '1a740ab9aa52ed9397704b38ea19cc6e',
                'valid': '1968510800c5d75a96ca948572cc4dfc',
                'test': 'c2dc7b564f5576cf3f04bf0f39ecc9ff'}

    def __init__(self,
                 mode,
                 root=None,
                 transform=None,
                 backend='numpy'):
        super(FC100, self).__init__('FC100', mode, root, None, transform, backend)
        self._check_files()
        # self._label_image_names: List[ (label_name, List[image names]) ]
        self.data, self.labels, self.label_idx = self._load_data()

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.backend == 'pil':
            image = np.transpose(image, [1, 2, 0])
            image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.labels)

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        sample_labels = random.sample(list(self.label_idx.keys()), ways)
        # result: List[ (int(label), List[np.ndarray(image)]) ]
        result = []
        query_num = shots if query_num is None else query_num
        for label in sample_labels:
            image_list = [self.data[idx] for idx in random.sample(self.label_idx[label], shots + query_num)]
            result.append((label, image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def _check_files(self):
        if utils.check_file_exist(self.root, self.file_name[self.mode], self.file_md5[self.mode]):
            print("Using downloaded and verified FC100 files.")
        elif utils.check_file_exist(self.root + '/..', self.zip_name, self.zip_md5):
            print("Using downloaded and verified FC100.zip file. Decompressing...")
            utils.decompress(self.root + '/../' + self.zip_name, self.root)
            print("Decompress finished.")
        else:
            error_info = "Data files not found. Please download our FC100 file manually from\n" \
                         + self.zip_url + "\nand place the .zip file under 'root' which you can set, " \
                         + "or under default root: " + utils.DATA_HOME
            raise RuntimeError(error_info)

    def _load_data(self):
        datafile = self.root + '/' + self.file_name[self.mode]
        with open(datafile, "rb") as f:
            data_dict = pkl.load(f, encoding='latin1')
            labels, data = data_dict['labels'], data_dict['data']
            data = np.transpose(data, [0, 3, 1, 2])
            label_idx = {}
            for i, label in enumerate(labels):
                if label in label_idx:
                    label_idx[label].append(i)
                else:
                    label_idx[label] = [i]
            return data, labels, label_idx
