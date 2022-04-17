from ast import Raise
from collections import defaultdict
import os
from typing import Union
from PIL.Image import Image
import numpy as np
import pickle as pkl
import random
from paddle.vision import image_load
from paddle.vision.transforms import resize

import pandas as pd
from . import CVDataset
from paddlefsl.task_sampler import TaskSet
import paddlefsl.utils as utils

__all__ = ['MiniImageNet']


def _read_image_data(file: str, backend: str = 'pil'):
    """read the image data to numpy array

    Args:
        file (str): the path of the image file.
        backend (str, optional): the backend of image reader, which can be cv2/pil/None. Defaults to 'pil'.

    Raises:
        ValueError: the invalid data type.

    Returns:
        numpy.ndarray: the image data.
    """
    data: Union[Image, np.ndarray] = image_load(file, backend)
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Image):
        np_data = np.array(data.getdata())
        np_data = np_data.reshape(data.width, data.height, 3)
        return np_data
    else:
        raise ValueError('not supported backend: {}'.format(backend))


class MiniImageNet(CVDataset):
    """
    Implementation of `mini-ImageNet`_ datasets
    The mini-ImageNet datasets was proposed by Vinyals et al., 2016[1] for few-shot learning. It is a
    subset split from ImageNet datasets, 2015[2]. Mini-ImageNet contains 100 classes with 600 samples
    of 84×84 color images per class. These 100 classes are divided into 64, 16, and 20 classes
    respectively for sampling tasks for training, validation and test.

    The split we use is created by Ravi and Larochelle, 2017. Originally, to generate mini-ImageNet,
    users should first download the full ImageNet and split it. To let our users avoid downloading
    unnecessary data, we pre-generated mini-ImageNet into .pkl files and compressed them. This class
    will use our own data files on Google Drive. But since Google Drive cannot scan virus in large
    files, users will have to manually confirm and download .tar.gz file, and place it properly.

    Download URL: https://drive.google.com/file/d/1LLUjwSUpWGSWizl3JZxd08V30_dIaRBx/view

    Refs:
        1. Vinyals et al. 2016. “Matching Networks for One Shot Learning.”
        2. Russakovsky, et al. 2015. "Imagenet large scale visual recognition challenge."
        3. Ravi, Sachin, and Hugo Larochelle. 2016. "Optimization as a model for few-shot learning."

    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        root(str, optional): root directory of the .zip file, can be set None. If None, it will be set
            default root: '<path to>/paddlefsl/../raw_data/'. This code will check whether root contains .zip file.
            If not, error occurs to inform the user to download file manually.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'numpy'}. Default 'numpy'.

    Returns:
        Dataset: instance of Mini-ImageNet dataset.

    Examples:
        ..code-block:: python

            # Before using this dataset, download file and place it under 'root'.
            from paddlefsl.vision.datasets import MiniImageNet
            training_set = MiniImageNet(mode='train')
            validation_set = MiniImageNet(mode='valid', backend='pil')
            image, label = validation_set[0]
            image.show() # System shows an image of a bird.
            image, label = training_set[0]
            print(image, label) # Numpy array and its label
            print(len(training_set)) # 38400
            task = training_set.sample_task_set(ways=5, shots=5)
            print(task.support_images.shape) # (25, 3, 84, 84)
            print(task.query_images.shape) # (25, 3, 84, 84)

    """

    tar_url = 'https://drive.google.com/file/d/1LLUjwSUpWGSWizl3JZxd08V30_dIaRBx/view'
    tar_name = 'mini-imagenet.tar.gz'
    tar_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    file_name = {'train': 'mini-imagenet-cache-train.pkl',
                 'valid': 'mini-imagenet-cache-val.pkl',
                 'test': 'mini-imagenet-cache-test.pkl'}
    file_md5 = {'train': '069993a77e65a83bbf8f8de4c9dcca68',
                'valid': 'f87ddf9b5211920dd73520274371eb6a',
                'test': '12158d649fd08945294de4ad38cda81d'}
    local_file_name = {
        'train': 'train.csv',
        'valid': 'val.csv',
        'test': 'test.csv'
    }

    def __init__(self,
                 mode,
                 root=None,
                 transform=None,
                 backend='numpy'
                 ):
        
        # Set self.mode, self.root, self.transform and self.backend in parent class
        super(MiniImageNet, self).__init__('mini-imagenet', mode, root, None, transform, backend)

        # 1. check data type
        if self._check_flat_data(root):
            self._gen_processed_file(root)

            # clear the md5 for file checking
            self.file_md5[mode] = None

        # 2. Check exist and download the data file
        self._check_files()

        # 3. Load data from pkl file
        # image_data: np.ndarray of shape (image_num, 3, 84, 84)
        # class_dict: Dict[str(class label): List[index of image in image_data]]
        self._image_data, self._class_dict = self._load_data()

    def _check_flat_data(self, image_root_dir: str):
        """check the data of mini-imagenet is the flat data structure:
            ./train.csv
            ./val.csv
            ./test.csv
            ./images/
                01.jpg
                02.jpg
                ...

            this structure of mini-imagenet data can be from `miniimagenet` data of aistudio.
        """
        if not os.path.exists(image_root_dir):
            return False

        # 1. check if there are train.cvs, val.csv, test.csv file which contains the metadata
        if not os.path.exists(os.path.join(image_root_dir, self.local_file_name[self.mode])):
            return False

        # 2. check if there are images/ folder which contains the images
        images_dir = os.path.join(image_root_dir, 'images')
        return os.path.exists(images_dir)

    def _gen_processed_file(self, image_root_dir: str):
        """the image data dir which contains the train/val/test.csv and images/ folder

        Args:
            image_root_dir (str): the directory which contains the train/val/test.csv 
                and images/ folder, eg: /home/aistudio/data/data105646/mini-imagenet-sxc/
        """
        # 1. check the generated pickle file is exist

        # self.root = root + 'mini-imagenet'
        pickle_file = os.path.join(self.root, self.file_name[self.mode])
        if os.path.exists(pickle_file):
            return
        
        # 2. generate the pickle file based on the data in the train/val/test.csv
        image_data, class_dict = [], defaultdict(list)

        csv_file = os.path.join(image_root_dir, self.local_file_name[self.mode])
        for row_index, row in pd.read_csv(csv_file).iterrows():
            file_path = os.path.join(image_root_dir, 'images', row['filename'])
            image: np.ndarray = _read_image_data(file=file_path, backend=self.backend)

            if image.shape != [84, 84, 3]:
                image = resize(image, size=[84, 84])

            image_data.append(image)
            class_dict[row['label']].append(row_index)

        image_data = np.array(image_data)

        # 3. save the pickle file
        with open(pickle_file, 'wb') as f:
            pkl.dump(dict(image_data=image_data, class_dict=class_dict), f)

    def __getitem__(self, idx):
        image = self._image_data[idx]
        if self.backend == 'pil':
            image = np.transpose(image, [1, 2, 0])
            image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        for label in self._class_dict.keys():
            if idx in self._class_dict[label]:
                return image, label
        raise RuntimeError("Dataset error: cannot find label of the image.")

    def __len__(self):
        return len(self._image_data)

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        query_num = shots if query_num is None else query_num
        sample = random.sample(list(self._class_dict.keys()), ways)
        # result: List[ (str(label name), List[np.ndarray(image)]) ]
        result = []
        for label in sample:
            image_list = [self._image_data[i] for i in random.sample(self._class_dict[label], shots + query_num)]
            if self.transform is not None:
                image_list = [self.transform(image) for image in image_list]
            result.append((label, image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def _check_files(self):
        if utils.check_file_exist(self.root, self.file_name[self.mode], self.file_md5[self.mode]):
            print("Using downloaded and verified mini-imagenet files.")
        elif utils.check_file_exist(self.root + '/..', self.tar_name, self.tar_md5):
            print("Using downloaded and verified .tar.gz file. Decompressing...")
            utils.decompress(self.root + '/../' + self.tar_name, self.root)
            print("Decompress finished.")
        else:
            error_info = "Data files not found. Please download our mini-ImageNet file manually from\n" \
                         + self.tar_url + "\nand place the .tar.gz file under 'root' which you can set, " \
                         + "or under default root: " + utils.DATA_HOME
            raise RuntimeError(error_info)

    def _load_data(self):
        datafile = self.root + '/' + self.file_name[self.mode]
        with open(datafile, "rb") as f:
            # data: Dict['image_data': np.ndarray, 'class_dict': Dict]
            # image_data: np.ndarray of shape (image_num, 84, 84, 3)
            # class_dict: Dict[str(class label): List[index of image in image_data]]
            data = pkl.load(f)
            image_data = data['image_data']
            image_data = np.transpose(image_data, [0, 3, 1, 2])
            class_dict = data['class_dict']
        return image_data, class_dict
