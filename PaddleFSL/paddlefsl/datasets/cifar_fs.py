from . import CVDataset
from paddlefsl.task_sampler import TaskSet
import paddlefsl.utils as utils
from PIL import Image
import numpy as np
import random


__all__ = ['CifarFS']


class CifarFS(CVDataset):
    """
    Implementation of `CifarFS` dataset.
    The CifarFS(Cifar100 Few Shot) dataset was introduced by Bertinetto L et al., 2019[1].
    CifarFS consists of 100 image classes, each containing 600 samples of size 32*32, RGB format. It was split into
    64 classes for training, 16 classes for validating and 20 classes for testing.
    Users will have to manually confirm and download .zip file and place it properly.

    Download URL: https://drive.google.com/file/d/1nN1u2ZeD0L90uG5Y_Ml6uQR6z-o6aBLL/view

    Refs:
        1. Bertinetto L et al. 2019. “Meta-learning with differentiable closed-form solvers” ICLR.

    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        image_size(tuple, optional): size of the single image, for example (28, 28), default None. If None, this
            code will return the original image size.
        root(str, optional): root directory of the .zip file, can be set None. If None, it will be set
            default root: '<path to>/paddlefsl/../raw_data/'. This code will check whether root contains .zip file.
            If not, error occurs to inform the user to download file manually.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned: PIL.Image or numpy.ndarray.
            Should be one of {'pil', 'numpy'}. Default 'numpy'.

    Returns:
        Dataset: instance of CifarFS datasets

    Examples:
        ..code-block:: python

            from paddlefsl.vision.datasets import CifarFS
            training_set = CifarFS(mode='train')
            validating_set = CifarFS(mode='valid', backend='pil')
            image, label = training_set[16]
            print(image.shape)  # (3, 32, 32)
            print(label)  # 'train'
            image, label = validating_set[15]
            image.show()  # shows the image of an otter
            print(label)  # 'otter'
            print(len(training_set))  # 64 * 600 = 38400
            task = training_set.sample_task_set(ways=5, shots=5)
            print(task.support_images.shape)  # (25, 3, 32, 32)

    """
    zip_url = 'https://drive.google.com/file/d/1nN1u2ZeD0L90uG5Y_Ml6uQR6z-o6aBLL/view'
    zip_name = 'cifar100.zip'
    zip_md5 = 'fdb4405027b809aa4c059047f21ca3d1'

    def __init__(self,
                 mode,
                 root=None,
                 image_size=None,
                 transform=None,
                 backend='numpy'):
        super(CifarFS, self).__init__('cifar100', mode, root, image_size, transform, backend)
        self._check_files()
        # self._label_image_names: List[ (label_name, List[image names]) ]
        self._labels_image_names = self._load_data()

    def __getitem__(self, idx):
        label_idx, image_idx = int(idx / 600), idx % 600
        label_name = self._labels_image_names[label_idx][0]
        image_name = self._labels_image_names[label_idx][1][image_idx]
        image = self._get_image_by_name(label_name, image_name)
        return image, label_name

    def __len__(self):
        return len(self._labels_image_names) * 600

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        query_num = shots if query_num is None else query_num
        sample = random.sample(self._labels_image_names, ways)
        # result: List[ (str(label name), List[np.ndarray(image)]) ]
        result = []
        for (label_name, image_names) in sample:
            image_list = [self._get_image_by_name(label_name, image_name, backend='numpy')
                          for image_name in random.sample(image_names, shots + query_num)]
            result.append((label_name, image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def _check_files(self):
        if utils.check_file_exist(self.root + '/..', self.zip_name, self.zip_md5):
            print("Using downloaded and verified .zip file. Decompressing...")
            utils.decompress(self.root + '/../' + self.zip_name, self.root + '/..')
            print("Decompress finished.")
        else:
            error_info = "Data files not found. Please download CIFAR-FS file manually from\n" \
                         + self.zip_url + "\nand place the .zip file under 'root' which you can set, " \
                         + "or under default root: " + utils.DATA_HOME
            raise RuntimeError(error_info)

    def _load_data(self):
        mode = 'val' if self.mode == 'valid' else self.mode
        with open(self.root + '/splits/bertinetto/' + mode + '.txt') as f:
            label_names = f.read().split('\n')[:-1]
        label_image_names = [(label, [image for image in utils.list_files(self.root + '/data/' + label, '.png')])
                             for label in label_names]
        return label_image_names

    def _get_image_by_name(self, label_name, image_name, backend=None):
        backend = self.backend if backend is None else backend
        image = Image.open(self.root + '/data/' + label_name + '/' + image_name, mode='r').convert('RGB')
        if self.image_size is not None:
            image = image.resize(self.image_size, Image.ANTIALIAS)
        if backend == 'numpy':
            image = np.array(image)
            image = np.transpose(image, [2, 0, 1])
        if self.transform is not None:
            image = self.transform(image)
        return image
