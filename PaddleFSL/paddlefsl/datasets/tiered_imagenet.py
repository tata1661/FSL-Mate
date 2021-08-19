from paddlefsl.task_sampler import TaskSet
import paddlefsl.utils as utils
import pickle as pkl
import numpy as np
import random
from . import CVDataset
from PIL import Image

__all__ = ['TieredImageNet']


class TieredImageNet(CVDataset):
    """
    Implementation of `tiered-ImageNet`_ datasets
    The tiered-ImageNet datasets was proposed by M. Ren et al., 2018[1] for few-shot learning. It is a
    subset split from ImageNet datasets, 2015[2]. Tiered-ImageNet contains 34 categories, each category contains
    10-30 classes. Samples are 84×84 color images. These 34 categories are divided into 20, 6, and 8 categories
    respectively for sampling tasks for training, validation and test.
    Users will have to manually confirm and download .tar file and place it properly.

    Download URL: https://drive.google.com/file/d/1fQ6lI5pCnOEt9MHWdqFN1cdSU2SbMKzx/view

    Refs:
        1. Ren M, Triantafillou E, Ravi S, et al. 2018. “Meta-learning for semi-supervised few-shot classification”
        2. Russakovsky, et al. 2015. "Imagenet large scale visual recognition challenge."

    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        root(str, optional): root directory of the .tar file, can be set None. If None, it will be set
            default root: '<path to>/paddlefsl/../raw_data/'. This code will check whether root contains .tar file.
            If not, error occurs to inform the user to download file manually.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'numpy'}. Default 'numpy'.

    Returns:
        Dataset: instance of Tiered-ImageNet dataset.

    Examples:
        ..code-block:: python

            # Before using this dataset, download file and place it under 'root'.
            from paddlefsl.vision.datasets import TieredImageNet

            validation_set = TieredImageNet(mode='valid')
            image, label = validation_set[0]
            print(image.shape, label)  # (3, 84, 84) cab, hack, taxi, taxicab
            import numpy as np
            from PIL import Image
            image = np.transpose(image, [1, 2, 0])
            image = Image.fromarray(image)
            image.show()  # Image of a blue cab

    """
    tar_url = 'https://drive.google.com/file/d/1fQ6lI5pCnOEt9MHWdqFN1cdSU2SbMKzx/view'
    tar_name = 'tiered-imagenet.tar.xz'
    tar_md5 = 'a8a22d60d2d6d2bde6697d3a24389264'
    image_file = {'train': 'train_images_png.pkl',
                  'valid': 'val_images_png.pkl',
                  'test': 'test_images_png.pkl'}
    image_md5 = {'train': '0fcfb86f4423c40bc6799b9ebe8f3510',
                 'valid': 'ca9f51a0106ca864a4a759233c4deb89',
                 'test': '608ee004b81796ece2da6e5bb949ccaa'}
    label_file = {'train': 'train_labels.pkl',
                  'valid': 'val_labels.pkl',
                  'test': 'test_labels.pkl'}
    label_md5 = {'train': '608ee004b81796ece2da6e5bb949ccaa',
                 'valid': '755f959b068510cfc1abfde713cc734c',
                 'test': '04543459f62dc3532cb20047b1282997'}

    def __init__(self,
                 mode,
                 root=None,
                 transform=None,
                 backend='numpy'):
        super(TieredImageNet, self).__init__('tiered-imagenet', mode, root, None, transform, backend)
        self._check_files()
        self._image_list, self._label_num, self._label_name = self._load_data()
        self._label_image_idx = {label: [] for label in range(len(self._label_name))}
        for i in range(len(self._image_list)):
            label = self._label_num[i]
            self._label_image_idx[label].append(i)

    def __getitem__(self, idx):
        image, label_num = self._image_list[idx], self._label_num[idx]
        if self.backend == 'pil':
            image = np.transpose(image, [1, 2, 0])
            image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self._label_name[label_num]

    def __len__(self):
        return len(self._image_list)

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        query_num = shots if query_num is None else query_num
        sample = random.sample(range(len(self._label_name)), ways)
        # result: List[ (str(label name), List[np.ndarray(image)]) ]
        result = []
        for num in sample:
            image_idx = random.sample(self._label_image_idx[num], shots + query_num)
            image_list = [self._image_list[i] for i in image_idx]
            if self.transform is not None:
                image_list = [self.transform(image) for image in image_list]
            result.append((self._label_name[num], image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def _check_files(self):
        print("Checking tiered-imagenet files. The datasets is rather large, please be patient...")
        if utils.check_file_exist(self.root, self.image_file[self.mode], self.image_md5[self.mode]) and \
                utils.check_file_exist(self.root, self.label_file[self.mode], self.label_md5[self.mode]):
            print("Using downloaded and verified tiered-imagenet files.")
        elif utils.check_file_exist(self.root + '/..', self.tar_name, self.tar_md5):
            print("Using downloaded and verified .tar.xz file. Decompressing...")
            utils.decompress(self.root + '/../' + self.tar_name, self.root)
            print("Decompress finished.")
        else:
            error_info = "Data files not found. Please download our tiered-ImageNet file manually from\n" \
                         + self.tar_url + "\nand place the .tar file under 'root' which you can set, " \
                         + "or under default root: " + utils.DATA_HOME
            raise RuntimeError(error_info)

    def _load_data(self):
        image_file = self.root + '/' + self.image_file[self.mode]
        label_file = self.root + '/' + self.label_file[self.mode]
        with open(image_file, 'rb') as f:
            image_list = pkl.load(f)
        with open(label_file, 'rb') as f:
            # full_label: Dict['label_specific': -, 'label_general': -, 'label_specific_str': -, 'label_general_str': -]
            full_label = pkl.load(f, encoding='bytes')
            label_num, label_name = full_label['label_specific'], full_label['label_specific_str']
        return image_list, label_num, label_name
