from . import CVDataset
from paddlefsl.task_sampler import TaskSet
import paddlefsl.utils as utils
from PIL import Image
import numpy as np
import random
import pickle as pkl


__all__ = ['CubFS']


class CubFS(CVDataset):
    """
    Implementation of `CUB-FS` dataset.
    The CubFS(Caltech-UCSD Birds-200-2011 few shot) dataset was introduced by Catherine Wah et al., 2011[1].
    This dataset contains 200 classes of birds, totally 11,788 samples. We pre-process the dataset including clipping
    the image into 84*84 RGB image and splitting the training(100 classes), validation(50 classes) and
    testing(50 classes) set.
    Users will have to manually confirm and download .zip file and place it properly.

    Download URL: https://drive.google.com/file/d/1EiKOk6LAqlYwDJzUQRDUjGMsvUGBT1U8/view

    Refs:
        1. Catherine Wah, Steve Branson, Peter Welinder et al. 2011. “The caltech-ucsd birds-200-2011 dataset”

    Args:
        mode(str): mode of the datasets, whether used as meta-training, meta-validation or meta-testing.
        root(str, optional): root directory of the .zip file, can be set None. If None, it will be set
            default root: '<path to>/paddlefsl/../raw_data/'. This code will check whether root contains .zip file.
            If not, error occurs to inform the user to download file manually.
        transform(callable, optional): transform to perform on image, None for no transform.
        backend(str, optional): Specifies which type of image to be returned: PIL.Image or numpy.ndarray.
            Should be one of {'pil', 'numpy'}. Default 'numpy'.

    Returns:
        Dataset: instance of CUB-FS datasets

    Examples:
        ..code-block:: python

            from paddlefsl.vision.datasets import CubFS
            training_set = CifarFS(mode='train')
            validating_set = CifarFS(mode='valid', backend='pil')
            image, label = training_set[16]
            print(image.shape)  # (3, 84, 84)
            image, label = validating_set[15]
            image.show()  # shows the image of an otter
            print(label)  # 'otter'
            print(len(training_set))  # 5884
            task = training_set.sample_task_set(ways=5, shots=5)
            print(task.support_images.shape)  # (25, 3, 84, 84)

    """
    zip_url = 'https://drive.google.com/file/d/1EiKOk6LAqlYwDJzUQRDUjGMsvUGBT1U8/view'
    zip_name = 'cubfs.zip'
    zip_md5 = '3d802a70dbe65576b1b366a7cdf84285'
    file_name = {'train': 'cub_train.pkl',
                 'valid': 'cub_val.pkl',
                 'test': 'cub_test.pkl'}
    file_md5 = {'train': '1708d2d8f84ebbcc1c3ad1736506f8b7',
                'valid': 'a434135d900781e7277ef6020ea9206f',
                'test': '7e29b4c9a8116f925310c50c90c2174a'}

    def __init__(self,
                 mode,
                 root=None,
                 transform=None,
                 backend='numpy'):
        super(CubFS, self).__init__('cubfs', mode, root, None, transform, backend)
        self._check_files()
        # self._label_image_names: List[ (label_name, List[image names]) ]
        self.data, self.labels, self.label_idx = self._load_data()

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.backend == 'pil':
            image = np.transpose(image, [1, 2, 0])
            image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

    def sample_task_set(self, ways=5, shots=5, query_num=None):
        query_num = shots if query_num is None else query_num
        sample_labels = random.sample(list(self.label_idx.keys()), ways)
        # result: List[ (int(label), List[np.ndarray(image)]) ]
        result = []
        for label in sample_labels:
            image_list = [self.data[idx] for idx in random.sample(self.label_idx[label], shots + query_num)]
            result.append((label, image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)

    def _check_files(self):
        if utils.check_file_exist(self.root, self.file_name[self.mode], self.file_md5[self.mode]):
            print("Using downloaded and verified CubFS files.")
        elif utils.check_file_exist(self.root + '/..', self.zip_name, self.zip_md5):
            print("Using downloaded and verified cubfs.zip file. Decompressing...")
            utils.decompress(self.root + '/../' + self.zip_name, self.root)
            print("Decompress finished.")
        else:
            error_info = "Data files not found. Please download our CubFS file manually from\n" \
                         + self.zip_url + "\nand place the .zip file under 'root' which you can set, " \
                         + "or under default root: " + utils.DATA_HOME
            raise RuntimeError(error_info)

    def _load_data(self):
        datafile = self.root + '/' + self.file_name[self.mode]
        with open(datafile, "rb") as f:
            data_dict = pkl.load(f)
            labels, data = data_dict['label'], data_dict['data']
            data = np.transpose(data, [0, 3, 1, 2])
            label_idx = {label: [] for label in set(labels)}
            for i, label in enumerate(labels):
                label_idx[label].append(i)
            return data, labels, label_idx
