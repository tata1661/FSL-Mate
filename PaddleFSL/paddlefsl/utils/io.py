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

import os
import hashlib
import requests
import tqdm
import zipfile
import tarfile


__all__ = ['PACKAGE_HOME', 'DATA_HOME',
           'process_root', 'download_url', 'check_file_exist',
           'check_md5', 'decompress', 'list_dir', 'list_files',
           'clear_file', 'get_info_str', 'print_training_info']


_current_path = os.path.dirname(__file__)
PACKAGE_HOME = os.path.abspath(os.path.join(_current_path, '..', '..'))
DATA_HOME = os.path.join(PACKAGE_HOME, 'raw_data')


def process_root(root, module_name=''):
    """
    Make a directory or check the existing directory of a module under 'root' directory.

    Args:
        root(str): root directory, can be set None. If None, it will be set DATA_HOME
        module_name(str, optional): module directory name, default ''. By default, this function does not make new
            directory under root, and returns the root path.

    Returns:
        dir_name(str): new directory path.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            root = '~/test'
            module_dir = utils.process_root(root, module_name='test_module')
            # module_dir: '/home/<usr_name>/test/test_module', and the directory is created.

    """
    if root is None:
        root = DATA_HOME
    else:
        root = os.path.expanduser(root)
    dir_name = os.path.join(root, module_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def download_url(dir_name, url, md5=None):
    """
    Download file from url to directory and check the md5 code.

    Args:
        dir_name(str): directory to place the downloaded file.
        url(str): the file url.
        md5(str, optional): md5 code of the file, default None. If None, the function does not check the md5 code.

    Returns:
        None.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            background_url = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
            background_md5 = '68d2efa1b9178cc56df9314c21c6e718'
            dir_name = '~/test'
            utils.download_url(dir_name, background_url, background_md5)
            # images_background.zip will be downloaded into '~/test' directory.

    """
    # Check whether the file has already been downloaded.
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    file_name = url.split('?')[0].split('/')[-1]
    if check_file_exist(dir_name, file_name, md5):
        print('Using downloaded and verified file: ' + file_name)
        return os.path.join(dir_name, file_name)
    # Begin to download
    retry, retry_limit = 0, 3
    file_path = os.path.join(dir_name, file_name)
    while retry < retry_limit:
        print('Beginning to download from', url, 'to', dir_name)
        try:
            r = requests.get(url=url, stream=True, timeout=5)
            total_size = int(r.headers['content-length'])
            size, chunk_size = 0, 1024
            with open(file_path, 'wb') as f:
                for data in tqdm.tqdm(r.iter_content(chunk_size=1024), total=total_size // 1024):
                    f.write(data)
                    size = size + len(data)
            if check_file_exist(dir_name, file_name, md5):
                print("\nDownload finished successfully.")
                return file_path
            else:
                print()
        except Exception:
            retry += 1
            print("Network error or file corrupted, retrying, count", retry)
            continue
    raise RuntimeError("Network error or file corrupted when downloading from:", url)


def check_file_exist(dir_name, file_name, md5=None):
    """
    Check whether a file exist in a directory. It does not recursively check the sub-directories.

    Args:
        dir_name(str): directory to check.
        file_name(str): the file name.
        md5(str, optional): md5 code of the file, default None. If None, this function does not check the md5 code.

    Returns:
        bool: whether a file exist in a directory.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            dir_name = '~/test'
            file_name = 'images_background.zip'
            file_md5 = '68d2efa1b9178cc56df9314c21c6e718'
            exist = utils.check_file_exist(dir_name, file_name, file_md5)

    """
    dir_name = os.path.expanduser(dir_name)
    file_path = os.path.join(dir_name, file_name)
    if md5 is not None:
        return os.path.exists(file_path) and check_md5(file_path, md5)
    else:
        return os.path.exists(file_path)


def check_md5(file_path, md5):
    """
    Check whether a file is correct using known md5 code.

    Args:
        file_path(str): path of the file.
        md5(str): known md5 code of the file.

    Returns:
        bool: whether the file's md5 code is the given md5.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            file_path = '~/test/images_background.zip'
            true_md5 = '68d2efa1b9178cc56df9314c21c6e718'
            false_md5 = 'md5'
            check1 = utils.check_md5(file_path, true_md5) # True
            check2 = utils.check_md5(file_path, false_md5) # False

    """
    file_path = os.path.expanduser(file_path)
    hash_md5 = hashlib.md5()
    f = open(file_path, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    # print(file_path, hash_md5.hexdigest())
    return hash_md5.hexdigest() == md5


def decompress(file_path, dst_dir=None):
    """
    Decompress the .zip or .tar.gz file into destination directory.

    Args:
        file_path(str): .zip or .tar.gz file path.
        dst_dir(str, optional): destination directory, default None. If None, it will be decompressed into the same
            directory as the original file.

    Returns:
        None.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            file_path = '~/test/images_background.zip'
            utils.decompress(file_path)

    """
    # Set file_path and destination directory path
    file_path = os.path.expanduser(file_path)
    if dst_dir is None:
        dst_dir = os.path.dirname(file_path)
    dst_dir = os.path.expanduser(dst_dir)
    # If the file is .zip file
    if zipfile.is_zipfile(file_path):
        fz = zipfile.ZipFile(file_path, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        fz.close()
    # If the file is .tar.gz file
    if tarfile.is_tarfile(file_path):
        ft = tarfile.open(file_path)
        ft.extractall(dst_dir)


def list_dir(dir_name, prefix=False):
    """
    List all directories at a given root

    Args:
        dir_name (str): Path to directory whose folders need to be listed.
        prefix (bool, optional): If true, prepends the path to each result, otherwise only returns the name of
            the directories found.

    Returns:
        directories(List): a list of str.

    Examples:
        ..code-blocks:: python

            import paddlefsl.utils as utils
            dir_name = '~/test'
            dirs = utils.list_dir(dir_name) # dirs: ['test_module']

    """
    dir_name = os.path.expanduser(dir_name)
    directories = [p for p in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, p))]
    if prefix is True:
        directories = [os.path.join(dir_name, d) for d in directories]
    return directories


def list_files(root: str, suffix: str, prefix: bool = False):
    """
    List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed.
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly.
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found.

    Returns:
        files(List): a list of str.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            dir_name = '~/test/test_module'
            files = utils.list_files(dir_name, '.zip') # files: ['images_background.zip']

    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def clear_file(file_path):
    """
    Clear all the contents in a file.

    Args:
        file_path(str): path of the file. The file must exist before calling this function.

    Returns:
        None.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            file_path = '~/test/test.txt'
            utils.clear_file(file_path)

    """
    with open(file_path, 'w') as f:
        f.seek(0)
        f.truncate()


def get_info_str(*key):
    """
    Returns a continuous string of a list of objects. Each object is transferred to a string and strings are separated
    by '_' to form the final string. If a string of an object is too long or the object is a class, it will be replaced
    by the class name of the object. All the string will be transferred to lower case.

    Args:
        *key: several objects

    Returns:
        information_string(str): a continuous string of the input objects.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            model = paddlefsl.vision.backbones.Conv4(input_size=(84, 84), output_size=5)
            ways = 5
            shots = 5
            info_str = utils.get_info_str(model, ways, 'ways', shots, 'shots') # info_str: 'conv4_5_ways_5_shots'

    """
    result = ''
    for k in key:
        s = str(k).lower()
        if '\n' in s or '<' in s:
            s = type(k).__name__.lower()
        if s == 'sequential':
            s = k._full_name
        if s != '':
            result += s + '_'
    return result[:-1]


def print_training_info(iteration,
                        train_loss,
                        train_acc=None,
                        valid_loss=None,
                        valid_acc=None,
                        report_file=None,
                        info=None):
    """
    Print training information in terminal or in file, including iteration number, training or validation loss and
    accuracy, and other information.

    Args:
        iteration(int): iteration number.
        train_loss(float): training loss.
        train_acc(float, optional): training accuracy, default None.
        valid_loss(float, optional): validation loss, default None.
        valid_acc(float, optional): validation accuracy, default None.
        report_file(str, optional): file path to which the information will be saved, default None. If None, this
            function will only print information in terminal. If not None, this function will add information to
            the tail of the file, so the file must exist before calling this function.
        info(object, optional): other information to print. This will only be printed in terminal after iteration.

    Returns:
        None. This function prints the information in terminal or save information in file.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            train_loss, train_acc = 0.85, 0.76
            utils.print_training_info(0, train_loss, train_acc, info='just a test')

    """
    print('Iteration', iteration, end='\t' if info is not None else '\n')
    if info is not None:
        print(info)
    print('Training Loss', train_loss, end='\t' if train_acc is not None else '\n')
    if train_acc is not None:
        print('Training Accuracy', train_acc)
    if valid_loss is not None:
        print('Validation Loss', valid_loss, end='\t' if train_acc is not None else '\n')
    if valid_acc is not None:
        print('Validation Accuracy', valid_acc)
    print()
    if report_file is not None:
        file = open(report_file, 'a')
        file.write('iter' + '\t' + str(iteration) + '\t')
        file.write(str(train_loss) + '\t')
        if train_acc is not None:
            file.write(str(train_acc) + '\t')
        if valid_loss is not None:
            file.write(str(valid_loss) + '\t')
        if valid_acc is not None:
            file.write(str(valid_acc) + '\t')
        file.write('\n')
        file.close()
