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
import paddlefsl


def process_root_test():
    root = '~/test'
    module_dir = utils.process_root(root, module_name='test_module')
    print(module_dir)


def download_url_test():
    background_url = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
    background_md5 = '68d2efa1b9178cc56df9314c21c6e718'
    dir_name = '~/test'
    utils.download_url(dir_name, background_url)


def check_file_exist_test():
    dir_name = '~/test'
    file_name = 'images_background.zip'
    file_md5 = '68d2efa1b9178cc56df9314c21c6e718'
    exist = utils.check_file_exist(dir_name, file_name, file_md5)
    print(exist)


def check_md5_test():
    file_path = '~/test/images_background.zip'
    true_md5 = '68d2efa1b9178cc56df9314c21c6e718'
    false_md5 = 'md5'
    check1 = utils.check_md5(file_path, true_md5)
    check2 = utils.check_md5(file_path, false_md5)
    print(check1, check2)


def decompress_test():
    file_path = '~/test/images_background.zip'
    utils.decompress(file_path)


def list_dir_test():
    dir_name = '~/test'
    dirs = utils.list_dir(dir_name)
    print(dirs)


def list_files_test():
    dir_name = '~/test/test_module'
    files = utils.list_files(dir_name, '.zip')
    print(files)


def clear_file_test():
    file_path = '~/test/test.txt'
    utils.clear_file(file_path)

def get_info_str_test():
    max_len, embedding_dim = 100, 50
    position_embedding = paddlefsl.backbones.RCPositionEmbedding(max_len=max_len, embedding_dim=embedding_dim)
    conv_model = paddlefsl.backbones.RCConv1D(max_len=max_len, embedding_size=position_embedding.embedding_size)
    model = paddle.nn.Sequential(
        position_embedding,
        conv_model
    )
    model._full_name = 'glove50_cnn'
    ways = 5
    shots = 5
    info_str = utils.get_info_str(model, ways, 'ways', shots, 'shots')
    print(info_str)  # 'conv_5_ways_5_shots'


def print_training_info_test():
    train_loss, train_acc = 0.85, 0.76
    utils.print_training_info(0, train_loss, train_acc, info='just a test')
    # 'Iteration 0	just a test'
    # 'Training Loss 0.85	Training Accuracy 0.76'


if __name__ == '__main__':
    process_root_test()
    download_url_test()
    check_file_exist_test()
    check_md5_test()
    decompress_test()
    list_dir_test()
    list_files_test()
    clear_file_test()
    get_info_str_test()
    print_training_info_test()
