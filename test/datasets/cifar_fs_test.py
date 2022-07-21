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

from paddlefsl.datasets import CifarFS

root = '~/.cache/paddle/dataset'
training_set = CifarFS(mode='train', root=root)
validating_set = CifarFS(mode='valid', backend='pil', root=root)


def get_item_test():
    image, label = training_set[16]
    print(image.shape)  # (3, 32, 32)
    print(image)  # A numpy array
    image, label = validating_set[15]
    image.show()  # Shows an image
    print(label)  # 'otter'


def len_test():
    print(len(training_set))  # 38400


def sample_task_set_test():
    task = training_set.sample_task_set(ways=5, shots=5)
    print(task.support_data.shape)  # (25, 3, 32, 32)
    print(task.query_data.shape)  # (25, 3, 32, 32)


if __name__ == '__main__':
    get_item_test()
    len_test()
    sample_task_set_test()
