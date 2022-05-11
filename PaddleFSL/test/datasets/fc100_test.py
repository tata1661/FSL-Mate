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

from paddlefsl.datasets import FC100

train_dataset = FC100(mode='train', root='~/.cache/paddle/dataset')


def get_item_test():
    image, label = train_dataset[3001]
    print(image, label)  # A numpy array and label 70


def len_test():
    print(len(train_dataset))  # 36000


def sample_taskset_test():
    task = train_dataset.sample_task_set()
    print(task.support_data)  # A numpy array
    print(task.query_data.shape)  # (25, 3, 32, 32)


if __name__ == '__main__':
    get_item_test()
    len_test()
    sample_taskset_test()
