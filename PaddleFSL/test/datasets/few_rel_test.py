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

from paddlefsl.datasets import FewRel
from paddlefsl.backbones import RCInitVector

init_vector = RCInitVector()

train_dataset = FewRel(mode='train', max_len=100, vector_initializer=init_vector)
valid_dataset = FewRel(mode='valid', backend='text')


def get_item_test():
    print(train_dataset[0])
    print(valid_dataset[0])


def len_test():
    print(len(train_dataset))
    print(len(valid_dataset))


def sample_task_set_test():
    taskset = train_dataset.sample_task_set(ways=5, shots=1)
    print(taskset.support_data.shape)


def get_label_name_test():
    data, label = train_dataset[10]
    label_name = train_dataset.get_label_name(label)
    print(label, label_name)


if __name__ == '__main__':
    get_item_test()
    len_test()
    sample_task_set_test()
    get_label_name_test()
