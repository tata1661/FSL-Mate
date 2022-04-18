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

import numpy as np
from paddlefsl.task_sampler import TaskSet


label1, label0 = 'one', 'zero'
data1 = [np.ones(shape=(10, 10), dtype=float) for i in range(20)]
data0 = [np.zeros(shape=(10, 10), dtype=float) for j in range(20)]
label_names_images = [(label0, data0), (label1, data1)]
task = TaskSet(label_names_images, ways=2, shots=5)


def label_to_name_test():
    image, label = task.support_data[5], task.support_labels[5]
    print(image, label)
    print(task.label_to_name(label))


def transfer_backend_test():
    task.transfer_backend('tensor')
    print(type(task.support_data))
    task.transfer_backend('numpy')
    print(type(task.support_data))


if __name__ == '__main__':
    label_to_name_test()
    transfer_backend_test()
