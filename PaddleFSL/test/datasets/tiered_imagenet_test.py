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

from paddlefsl.datasets import TieredImageNet

validation_set = TieredImageNet(mode='valid')


def get_item_test():
    image, label = validation_set[0]
    print(image, label)
    import numpy as np
    from PIL import Image
    image = np.transpose(image, [1, 2, 0])
    image = Image.fromarray(image)
    image.show()


def len_test():
    print(len(validation_set))


def sample_task_set_test():
    task = validation_set.sample_task_set(ways=5, shots=5)
    print(task.support_data.shape)
    print(task.query_data.shape)


if __name__ == '__main__':
    get_item_test()
    len_test()
    sample_task_set_test()
