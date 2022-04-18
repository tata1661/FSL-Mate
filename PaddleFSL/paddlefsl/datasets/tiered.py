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

import cv2
from tqdm import tqdm
import pickle as pkl
import numpy as np


def main(file_name):
    image_file = '/home/wangyaqing/zhaozijing/dataset/tiered-imagenet/' + file_name + '.pkl'
    output_file = '/home/wangyaqing/zhaozijing/dataset/tiered-imagenet/' + file_name[:-4] + '.pkl'
    with open(image_file, 'rb') as f:
        image_data, image_list = pkl.load(f), []
        for item in tqdm(image_data, desc=file_name):
            image = cv2.imdecode(item, cv2.IMREAD_COLOR)
            image = np.transpose(image, [2, 0, 1])
            image_list.append(image.tolist())
        image_data = None
    with open(output_file, 'wb') as f:
        pkl.dump(image_list, f)


if __name__ == '__main__':
    for name in ['train_images_png', 'val_images_png', 'test_images_png']:
        main(name)
