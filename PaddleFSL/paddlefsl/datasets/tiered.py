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
