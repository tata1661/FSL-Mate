from paddlefsl.datasets import TieredImageNet

validation_set = TieredImageNet(mode='valid', root='~/.cache/paddle/dataset')


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
