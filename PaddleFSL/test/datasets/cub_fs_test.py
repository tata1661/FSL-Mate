from paddlefsl.datasets import CubFS

root = '~/.cache/paddle/dataset'
training_set = CubFS(mode='train', root=root)
validating_set = CubFS(mode='valid', root=root, backend='pil')


def get_item_test():
    image, label = training_set[16]
    print(image.shape)  # (3, 84, 84)
    print(image)  # A numpy array
    image, label = validating_set[15]
    image.show()  # Shows an image
    print(label)  # 0


def len_test():
    print(len(training_set))  # 5884


def sample_task_set_test():
    task = training_set.sample_task_set(ways=5, shots=5)
    print(task.support_data.shape)  # (25, 3, 84, 84)
    print(task.query_data.shape)  # (25, 3, 84, 84)


if __name__ == '__main__':
    get_item_test()
    len_test()
    sample_task_set_test()
