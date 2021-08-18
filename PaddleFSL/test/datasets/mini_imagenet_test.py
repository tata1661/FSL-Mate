from paddlefsl.datasets import MiniImageNet

root = '~/.cache/paddle/dataset'
training_set = MiniImageNet(mode='train', root=root)
validation_set = MiniImageNet(mode='valid', root=root, backend='pil')
testing_set = MiniImageNet(mode='test', root=root)


def get_item_test():
    image, label = validation_set[0]
    image.show()
    image, label = training_set[0]
    print(image, label)  # A numpy array and label 'n01532829'


def len_test():
    print(len(training_set))  # 38400


def sample_task_set_test():
    task = training_set.sample_task_set(ways=5, shots=5)
    print(task.support_data.shape)  # (25, 3, 84, 84)
    print(task.query_data.shape)  # (25, 3, 84, 84)


if __name__ == '__main__':
    get_item_test()
    len_test()
    sample_task_set_test()
