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
