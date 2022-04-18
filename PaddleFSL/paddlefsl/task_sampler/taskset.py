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

import paddle
import numpy as np
import random


__all__ = ['TaskSet']


class TaskSet:
    """
    Task set of N ways and K shots according to meta-learning task.

    Args:
        label_names_data(  List[ (object, List[np.ndarray]) ]  ): A list of Tuple(object, List[np.ndarray]).
            The list of ndarray(each represents a sample of data) has the same label of Tuple[0] object.
        ways(int): Number of classes of the task.
        shots(int): Number of samples of each class when training.
        query_num(int, optional): Number of samples for query, default None. If None, it will be set the same as shots.

    Returns:
        TaskSet with N ways and K shots.

    Examples:
        ..code-block:: python

            from paddlefsl.vision.task_sampler import TaskSet
            label1, label0 = 'one', 'zero'
            data1 = [np.ones(shape=(16, 16), dtype=float) for i in range(20)]
            data0 = [np.zeros(shape=(16, 16), dtype=float) for j in range(20)]
            label_names_data = [(label0, data0), (label1, data1)]
            task = TaskSet(label_names_data, ways=2, shots=5)
            data = task.support_data # data: numpy array of shape=(10, 1, 16, 16), shuffled.

    """

    def __init__(self, label_names_data, ways, shots, query_num=None):

        # Set ways and shots
        if len(label_names_data) != ways:
            raise RuntimeError("Classes of the input data do not match 'ways'.")
        self.ways = ways
        self.shots = shots
        self.query_num = shots if query_num is None else query_num

        # Transfer labels to category number
        # self.label_names: List[object(label name)]
        # self._labels_data: List[ (np.ndarray, List[np.ndarray]) ]
        self.label_names = [label_name for (label_name, data) in label_names_data]
        self._labels_data = self._generate_label(label_names_data)

        # Sample support set and query set
        # self.support_data: np.ndarray of data batch, shape: (ways*shots, data.shape)
        # self.support_labels: np.ndarray of shape: (ways*shots, ways)
        support, query = self._sample_support_query()
        self.support_data, self.support_labels = support
        self.query_data, self.query_labels = query

    def label_to_name(self, label):
        """
        Get the original name of the label.

        Args:
            label(numpy.ndarray): label to be translated. Should be a numpy array with shape=[1].

        Returns:
            label_name(str): original name of the label.

        Examples:
            ..code-block:: python

                from paddlefsl.vision.task_sampler import TaskSet
                from paddlefsl.vision.task_sampler import TaskSet
                label1, label0 = 'one', 'zero'
                data1 = [np.ones(shape=(16, 16), dtype=float) for i in range(20)]
                data0 = [np.zeros(shape=(16, 16), dtype=float) for j in range(20)]
                label_names_data = [(label0, data0), (label1, data1)]
                task = TaskSet(label_names_data, ways=2, shots=5)
                data, label = task.support_data[5], task.support_labels[5]
                print(data) # numpy array of shape=(1, 16, 16), all 0 or all 1
                print(label) # numpy array of shape=(1), 0 or 1 according to data
                print(task.label_to_name(label)) # 'zero' or 'one' according to label

        """
        try:
            return self.label_names[int(label.tolist()[0])]
        except Exception:
            raise ValueError('Invalid input label.')

    def transfer_backend(self, target_backend):
        """
        Transfer data backend between numpy.ndarray and paddle.tensor.

        Args:
            target_backend(str): Target backend. Should be one of ['numpy', 'tensor']

        Returns:
            None. Data type in the object will be changed into target backend.

        Examples:
            ..code-block:: python

                from paddlefsl.vision.task_sampler import TaskSet
                from paddlefsl.vision.task_sampler import TaskSet
                label1, label0 = 'one', 'zero'
                data1 = [np.ones(shape=(16, 16), dtype=float) for i in range(20)]
                data0 = [np.zeros(shape=(16, 16), dtype=float) for j in range(20)]
                label_names_data = [(label0, data0), (label1, data1)]
                task = TaskSet(label_names_data, ways=2, shots=5)
                print(type(task.support_data))  # <class 'numpy.ndarray'>
                task.transfer_backend('tensor')
                print(type(task.support_data))  # <class 'paddle.VarBase'>
                task.transfer_backend('numpy')
                print(type(task.support_data))  # <class 'numpy.ndarray'>

        """
        if target_backend not in ['numpy', 'tensor']:
            raise ValueError('Invalid input target_backend. Should be one of ' + str(['numpy', 'tensor']))
        if type(self.support_data) is np.ndarray and target_backend == 'tensor':
            self.support_data = paddle.to_tensor(self.support_data, dtype='float32')
            self.support_labels = paddle.to_tensor(self.support_labels)
            self.query_data = paddle.to_tensor(self.query_data, dtype='float32')
            self.query_labels = paddle.to_tensor(self.query_labels)
        elif type(self.support_data) is paddle.Tensor and target_backend == 'numpy':
            self.support_data, self.support_labels = self.support_data.numpy(), self.support_labels.numpy()
            self.query_data, self.query_labels = self.query_data.numpy(), self.query_labels.numpy()

    def _generate_label(self, label_names_data):
        categories = np.array([[i] for i in range(self.ways)])
        labels_data = []
        for i in range(self.ways):
            data = label_names_data[i][1]
            labels_data.append((categories[i], data))
        return labels_data

    def _sample_support_query(self):
        support_data_labels, query_data_labels = [], []
        for (label, data) in self._labels_data:
            if self.shots > len(data) or self.shots + self.query_num > len(data):
                error_info = 'Not enough samples for current shots and query_num setting. ' \
                             'In class: ' + self.label_to_name(label) + \
                             ' , there is only ' + str(len(data)) + ' samples, but TaskSet requires ' + \
                             str(self.shots) + ' samples for support and ' + str(self.query_num) + ' samples for query.'
                raise RuntimeError(error_info)
            random.shuffle(data)
            support_data_labels.extend([(d, label) for d in data[:self.shots]])
            query_data_labels.extend([(d, label) for d in data[self.shots:self.shots + self.query_num]])
        random.shuffle(support_data_labels)
        random.shuffle(query_data_labels)
        support_data = np.stack([data for (data, label) in support_data_labels], axis=0)
        support_labels = np.stack([label for (data, label) in support_data_labels], axis=0)
        query_data = np.stack([data for (data, label) in query_data_labels], axis=0)
        query_labels = np.stack([label for (data, label) in query_data_labels], axis=0)
        return (support_data, support_labels), (query_data, query_labels)
