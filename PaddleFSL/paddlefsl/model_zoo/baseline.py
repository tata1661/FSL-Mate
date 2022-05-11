# Copyright 2022 PaddleFSL Authors
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
import paddlefsl.utils as utils
import numpy as np
from tqdm import tqdm


def _get_prediction(baseline_score, support_labels):
    acc = utils.classification_acc(baseline_score, support_labels)
    query_labels = paddle.squeeze(support_labels, axis=1)
    loss = paddle.nn.functional.cross_entropy(baseline_score, query_labels.cast('int64'))
    return loss, acc


def meta_training(train_dataset,
                  valid_dataset,
                  model,
                  classifier,
                  lr=None,
                  optimizer=None,
                  epochs=100,
                  episodes=1000,
                  ways=5,
                  shots=5,
                  query_num=15,
                  report_epoch=1,
                  lr_step_epoch=10,
                  save_model_epoch=20,
                  save_model_root='~/trained_models'):
    """
    Implementation of 'Baseline(Baseline Model for Few-shot Learning)[1]', meta-training.
    This function trains a given model with given datasets and hyper-parameters.

    Refs:
        1.Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, Jia-Bin Huang.
        2019. "A Closer Look at Few-shot Classification".

    Args:
        train_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-training.
        valid_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-validation.
        model(paddle.nn.Layer): Embedding model of the baseline under training.
        classifier(paddle.nn.Layer): classifier of the baseline.
        lr(int or paddle.optimizer.lr.LRScheduler, optional): Learning rate. If input int, the function will not use
            lr_scheduler. Default None. If input None, the function will use fixed learning rate 0.001.
        optimizer(paddle.optimizer, optional): Optimizer. Default None. If None, the function will use Adam.
        epochs(int, optional): Training epochs/iterations. Default 100.
        episodes(int): Training episodes per epoch. Default 1000.
        ways(int, optional): Number of classes in a task, default 5.
        shots(int, optional): Number of training samples per class, default 5.
        query_num(int, optional): Number of query points per class, default 15.
        report_epoch(int, optional): number of iterations between printing two reports, default 1.
        lr_step_epoch(int, optional): number of iterations that the learning rate scheduler steps, default 10.
        save_model_epoch(int, optional): number of iterations between saving two model statuses, default 20.
        save_model_root(str, optional): root directory to save model statuses, default '~/trained_models'

    Returns:
        str: directory where model statuses are saved. This function reports the loss and accuracy every 'report_iter'
            iterations in terminal as well as in 'training_report.txt' file. This function saves model status every
            'save_model_iter' iterations as 'epoch_x.params'.

    """
    # Set learning rate scheduler and optimizer
    lr = 0.001 if lr is None else lr
    if optimizer is None:
        optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                          parameters=model.parameters() + classifier.parameters())
    # Set training configuration information and
    module_info = utils.get_info_str('baseline', train_dataset, 'conv', str(ways) + 'ways', str(shots) + 'shots')
    if type(lr) is float:
        train_info = utils.get_info_str('lr' + str(lr), 'episodes' + str(episodes))
    else:
        train_info = utils.get_info_str('lr' + str(lr.base_lr), lr, 'episodes' + str(episodes))
    # Make directory to save report and parameters
    module_dir = utils.process_root(save_model_root, module_info)
    train_dir = utils.process_root(module_dir, train_info)
    report_file = train_dir + '/training_report.txt'
    utils.clear_file(report_file)
    # Meta training iterations

    for epoch in range(epochs):
        train_loss, train_acc, valid_loss, valid_acc = 0.0, 0.0, 0.0, 0.0
        for _ in tqdm(range(episodes), desc='epoch ' + str(epoch + 1)):
            # Clear gradient, loss and accuracy
            optimizer.clear_grad()
            # Sample a task from dataset
            task = train_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
            task.transfer_backend('tensor')
            # Get training prediction loss and accuracy
            model.train()
            classifier.train()
            support_embeddings = model(task.support_data)
            baseline_score = classifier(support_embeddings)
            loss, acc = _get_prediction(baseline_score, task.support_labels)
            train_loss += loss.numpy()[0]
            train_acc += acc
            # Update model
            loss.backward()
            optimizer.step()
            # Validation
            if (epoch + 1) % report_epoch == 0 or epoch + 1 == epochs:
                model.eval()
                classifier.eval()
                task = valid_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
                task.transfer_backend('tensor')
                support_embeddings = model(task.support_data)
                baseline_score = classifier(support_embeddings)
                loss, acc = _get_prediction(baseline_score, task.support_labels)
                valid_loss += loss.numpy()[0]
                valid_acc += acc
        # Average the accumulation through batches
        train_loss, train_acc = train_loss / episodes, train_acc / episodes
        valid_loss, valid_acc = valid_loss / episodes, valid_acc / episodes
        # Learning rate decay
        if type(lr) is not float and (epoch + 1) % lr_step_epoch == 0:
            lr.step()
        # Report training and validation information
        if (epoch + 1) % report_epoch == 0 or epoch + 1 == epochs:
            utils.print_training_info(epoch + 1, train_loss, train_acc, valid_loss, valid_acc,
                                      report_file=report_file, info=[module_info, train_info])
        # Save model
        if (epoch + 1) % save_model_epoch == 0 or epoch + 1 == epochs:
            paddle.save(model.state_dict(), train_dir + '/epoch' + str(epoch + 1) + '.params')
            paddle.save(classifier.state_dict(), train_dir + '/epoch' + str(epoch + 1) + '_classifier.params')
    return train_dir, model, classifier


def meta_testing(test_dataset,
                 model,
                 classifier,
                 epochs=10,
                 episodes=1000,
                 ways=5,
                 shots=5,
                 query_num=15):
    """
    Implementation of 'Baseline(Baseline Model for Few-shot Learning)[1]', meta-testing.
    This function test a trained model with given datasets and hyper-parameters.

    Args:
        test_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-testing.
        model(paddle.nn.Layer): Embedding model of the baseline under training.
        classifier(paddle.nn.Layer): classifier of the baseline.
        epochs(int, optional): Testing epochs/iterations. Default 10.
        episodes(int): Testing episodes per epoch. Default 1000.
        ways(int, optional): Number of classes in a task, default 5.
        shots(int, optional): Number of training samples per class, default 5.
        query_num(int, optional): Number of query points per class, default 15.

    Returns:
        None. This function prints the testing results, including accuracy in each epoch and the average accuracy.

    """
    module_info = utils.get_info_str('baseline', test_dataset, 'conv', str(ways) + 'ways', str(shots) + 'shots')
    loss_list, acc_list = [], []
    model.eval()
    classifier.eval()
    for epoch in range(epochs):
        test_loss, test_acc = 0.0, 0.0
        for _ in range(episodes):
            task = test_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
            task.transfer_backend('tensor')
            support_embeddings = model(task.support_data)
            baseline_score = classifier(support_embeddings)
            loss, acc = _get_prediction(baseline_score, task.support_labels)
            test_loss += loss.numpy()[0]
            test_acc += acc
        loss_list.append(test_loss / episodes)
        acc_list.append(test_acc / episodes)
        print('Test Epoch', epoch, [module_info], 'Loss', test_loss / episodes, '\t', 'Accuracy', test_acc / episodes)
    print('Test finished', [module_info])
    print('Test Loss', np.mean(loss_list), '\tTest Accuracy', np.mean(acc_list), '\tStd', np.std(acc_list))
