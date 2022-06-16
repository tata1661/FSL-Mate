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
import paddle.nn.functional as F


def get_match(embeddings, labels, ways, shots):
    match_embeddings = [paddle.zeros_like(embeddings[0]) for _ in range(ways * shots)]
    class_c = paddle.zeros([ways])
    for i in range(len(embeddings)):
        idx = int(labels.numpy()[i])
        match_embeddings[idx * shots + class_c[idx]] += embeddings[i]
        class_c[idx] += 1
    match_embeddings = paddle.stack(match_embeddings)
    return match_embeddings


def meta_training(train_dataset,
                  valid_dataset,
                  feature_model,
                  gnn_model,
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
    Implementation of 'GNN(Learning to Compare: Graph Network for Few-Shot Learning)[1]', meta-training.
    This function trains a given model with given datasets and hyper-parameters.

    Refs:
        1.V. Garcia and J. Bruna. 2018. "Few-Shot Learning with Graph Neural Networks."

    Args:
        train_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-training.
        valid_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-validation.
        feature_model(paddle.nn.Layer): Embedding model of the GNN.
        gnn_model(paddle.nn.Layer): Graph model of the GNN.
        lr(int or paddle.optimizer.lr.LRScheduler, optional): Learning rate. If input int, the function will not use
            lr_scheduler. Default None. If input None, the function will use fixed learning rate 0.001.
        optimizer(paddle.optimizer, optional): Optimizer. Default None. If None, the function will use Adam.
        epochs(int, optional): Training epochs/iterations. Default 100.
        episodes(int): Training episodes per epoch. Default 1000.
        ways(int, optional): Number of classes in a task, default 5.
        shots(int, optional): Number of training samples per class, default 5.
        query_num(int, optional): Number of query points per class, default 15.
        report_epoch(int, optional): number of iterations between printing two reports, default 1.
        lr_step_epoch(int, optional): number of iterations that learning rate scheduler steps, default 10.
        save_model_epoch(int, optional): number of iterations between saving two model statuses, default 20.
        save_model_root(str, optional): root directory to save model statuses, default '~/trained_models'

    Returns:
        str: directory where model statuses are saved. This function reports the loss and accuracy every 'report_iter'
            iterations in terminal as well as in 'training_report.txt' file. This function saves model status every
            'save_model_iter' iterations as 'epoch_x_embedding.params' and 'epoch_x_relation.params'.

    """
    # Set learning rate scheduler and optimizer
    lr = 0.001 if lr is None else lr
    if optimizer is None:
        optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                          parameters=feature_model.parameters() + gnn_model.parameters())
    # Set training configuration information
    module_info = utils.get_info_str('gnn', train_dataset, feature_model,
                                     str(ways) + 'ways', str(shots) + 'shots')
    if type(lr) is not float:
        train_info = utils.get_info_str('lr' + str(lr.base_lr), lr, 'episodes' + str(episodes))
    else:
        train_info = utils.get_info_str('lr' + str(lr), 'episodes' + str(episodes))
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
            # Calculate relation score
            feature_model.train()
            gnn_model.train()
            support_embeddings, query_embeddings = feature_model(task.support_data), feature_model(task.query_data)
            match_embeddings = get_match(support_embeddings, task.support_labels, ways, shots)
            gnn_score = gnn_model(match_embeddings, query_embeddings, ways, shots, ways * query_num)
            acc = utils.classification_acc(gnn_score, task.query_labels)
            loss = F.cross_entropy(gnn_score, task.query_labels.cast('int64'))
            train_loss += loss.numpy()[0]
            train_acc += acc
            # Update model
            loss.backward()
            optimizer.step()
            # Validation
            if (epoch + 1) % report_epoch == 0 or epoch + 1 == epochs:
                task = valid_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
                task.transfer_backend('tensor')
                feature_model.eval()
                gnn_model.eval()
                support_embeddings, query_embeddings = feature_model(task.support_data), feature_model(task.query_data)
                match_embeddings = get_match(support_embeddings, task.support_labels, ways, shots)
                gnn_score = gnn_model(match_embeddings, query_embeddings, ways, shots, ways * query_num)
                acc = utils.classification_acc(gnn_score, task.query_labels)
                loss = F.cross_entropy(gnn_score, task.query_labels.cast('int64'))
                valid_loss += loss.numpy()[0]
                valid_acc += acc
        # Learning rate decay
        if type(lr) is not float and (epoch + 1) % lr_step_epoch == 0:
            lr.step()
        # Report training and validation information
        if (epoch + 1) % report_epoch == 0 or epoch + 1 == epochs:
            utils.print_training_info(epoch + 1, train_loss / episodes, train_acc / episodes,
                                      valid_loss / episodes, valid_acc / episodes,
                                      report_file=report_file, info=[module_info, train_info])
        # Save model
        if (epoch + 1) % save_model_epoch == 0 or epoch + 1 == epochs:
            paddle.save(feature_model.state_dict(), train_dir + '/epoch' + str(epoch + 1) + '_embedding.params')
            paddle.save(gnn_model.state_dict(), train_dir + '/epoch' + str(epoch + 1) + '_gnn.params')
    return train_dir, feature_model, gnn_model


def meta_testing(test_dataset,
                 feature_model,
                 gnn_model,
                 epochs=10,
                 episodes=1000,
                 ways=5,
                 shots=5,
                 query_num=15):
    """
    Implementation of 'GNN(Learning to Compare: Graph Network for Few-Shot Learning)', meta-training.
    This function test a trained model with given datasets and hyper-parameters.

    Args:
        test_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-testing.
        feature_model(paddle.nn.Layer): Embedding model of the GNN.
        gnn_model(paddle.nn.Layer): Graph model of the GNN.
        epochs(int, optional): Testing epochs/iterations. Default 10.
        episodes(int): Testing episodes per epoch. Default 1000.
        ways(int, optional): Number of classes in a task, default 5.
        shots(int, optional): Number of training samples per class, default 5.
        query_num(int, optional): Number of query points per class, default 15.

    Returns:
        None. This function prints the testing results, including accuracy in each epoch and the average accuracy.

    """
    module_info = utils.get_info_str('gnn', test_dataset, feature_model,
                                     str(ways) + 'ways', str(shots) + 'shots')
    loss_list, acc_list = [], []
    feature_model.eval()
    gnn_model.eval()
    for epoch in range(epochs):
        test_loss, test_acc = 0.0, 0.0
        for _ in range(episodes):
            task = test_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
            task.transfer_backend('tensor')
            support_embeddings, query_embeddings = feature_model(task.support_data), feature_model(task.query_data)
            match_embeddings = get_match(support_embeddings, task.support_labels, ways, shots)
            gnn_score = gnn_model(match_embeddings, query_embeddings, ways, shots, ways * query_num)
            acc = utils.classification_acc(gnn_score, task.query_labels)
            loss = F.cross_entropy(gnn_score, task.query_labels.cast('int64'))
            test_loss += loss.numpy()[0]
            test_acc += acc
        loss_list.append(test_loss / episodes)
        acc_list.append(test_acc / episodes)
        print('Test Epoch', epoch, [module_info], 'Loss', test_loss / episodes, '\t', 'Accuracy', test_acc / episodes)
    print('Test finished', [module_info])
    print('Test Loss', np.mean(loss_list), '\tTest Accuracy', np.mean(acc_list), '\tStd', np.std(acc_list))
