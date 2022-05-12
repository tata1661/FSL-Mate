# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from tqdm import tqdm
import numpy as np
import paddlefsl.utils as utils
from PIL import ImageEnhance
from paddle.vision import transforms as pdtransforms


def baselinepp_training(train_loader,
                        baselineppmodel,
                        lossfuc,
                        lr=None,
                        optimizer=None,
                        epochs=400,
                        ways=5,
                        report_epoch=1,
                        lr_step_epoch=10,
                        save_model_epoch=20,
                        save_model_root='~/trained_models'):
    """
    Implementation of 'Baselin++',training.
    This function trains a given model with given dataloader and hyper-parameters.

    Args:
        train_loader: dataloader of traindataset
        baselineppmodel:model under training
        lossfuc:lossfunction
        lr:Learning rate. If input int, the function will not uselr_scheduler. Default None.
         If input None, the function will use fixed learning rate 0.001.
        epochs:Training epochs/iterations. Default 400.
        ways:Number of classes in a task, default 5.
        report_epoch:number of iterations between printing two reports, default 1.
        lr_step_epoch:number of iterations that the learning rate scheduler steps, default 10.
        save_model_epoch:number of iterations between saving two model statuses, default 20.
        save_model_root:root directory to save model statuses, default '~/trained_models'

    Returns: directory where model statuses are saved. This function reports the loss and accuracy every 'report_iter'
            iterations in terminal as well as in 'training_report.txt' file. This function saves model status every
            'save_model_iter' iterations as 'epoch_x.params'.

    """
    # Set learning rate scheduler and optimizer
    lr = 0.001 if lr is None else lr
    if optimizer is None:
        optimizer = paddle.optimizer.Adam(parameters=baselineppmodel.parameters(), learning_rate=lr)
    # Set training configuration information
    module_info = utils.get_info_str('baselineppnet', train_loader, 'conv', str(ways) + 'ways')
    if type(lr) is float:
        train_info = utils.get_info_str('lr' + str(lr))
    else:
        train_info = utils.get_info_str('lr' + str(lr.base_lr))
    # Make directory to save report and parameters
    module_dir = utils.process_root(save_model_root, module_info)
    train_dir = utils.process_root(module_dir, train_info)
    report_file = train_dir + '/training_report.txt'
    utils.clear_file(report_file)
    # Meta training iterations
    baselineppmodel.train()
    for epoch in range(epochs):
        train_loss, train_acc, valid_loss, valid_acc = 0.0, 0.0, 0.0, 0.0
        for X, y, _ in tqdm(train_loader, desc='train ' + str(epoch + 1), leave=False):
            optimizer.clear_grad()
            y_hat = baselineppmodel(X)
            loss = lossfuc(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
        # Average the accumulation through batches
        train_loss = train_loss / len(train_loader)
        # Learning rate decay
        if type(lr) is not float and (epoch + 1) % lr_step_epoch == 0:
            lr.step()
        # Report training information
        if (epoch + 1) % report_epoch == 0 or epoch + 1 == epochs:
            utils.print_training_info(epoch + 1, train_loss,
                                      report_file=report_file, info=[module_info, train_info])
        # Save model
        if (epoch + 1) % save_model_epoch == 0 or epoch + 1 == epochs:
            paddle.save(baselineppmodel.state_dict(), train_dir + '/epoch' + str(epoch + 1) + '.params')
    return train_dir, baselineppmodel


def baselinepp_testing(test_dataset,
                       baselineppmodel,
                       lossfuc,
                       optimizerdict=None,
                       epochs=600,
                       episodes=100,
                       innerbatch_size=4,
                       ways=5,
                       shots=5,
                       query_num=16):
    """
    Implementation of 'Baselinepp', testing.
    This function test a trained model with given datasets and hyper-parameters.
    Args:
        test_dataset: Dataset for meta-testing.
        baselineppmodel:model under testing
        lossfuc:lossfunction
        lr: Learning rate.If input None, the function will use fixed learning rate 0.001.
        optimizerdict:dict of Optimizer. Default None. If None, the function will use Momentum.
        epochs:Testing epochs/iterations. Default 600.
        episodes:Testing episodes per epoch. Default 100.
        innerbatch_size:innerbatchsize of every episodes
        ways:Number of classes in a task, default 5.
        shots:Number of training samples per class, default 5.
        query_num:Number of query points per class, default 16.

    Returns:
        model after testing
    """
    # Set learning rate scheduler and optimizer
    module_info = utils.get_info_str('baselineppnet', test_dataset, 'conv', str(ways) + 'ways', str(shots) + 'shots')
    loss_list, acc_list = [], []
    for epoch in range(epochs):
        baselineppmodel.resetclassifier()
        task = test_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
        task.transfer_backend('tensor')
        support_query = paddle.concat(x=[task.support_data, task.query_data], axis=0)
        # get the output of encoder
        feature = baselineppmodel.encoder(support_query).detach()
        support_feature = feature[:ways * shots]
        query_feature = feature[ways * shots:]
        support_size = ways * shots
        # finetune
        baselineppmodel.train()

        if optimizerdict is None:
            optimizerdict = {'optimizer': paddle.optimizer.Momentum,
                             'param': {'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.01}}

        optimizer = optimizerdict['optimizer'](parameters=baselineppmodel.classifier.parameters(),
                                               **optimizerdict['param'])
        for _ in tqdm(range(episodes), desc='test ' + str(epoch + 1), leave=False):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, innerbatch_size):
                optimizer.clear_grad()
                selected_id = paddle.to_tensor(rand_id[i: min(i + innerbatch_size, support_size)])
                support_feature_batch = support_feature[selected_id]
                support_label_batch = task.support_labels[selected_id]

                scores = baselineppmodel.classifier(support_feature_batch)

                loss = lossfuc(scores, support_label_batch)
                loss.backward()
                optimizer.step()

        baselineppmodel.eval()
        # test
        with paddle.no_grad():
            query_hat = baselineppmodel.classifier(query_feature)
            test_loss = lossfuc(query_hat, task.query_labels).numpy()
        test_acc = paddle.static.accuracy(query_hat, task.query_labels.reshape((-1, 1))).numpy()
        loss_list.append(test_loss)
        acc_list.append(test_acc)
        print('Test Epoch', epoch, [module_info], 'Loss', test_loss, '\t', 'Accuracy', test_acc)
    print('Test finished', [module_info])
    print('Test Loss', np.mean(loss_list), '\tTest Accuracy', np.mean(acc_list), '\tStd', np.std(acc_list))
    return baselineppmodel


class ImageJitter(object):
    """
    Data augmentation
    """

    def __init__(self, transformdict):
        transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                                 Sharpness=ImageEnhance.Sharpness,
                                 Color=ImageEnhance.Color)
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = paddle.rand([len(self.transforms)])
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


def get_baselinepp_transfer(mode='train', image_size=84):
    """
    get a transferm for Data augmentation

    Args:
        mode: must be one of['train','valid','test']
        image_size: output size of image

    Returns:a transferm

    """
    if mode == 'train':
        transfer = pdtransforms.Compose([
            pdtransforms.RandomResizedCrop(image_size),
            ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
            pdtransforms.RandomHorizontalFlip(),
            pdtransforms.ToTensor(),
            pdtransforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])])
    elif mode == 'valid' or mode == 'test':
        transfer = pdtransforms.Compose([
            pdtransforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
            pdtransforms.CenterCrop((image_size, image_size)),
            pdtransforms.ToTensor(),
            pdtransforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])])
    else:
        transfer = None
    return transfer
