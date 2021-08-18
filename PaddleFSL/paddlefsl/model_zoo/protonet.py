import paddle
import paddlefsl.utils as utils
import numpy as np
from tqdm import tqdm


def get_prototypes(support_embeddings, support_labels, ways, shots):
    """
    Get prototypes of the support data in Prototypical Network. Given the support embeddings and labels, returns
    the prototypes of the relative classes in order.

    Args:
        support_embeddings(paddle.Tensor): support embeddings calculated by embedding model. shape:
            (batch_size, embedding_size), batch_size should be ways * shots, embedding_size can be multiple dimensions.
        support_labels(paddle.Tensor): labels of the embeddings. shape: (batch_size, 1). Support embedding number of
            each label should be the same and equal to shots.
        ways(int): ways of the task
        shots(int): shots of the task

    Returns:
        prototypes(Paddle.Tensor): shape: (ways, embedding_size). prototype[0] represents the prototype of label 0.
            According to Prototypical Network, the prototype of a label is the mean value of the support embeddings
            of the label.

    Examples:
        ..code-block:: python

            import paddle
            from paddlefsl.model_zoo import protonet

            support_embeddings = paddle.to_tensor([[1.1, 1.1, 1.1],
                                                   [0.0, 0.0, 0.0],
                                                   [0.9, 0.9, 0.9],
                                                   [0.0, 0.0, 0.0]])  # embedding size is 3
            support_labels = paddle.to_tensor([[1], [0], [1], [0]])
            prototypes = protonet.get_prototypes(support_embeddings, support_labels, ways=2, shots=2)
            print(prototypes)  # Tensor(shape=[2, 3], ... [[0., 0., 0.], [1., 1., 1.]])

    """
    prototypes = [paddle.zeros_like(support_embeddings[0]) for _ in range(ways)]
    for i in range(len(support_embeddings)):
        proto_idx = int(support_labels.numpy()[i])
        prototypes[proto_idx] += support_embeddings[i]
    prototypes = [prototype / shots for prototype in prototypes]
    prototypes = paddle.stack(prototypes)
    return prototypes


def _get_prediction(prototypes, query_embeddings, query_labels):
    shape = (query_embeddings.shape[0], prototypes.shape[0], prototypes.shape[1])
    prototypes = paddle.expand(prototypes, shape=shape)
    query_embeddings = paddle.unsqueeze(query_embeddings, axis=1)
    query_embeddings = paddle.broadcast_to(query_embeddings, shape=shape)
    output = - paddle.mean((prototypes - query_embeddings) ** 2, axis=-1)
    loss = paddle.nn.functional.cross_entropy(output, query_labels)
    loss = paddle.mean(loss, axis=0)
    acc = utils.classification_acc(paddle.nn.functional.softmax(output), query_labels)
    return loss, acc


def meta_training(train_dataset,
                  valid_dataset,
                  model,
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
    Implementation of 'ProtoNet(Prototypical Networks for Few-shot Learning)[1]', meta-training.
    This function trains a given model with given datasets and hyper-parameters.

    Refs:
        1.Snell J, Swersky K, Zemel R S. 2017. "Prototypical Networks for Few-shot Learning".

    Args:
        train_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-training.
        valid_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-validation.
        model(paddle.nn.Layer): Embedding model of the ProtoNet under training.
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
        optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)
    # Set training configuration information and
    module_info = utils.get_info_str('protonet', train_dataset, model, str(ways) + 'ways', str(shots) + 'shots')
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
            support_embeddings, query_embeddings = model(task.support_data), model(task.query_data)
            prototypes = get_prototypes(support_embeddings, task.support_labels, ways, shots)
            loss, acc = _get_prediction(prototypes, query_embeddings, task.query_labels)
            train_loss += loss.numpy()[0]
            train_acc += acc
            # Update model
            loss.backward()
            optimizer.step()
            # Validation
            if (epoch + 1) % report_epoch == 0 or epoch + 1 == epochs:
                model.eval()
                task = valid_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
                task.transfer_backend('tensor')
                support_embeddings, query_embeddings = model(task.support_data), model(task.query_data)
                prototypes = get_prototypes(support_embeddings, task.support_labels, ways, shots)
                loss, acc = _get_prediction(prototypes, query_embeddings, task.query_labels)
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
    return train_dir


def meta_testing(test_dataset,
                 model,
                 epochs=10,
                 episodes=1000,
                 ways=5,
                 shots=5,
                 query_num=15):
    """
    Implementation of 'ProtoNet(Prototypical Networks for Few-shot Learning)[1]', meta-testing.
    This function test a trained model with given datasets and hyper-parameters.

    Args:
        test_dataset(paddlefsl.vision.dataset.FewShotDataset): Dataset for meta-testing.
        model(paddle.nn.Layer): Embedding model of the ProtoNet under testing.
        epochs(int, optional): Testing epochs/iterations. Default 10.
        episodes(int): Testing episodes per epoch. Default 1000.
        ways(int, optional): Number of classes in a task, default 5.
        shots(int, optional): Number of training samples per class, default 5.
        query_num(int, optional): Number of query points per class, default 15.

    Returns:
        None. This function prints the testing results, including accuracy in each epoch and the average accuracy.

    """
    module_info = utils.get_info_str('protonet', test_dataset, model, str(ways) + 'ways', str(shots) + 'shots')
    loss_list, acc_list = [], []
    model.eval()
    for epoch in range(epochs):
        test_loss, test_acc = 0.0, 0.0
        for _ in range(episodes):
            task = test_dataset.sample_task_set(ways=ways, shots=shots, query_num=query_num)
            task.transfer_backend('tensor')
            support_embeddings, query_embeddings = model(task.support_data), model(task.query_data)
            prototypes = get_prototypes(support_embeddings, task.support_labels, ways, shots)
            loss, acc = _get_prediction(prototypes, query_embeddings, task.query_labels)
            test_loss += loss.numpy()[0]
            test_acc += acc
        loss_list.append(test_loss / episodes)
        acc_list.append(test_acc / episodes)
        print('Test Epoch', epoch, [module_info], 'Loss', test_loss / episodes, '\t', 'Accuracy', test_acc / episodes)
    print('Test finished', [module_info])
    print('Test Loss', np.mean(loss_list), '\tTest Accuracy', np.mean(acc_list), '\tStd', np.std(acc_list))
