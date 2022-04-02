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
import paddle.nn as nn
import paddle.nn.functional as F
import paddlefsl
import paddlefsl.utils as utils
from time import time
import random
import os
import numpy as np
import pgl.graph as G


def run(args, train_dataset, test_dataset, logger):

    log_names = ['Epoch']
    log_names += ['AUC-' + str(t) for t in args.test_tasks]
    log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
    logger.set_names(log_names)

    model_0 = ContextAwareRelationNet(args)
    utils.count_model_params(model_0)
    optimizer = paddle.optimizer.AdamW(parameters=model_0.parameters(), learning_rate=args.meta_lr, weight_decay=args.weight_decay, grad_clip=nn.ClipGradByNorm(1))
    criterion = nn.CrossEntropyLoss()

    t1 = time()
    print('Initial Evaluation')
    best_avg_auc = 0
    for epoch in range(1, args.epochs + 1):
        print('----------------- Epoch:', epoch,' -----------------')
        meta_training(args, model_0, optimizer, criterion, train_dataset, epoch)

        if epoch % args.eval_steps == 0 or epoch == 1 or epoch == args.epochs:
            print('Evaluation on epoch',epoch)
            best_avg_auc = meta_testing(args, model_0, criterion, test_dataset, epoch, best_avg_auc, logger)

        if epoch % args.save_steps == 0:
            save_path = os.path.join(args.trial_path, f"step_{epoch}.pth")
            paddle.save(model_0.state_dict(), save_path)
            print(f"Checkpoint saved in {save_path}")
        print('Time cost (min):', round((time()-t1)/60,3))
        t1=time()

    df = logger.conclude()
    logger.close()
    print(df)

    print('Train done.')
    print('Best Avg AUC:',best_avg_auc)
    return

def meta_training(args, model_0, optimizer, criterion, train_dataset, epoch):

    task_id_list = list(range(len(args.train_tasks)))
    if args.batch_task > 0:
        batch_task = min(args.batch_task, len(task_id_list))
        task_id_list = random.sample(task_id_list, batch_task)
    data_batches={}
    for task_id in task_id_list:
        task = args.train_tasks[task_id]
        db = train_dataset[task].get_train_data_sample(task, args.n_shot_train, args.n_query)
        data_batches[task_id]=db

    for k in range(args.update_step):
        losses_eval = []
        for task_id in task_id_list:
            train_data, _ = data_batches[task_id]
            model = utils.clone_model(model_0)
            model.train()

            for inner_step in range(args.inner_update_step):
                pred_adapt = get_prediction(model, train_data, train=True)
                loss_adapt = get_loss(criterion, train_data, pred_adapt, train=True, flag = 1)

                adapt_gradient_descent(model, args.inner_lr, loss_adapt)

            pred_eval = get_prediction(model, train_data, train=True)
            loss_eval = get_loss(criterion, train_data, pred_eval, train=True, flag = 0)

            losses_eval.append(loss_eval)

        losses_eval = paddle.stack(losses_eval)
        losses_eval = paddle.sum(losses_eval)

        losses_eval = losses_eval / len(task_id_list)

        optimizer.clear_grad()
        losses_eval.backward()
        optimizer.step()

        print('Train Epoch:', epoch,', train update step:', k, ', loss_eval:', losses_eval.numpy()[0])

    return model_0


def meta_testing(args, model_0, criterion, test_dataset, epoch, best_auc, logger):
    auc_scores = []
    for task_id in range(len(args.test_tasks)):
        task = args.test_tasks[task_id]
        adapt_data, eval_data = test_dataset[task].get_test_data_sample(task, args.n_shot_test, args.n_query, args.update_step_test)
        model = utils.clone_model(model_0)
        
        if args.update_step_test>0:
            model.train()
            
            for i, batch in enumerate(adapt_data['data_loader']):
                cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                'q_data': G.Graph.batch(batch), 'q_label': None}

                pred_adapt = get_prediction(model, cur_adapt_data, train=True)
                loss_adapt = get_loss(criterion, cur_adapt_data, pred_adapt, train=False)

                adapt_gradient_descent(model, args.inner_lr, loss_adapt)

                if i>= args.update_step_test-1:
                    break

        model.eval()
        with paddle.no_grad():
            pred_eval = get_prediction(model, eval_data, train=False)
            y_score = F.softmax(pred_eval['logits'],axis=-1).detach()[:,1]
            y_true = pred_eval['labels']
            if args.eval_support:
                y_s_score = F.softmax(pred_eval['s_logits'],axis=-1).detach()[:,1]
                y_s_true = eval_data['s_label']
                y_score=paddle.concat([y_score, y_s_score])
                y_true=paddle.concat([y_true, y_s_true])

            mm = paddle.metric.Auc()
            mm.update(preds = np.concatenate((1 - y_score.unsqueeze(-1).numpy(),y_score.unsqueeze(-1).numpy()),axis = 1), labels = y_true.unsqueeze(-1).numpy())
            auc = mm.accumulate()
            
        auc_scores.append(auc)

        print('Test Epoch:',epoch,', test for task:', task_id, ', AUC:', round(auc, 4))

    mid_auc = np.median(auc_scores)
    avg_auc = np.mean(auc_scores)
    best_auc = max(best_auc,avg_auc)
    logger.append([epoch] + auc_scores  +[avg_auc, mid_auc,best_auc], verbose=False)

    print('Test Epoch:', epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
            ', Best_Avg_AUC: ', round(best_auc, 4))

    return best_auc

def get_prediction(model, data, train=True):
    if train:
        s_logits, q_logits, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
        pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'node_emb': node_emb}
    else:
        s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
        pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

    return pred_dict

def get_loss(criterion, batch_data, pred_dict, train=True, flag = 0):
    if not train:
        losses_adapt = criterion(pred_dict['s_logits'], batch_data['s_label'])
    else:
        if flag:
            losses_adapt = criterion(pred_dict['s_logits'], batch_data['s_label'])
        else:
            losses_adapt = criterion(pred_dict['q_logits'], batch_data['q_label'])
    if paddle.isnan(losses_adapt).any() or paddle.isinf(losses_adapt).any():
        print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
        print(pred_dict['s_logits'])
        losses_adapt = paddle.zeros_like(losses_adapt)
    return losses_adapt

def adapt_gradient_descent(model, lr, loss, approximate=True):  
    # Do gradient descent on parameters
    gradients = []
    if len(model.parameters()) != 0:
        gradients = paddle.grad(loss,
                                model.parameters(),
                                retain_graph=not approximate,
                                create_graph=not approximate,
                                allow_unused=True)
    update_values = [- lr * grad if grad is not None else None for grad in gradients]
    for param, update in zip(model.parameters(), update_values):
        if update is not None:
            param.set_value(param.add(update))


class ContextAwareRelationNet(nn.Layer):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()

        self.graph_encoder = paddlefsl.backbones.GIN(args)
        self.fc_n = nn.Linear(args.emb_dim, 2)


    def to_one_hot(self,class_idx, num_classes=2):
        return paddle.eye(num_classes)[class_idx]

    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
        s_node_emb, s_emb = self.graph_encoder(s_data)
        q_node_emb, q_emb = self.graph_encoder(q_data)
        s_logits = self.fc_n(s_emb)
        q_logits = self.fc_n(q_emb)

        return s_logits, q_logits, s_node_emb

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False):
        _, s_emb = self.graph_encoder(s_data)

        y_true_list=[]
        q_logits_list, adj_list = [], []
        for q_data in q_loader:
            y_true_list.append(paddle.to_tensor(np.stack([i.y[0] for i in q_data])))
            q_data = G.Graph.batch(q_data)
            q_data.tensor()
            _, q_emb = self.graph_encoder(q_data)

            s_logit = self.fc_n(s_emb)
            q_logit = self.fc_n(q_emb)
            q_logits_list.append(q_logit)

        q_logits = paddle.concat(q_logits_list, 0)
        y_true = paddle.concat(y_true_list, 0)
        sup_labels={'support':s_label,'query':y_true_list}
        return s_logit, q_logits, y_true,adj_list,sup_labels

