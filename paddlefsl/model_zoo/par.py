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
from collections import OrderedDict


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
            adaptable_weights = get_adaptable_weights(model, args.adapt_weight)

            for inner_step in range(args.inner_update_step):
                pred_adapt = get_prediction(model, train_data, train=True)
                loss_adapt = get_loss(args, model, criterion, train_data, pred_adapt, train=True, flag = 1)

                adapt_gradient_descent(model, args.inner_lr, loss_adapt, adaptable_weights)

            pred_eval = get_prediction(model, train_data, train=True)
            loss_eval = get_loss(args, model, criterion, train_data, pred_eval, train=True, flag = 0)

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

                adaptable_weights = get_adaptable_weights(model, args.adapt_weight)
                pred_adapt = get_prediction(model, cur_adapt_data, train=True)
                loss_adapt = get_loss(args, model, criterion, cur_adapt_data, pred_adapt, train=False)

                adapt_gradient_descent(model, args.inner_lr, loss_adapt, memo = adaptable_weights)

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
        s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
        pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}
    else:
        s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
        pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

    return pred_dict

def get_loss(args, model, criterion, batch_data, pred_dict, train=True, flag = 0):
    n_support_train = args.n_shot_train
    n_support_test = args.n_shot_test
    n_query = args.n_query
    if not train:
        losses_adapt = criterion(pred_dict['s_logits'].reshape((2*n_support_test*n_query,2)), 
                                        paddle.expand(batch_data['s_label'],[n_query,n_support_test*2]).reshape((1,2*n_support_test*n_query)).squeeze(0))
    else:
        if flag:
            losses_adapt = criterion(pred_dict['s_logits'].reshape((2*n_support_train*n_query,2)), 
                                            paddle.expand(batch_data['s_label'],[n_query,n_support_train*2]).reshape((1,2*n_support_train*n_query)).squeeze(0))
        else:
            losses_adapt = criterion(pred_dict['q_logits'], batch_data['q_label'])
    if paddle.isnan(losses_adapt).any() or paddle.isinf(losses_adapt).any():
        print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
        print(pred_dict['s_logits'])
        losses_adapt = paddle.zeros_like(losses_adapt)

    if args.reg_adj > 0:
        n_support = batch_data['s_label'].shape[0]
        adj = pred_dict['adj'][-1]
        if train:
            if flag:
                s_label = paddle.expand(batch_data['s_label'], [n_query,batch_data['s_label'].shape[0]])
                n_d = n_query * n_support
                label_edge = model.label2edge(args, s_label).reshape((n_d, -1))
                pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
            else:
                s_label = paddle.expand(batch_data['s_label'], [n_query,batch_data['s_label'].shape[0]])
                q_label = batch_data['q_label'].unsqueeze(1)
                total_label = paddle.concat([s_label, q_label], 1)
                label_edge = model.label2edge(args, total_label)[:,:,-1,:-1]
                pred_edge = adj[:,:,-1,:-1]
        else:
            s_label = batch_data['s_label'].unsqueeze(0)
            n_d = n_support * args.rel_edge
            label_edge = model.label2edge(args, s_label).reshape((n_d, -1))
            pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
        adj_loss_val = F.mse_loss(pred_edge, label_edge)
        if paddle.isnan(adj_loss_val).any() or paddle.isinf(adj_loss_val).any():
            print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
            adj_loss_val = paddle.zeros_like(adj_loss_val)

        losses_adapt += args.reg_adj * adj_loss_val
    return losses_adapt

def get_adaptable_weights(model, adapt_weight=None):
    fenc = lambda x: x[0]== 'graph_encoder'
    frel = lambda x: x[0]== 'adapt_relation'
    fedge = lambda x: x[0]== 'adapt_relation' and 'layer_edge'  in x[1]
    fnode = lambda x: x[0]== 'adapt_relation' and 'layer_node'  in x[1]
    fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
    if adapt_weight==0:
        flag=lambda x: not fenc(x)
    elif adapt_weight==1:
        flag=lambda x: not frel(x)
    elif adapt_weight==2:
        flag=lambda x: not (fenc(x) or frel(x))
    elif adapt_weight==3:
        flag=lambda x: not (fenc(x) or fedge(x))
    elif adapt_weight==4:
        flag=lambda x: not (fenc(x) or fnode(x))
    elif adapt_weight==5:
        flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
    elif adapt_weight==6:
        flag=lambda x: not (fenc(x) or fclf(x))
    else:
        flag= lambda x: True
    adaptable_weights = []
    adaptable_names=[]
    for name, p in model.named_parameters():
        names=name.split('.')
        if flag(names):
            adaptable_weights.append(id(p))
            adaptable_names.append(name)
    return adaptable_weights

def adapt_gradient_descent(model, lr, loss, memo=None):
    memo = set() if memo is None else set(memo)
    # Do gradient descent on parameters
    gradients = []
    if len(model.parameters()) != 0:
        gradients = paddle.grad(loss,
                                model.parameters(),
                                retain_graph=False,
                                create_graph=False,
                                allow_unused=True)
    update_values = [- lr * grad if grad is not None else None for grad in gradients]
    for param, update in zip(model.parameters(), update_values):
        if update is not None:
            param_ptr = id(param)
            if param_ptr in memo:
                param.set_value(param.add(update))

class ContextAwareRelationNet(nn.Layer):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()

        self.graph_encoder = paddlefsl.backbones.GIN(args)
        self.encode_projection = ContextMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                    batch_norm=args.batch_norm,dropout=args.map_dropout,
                                    pre_fc=args.map_pre_fc,ctx_head=args.ctx_head)
        self.adapt_relation = TaskAwareRelation(inp_dim=args.map_dim, hidden_dim=args.rel_hidden_dim,
                                                num_layers=args.rel_layer, edge_n_layer=args.rel_edge_layer,
                                                top_k=args.rel_k, res_alpha=args.rel_res,
                                                batch_norm=args.batch_norm,edge_dim=args.rel_edge, adj_type=args.rel_adj,
                                                activation=args.rel_act, node_concat=args.rel_node_concat,dropout=args.rel_dropout,
                                                pre_dropout=args.rel_dropout2)


    def to_one_hot(self,class_idx, num_classes=2):
        return paddle.eye(num_classes)[class_idx]

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False, return_adj=False, return_emb=False):
        if not return_emb:
            s_logits, q_logits, adj = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        else:
            s_logits, q_logits, adj, s_rel_emb, q_rel_emb = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
        if q_pred_adj:
            q_sim = adj[-1][:, 0, -1, :-1]
            q_logits = q_sim @ self.to_one_hot(s_label)

        if not return_emb:
            return s_logits, q_logits, adj
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb

    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
        s_node_emb, s_emb = self.graph_encoder(s_data)
        q_node_emb, q_emb = self.graph_encoder(q_data)
        s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)

        s_logits, q_logits, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)

        return s_logits, q_logits, adj, s_node_emb

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False):
        _, s_emb = self.graph_encoder(s_data)

        y_true_list=[]
        q_logits_list, adj_list = [], []
        for q_data in q_loader:
            y_true_list.append(paddle.to_tensor(np.stack([i.y[0] for i in q_data])))
            q_data = G.Graph.batch(q_data)
            q_data.tensor()
            _, q_emb = self.graph_encoder(q_data)

            s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit)
            if adj is not None:
                sim_adj = adj[-1].detach()
                adj_list.append(sim_adj)

        q_logits = paddle.concat(q_logits_list, 0)
        y_true = paddle.concat(y_true_list, 0)
        sup_labels={'support':s_label,'query':y_true_list}
        return s_logit, q_logits, y_true,adj_list,sup_labels


    def label2edge(self, args, label, mask_diag=True):
        # get size
        num_samples = label.shape[1]
        # reshape
        label_i = paddle.transpose(paddle.expand(label,[num_samples,label.shape[0],label.shape[1]]),[1,2,0])
        label_j = label_i.transpose((0, 2, 1))
        # compute edge
        edge = paddle.cast(paddle.equal(label_i, label_j),'float32')

        # expand
        edge = edge.unsqueeze(1)
        if args.rel_adj == 'dist':
            edge = 1 - edge
        if args.rel_edge == 2:
            edge = paddle.concat([edge, 1 - edge], 1)

        if mask_diag:
            diag_mask = 1.0 - paddle.expand(paddle.eye(edge.shape[2]),[edge.shape[0],args.rel_edge,edge.shape[2],edge.shape[2]])
            edge=edge*diag_mask
        if args.rel_act == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

class MLP(nn.Layer):
    def __init__(self, inp_dim, hidden_dim, num_layers,batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
            if l < num_layers - 1:
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1D(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential()
            for i in layer_list:
                self.network.add_sublayer(i, layer_list[i])
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out

class Attention(nn.Layer):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """
    def __init__(self, dim, num_heads=1, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=False)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = paddle.transpose(self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]),(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ paddle.transpose(k, [0, 1, 3, 2])) * self.scale

        attn = F.softmax(attn, axis = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose((0,2,1,3)).reshape((B, N, C))
        return x



class ContextMLP(nn.Layer):
    def __init__(self, inp_dim, hidden_dim, num_layers,pre_fc=0,batch_norm=False, dropout=0.,ctx_head=1,):
        super(ContextMLP, self).__init__()
        self.pre_fc = pre_fc #0, 1
        if self.pre_fc:
            hidden_dim=int(hidden_dim//2)
            self.attn_layer = Attention(hidden_dim,num_heads=ctx_head,attention_dropout=dropout)        
            self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)
        else:
            self.attn_layer = Attention(inp_dim)    
            inp_dim=int(inp_dim*2)
            self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)

    def forward(self, s_emb, q_emb):
        if self.pre_fc:
            s_emb = self.mlp_proj(s_emb)
            q_emb = self.mlp_proj(q_emb)
        n_support = s_emb.shape[0]
        n_query = q_emb.shape[0]

        s_emb_rep = paddle.expand(s_emb,[n_query, s_emb.shape[0], s_emb.shape[1]])
        q_emb_rep = q_emb.unsqueeze(1)
        all_emb = paddle.concat([s_emb_rep, q_emb_rep], 1)
        orig_all_emb =  all_emb

        n_shot=int(n_support//2)
        all_emb_meann = all_emb[:,:n_shot].mean(1)
        all_emb_meanp = all_emb[:,n_shot:2*n_shot].mean(1)
        neg_proto_emb = paddle.transpose(paddle.expand(all_emb_meann,[n_support + 1, all_emb_meann.shape[0], all_emb_meann.shape[1]]),(1,0,2))
        pos_proto_emb = paddle.transpose(paddle.expand(all_emb_meanp,[n_support + 1, all_emb_meanp.shape[0], all_emb_meanp.shape[1]]),(1,0,2))
        all_emb = paddle.stack([all_emb, neg_proto_emb,pos_proto_emb], 2)
        
        q,s,n,d = all_emb.shape
        x=all_emb.reshape((q*s, n, d))
        attn_x =self.attn_layer(x)
        attn_x=attn_x.reshape((q, s, n, d))
        all_emb = attn_x[:,:,0,]

        all_emb = paddle.concat([all_emb, orig_all_emb],axis = -1)

        if not self.pre_fc:
            all_emb = self.mlp_proj(all_emb)

        return all_emb, None

class NodeUpdateNetwork(nn.Layer):
    def __init__(self, inp_dim, out_dim, n_layer=2, edge_dim=2, batch_norm=False, dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.edge_dim = edge_dim
        num_dims_list = [out_dim] * n_layer  # [num_features * r for r in ratio]
        if n_layer > 1:
            num_dims_list[0] = 2 * out_dim

        # layers
        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2D(
                in_channels=num_dims_list[l - 1] if l > 0 else (self.edge_dim + 1) * inp_dim,
                out_channels=num_dims_list[l],
                kernel_size=1,
                bias_attr=False)
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2D(num_features=num_dims_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0 and l == (len(num_dims_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2D(p=dropout)

        self.network = nn.Sequential()
        for i in layer_list:
            self.network.add_sublayer(i, layer_list[i])

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.shape[0]
        num_data = node_feat.shape[1]

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - paddle.expand(paddle.eye(num_data),[num_tasks, self.edge_dim, num_data, num_data])

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, axis=-1)

        # compute attention and aggregate
        aggr_feat = paddle.bmm(paddle.concat(paddle.split(edge_feat, 2, 1), self.edge_dim).squeeze(1), node_feat)

        node_feat = paddle.transpose(paddle.concat([node_feat, paddle.concat(paddle.split(aggr_feat,2, 1), -1)], -1),(0, 2, 1))

        # non-linear transform
        node_feat = paddle.transpose(self.network(node_feat.unsqueeze(-1)),(0, 2, 1, 3)).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Layer):
    def __init__(self, in_features, hidden_features, n_layer=3, top_k=-1,
                 edge_dim=2, batch_norm=False, dropout=0.0, adj_type='dist', activation='softmax'):
        super(EdgeUpdateNetwork, self).__init__()
        self.top_k = top_k
        self.adj_type = adj_type
        self.edge_dim = edge_dim
        self.activation = activation

        num_dims_list = [hidden_features] * n_layer  # [num_features * r for r in ratio]
        if n_layer > 1:
            num_dims_list[0] = 2 * hidden_features
        if n_layer > 3:
            num_dims_list[1] = 2 * hidden_features
        # layers
        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2D(in_channels=num_dims_list[l - 1] if l > 0 else in_features,
                                                       out_channels=num_dims_list[l],
                                                       kernel_size=1,
                                                       bias_attr=False)
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2D(num_features=num_dims_list[l], )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2D(p=dropout)

        layer_list['conv_out'] = nn.Conv2D(in_channels=num_dims_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential()
        for i in layer_list:
            self.sim_network.add_sublayer(i, layer_list[i])

    def softmax_with_mask(self, adj, mask=None):
        if mask is not None:
            adj_new = adj - (1 - mask.expand_as(adj)) * 1e8
        else:
            adj_new = adj
        n_q, n_edge, n1, n2 = adj_new.shape
        adj_new = adj_new.reshape((n_q * n_edge * n1, n2))
        adj_new = F.softmax(adj_new, dim=-1)
        adj_new = adj_new.reshape((n_q, n_edge, n1, n2))
        return adj_new

    def forward(self, node_feat, edge_feat=None):  # x: bs*N*num_feat
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = paddle.transpose(x_i, (0,2,1,3))
        x_ij = paddle.abs(x_i - x_j) # size: bs x fs X N x N  (2,128,11,11)
        x_ij = paddle.transpose(x_ij, (0,3,2,1))
        if self.adj_type == 'sim':
            x_ij = paddle.exp(-x_ij)

        sim_val = self.sim_network(x_ij)
        diag_mask = 1.0 - paddle.expand(paddle.eye(node_feat.shape[1]),[node_feat.shape[0], 1, node_feat.shape[1], node_feat.shape[1]])
        if self.activation == 'softmax':
            sim_val = self.softmax_with_mask(sim_val, diag_mask)
        elif self.activation == 'sigmoid':
            sim_val = F.sigmoid(sim_val) * diag_mask
        else:
            sim_val = sim_val * diag_mask

        if self.edge_dim == 2:
            if self.activation == 'softmax':
                dsim_val = self.softmax_with_mask(1 - sim_val, diag_mask)
            else:
                dsim_val = (1 - sim_val) * diag_mask
            adj_val = paddle.concat([sim_val, dsim_val], 1)
        else:
            adj_val = sim_val

        if self.top_k > 0:
            n_q, n_edge, n1, n2 = adj_val.shape
            k = min(self.top_k,n1)
            adj_temp = adj_val.reshape((n_q*n_edge*n1,n2))
            topk, indices = paddle.topk(adj_temp, k)
            mask = F.one_hot(indices,adj_temp.shape[1]).sum(1)
            mask = mask.reshape((n_q, n_edge, n1, n2))
            if self.activation == 'softmax':
                adj_val = self.softmax_with_mask(adj_val, mask)
            else:
                adj_val = adj_val * mask

        return adj_val, edge_feat


class TaskAwareRelation(nn.Layer):
    def __init__(self, inp_dim, hidden_dim, num_layers, edge_n_layer, num_class=2,
                res_alpha=0., top_k=-1, node_concat=True, batch_norm=False, dropout=0.0,
                 edge_dim=2, adj_type='sim', activation='softmax',pre_dropout=0.0):
        super(TaskAwareRelation, self).__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_concat = node_concat
        self.res_alpha = res_alpha
        self.dropout_rate = dropout
        self.pre_dropout = pre_dropout
        self.adj_type=adj_type
        node_n_layer = max(1, min(int(edge_n_layer // 2), 2))
        gnn_inp_dim = self.inp_dim
        if self.pre_dropout>0:
            self.predrop1 = nn.Dropout(p=self.pre_dropout)
        
        self.layer_edge = nn.LayerList()
        self.layer_node = nn.LayerList()

        for i in range(self.num_layers):
            module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                         top_k=top_k,
                                         edge_dim=edge_dim, batch_norm=batch_norm, adj_type=adj_type,
                                         activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
            module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                         edge_dim=edge_dim, batch_norm=batch_norm,
                                         dropout=dropout if i < self.num_layers - 1 else 0.0)
            self.layer_edge.append(module_w)
            self.layer_node.append(module_l)

            if self.node_concat:
                gnn_inp_dim = gnn_inp_dim + hidden_dim
            else:
                gnn_inp_dim = hidden_dim

        self.fc1 = nn.Sequential(nn.Linear(gnn_inp_dim, inp_dim), nn.LeakyReLU())
        if self.pre_dropout>0:
            self.predrop2 = nn.Dropout(p=self.pre_dropout)
        self.fc2 = nn.Linear(inp_dim, num_class)

        assert 0 <= res_alpha <= 1

    def forward(self, all_emb, q_emb=None, return_adj=False, return_emb=False):
        node_feat=all_emb
        if self.pre_dropout>0:
            node_feat=self.predrop1(node_feat)
        edge_feat_list = []
        if return_adj:
            x_i = node_feat.unsqueeze(2)
            x_j = paddle.transpose(x_i, (1, 2))
            init_adj = paddle.abs(x_i - x_j)
            init_adj = paddle.transpose(init_adj, (1, 3))# size: bs x fs X N x N  (2,128,11,11)
            if self.adj_type == 'sim':
                init_adj = paddle.exp(-init_adj)
            diag_mask = 1.0 - paddle.expand(paddle.eye(node_feat.shape[1]),[node_feat.shape[0], 1, 1, 1])
            init_adj = init_adj*diag_mask
            edge_feat_list.append(init_adj)
        
        for i in range(self.num_layers):
            adj, _ = self.layer_edge[i](node_feat)
            node_feat_new = self.layer_node[i](node_feat, adj)
            if self.node_concat:
                node_feat = paddle.concat([node_feat, node_feat_new], 2)
            else:
                node_feat = node_feat_new
            edge_feat_list.append(adj)
        if self.pre_dropout>0:
            node_feat=self.predrop2(node_feat)
        node_feat = self.fc1(node_feat)
        node_feat = self.res_alpha * all_emb +  node_feat

        s_feat = node_feat[:, :-1, :]
        q_feat = node_feat[:, -1, :]

        s_logits = self.fc2(s_feat)
        q_logits = self.fc2(q_feat)
        if return_emb:
            return s_logits, q_logits, edge_feat_list, s_feat,q_feat
        else:
            return s_logits, q_logits, edge_feat_list