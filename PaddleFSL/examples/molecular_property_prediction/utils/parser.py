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

import os
import argparse
import random
import numpy as np
import paddle

from paddlefsl.datasets.mol_dataset import obatin_train_test_tasks
from .utils import init_trial_path

def get_parser(root_dir='.',
               n_shot = 10,
               n_query = 16,
               meta_lr = 0.001,
               weight_decay = 5e-5,
               inner_lr = 0.05,
               epochs = 1000,
               eval_steps = 10,
               seed = 0,
               save_model_iter = 2000,
               method = 'maml'):
    parser = argparse.ArgumentParser(description='Task Aware Relation Graph for Few-shot Chemical Property Prediction')
    parser.add_argument('--setting', type=str, default='pre_par',
                        #choices=['par','pre_par']
                        )
    # dataset
    parser.add_argument('-r', '--root-dir', type=str, default=root_dir, help='root-dir')
    parser.add_argument('-d', '--dataset', type=str, default='tox21', help='data set name')  # ['tox21','sider','muv','toxcast']
    parser.add_argument('-td', '--test-dataset', type=str, default='tox21',
                        help='test data set name')  # ['tox21','sider','muv','toxcast']
    parser.add_argument('--preload_train_data', type=bool, default=True)  # 0
    parser.add_argument('--preload_test_data', type=bool, default=True)  # 0
    parser.add_argument("--run_task", type=int, default=-1, help="run on task")

    # few shot
    parser.add_argument("--n-shot-train", type=int, default=n_shot, help="train: number of  shot for each class")
    parser.add_argument("--n-shot-test", type=int, default=n_shot, help="test: number of  shot for each class")
    parser.add_argument("--n-query", type=int, default=n_query, help="number of query in few shot learning")

    # training
    parser.add_argument("--meta-lr", type=float, default=meta_lr,# 0.003, 0.001, 0.0006
                        help="Training: Meta learning rate")  
    parser.add_argument("--weight_decay", type=float, default=weight_decay,
                        help="Training: Meta learning weight_decay")  #5e-5
    parser.add_argument("--inner-lr", type=float, default=inner_lr, help="Training: Inner loop learning rate")  # 0.01 0.5
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='number of epochs to train (default: 100)')  # 2000
    parser.add_argument('--update_step', type=int, default=1)  # 5 1
    parser.add_argument('--update_step_test', type=int, default=1)  # 10
    parser.add_argument('--inner_update_step', type=int, default=1)  # 10
    parser.add_argument("--meta_warm_step", type=int, default=0, help="meta warp up step for encode")  # 9
    parser.add_argument("--meta_warm_step2", type=int, default=10000, help="meta warp up step 2 for encode")
    parser.add_argument("--second_order", type=int, default=1, help="second order or not")  # 9
    parser.add_argument("--batch_task", type=int, default=9, help="Training: Meta batch size")  # 9
    parser.add_argument("--adapt_weight", type=int, default=5, help="adaptable weights")  # 9
    parser.add_argument("--eval_support", type=int, default=0, help="Training: eval s")
    # model
    ## mol-encoder
    parser.add_argument('--enc_gnn', type=str, default="gin")
    parser.add_argument('--enc_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')##last##
    parser.add_argument('--enc_pooling', type=str, default="mean",#mean
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--enc_batch_norm', type=int, default=1,
                        help='use batch norm or not')
    parser.add_argument('--pretrained', type=int, default=1, help='pretrained or not')
    parser.add_argument('--pretrained_weight_path', type=str,
                        default=os.path.abspath(os.path.join(root_dir,"utils/supervised_contextpred.pdparams")), 
                        help='pretrained path')
    parser.add_argument('--num_node_type', type=int, default=120)
    parser.add_argument('--num_node_tag', type=int, default=3)
    parser.add_argument('--num_edge_type', type=int, default=6)
    parser.add_argument('--num_edge_direction', type=int, default=3)
    
    if method == 'par':
        ## context map
        parser.add_argument('--map_dim', type=int, default=128, help='map dimensions ')
        parser.add_argument('--map_layer', type=int, default=2, help='map layer ') 
        parser.add_argument('--map_pre_fc', type=int, default=0, help='pre fc layer ')
        parser.add_argument("--map-dropout", type=float, default=0.1, help="map dropout")
        parser.add_argument('--ctx_head', type=int, default=2, help='context layer ')####2
        ## relation graph
        ### par
        parser.add_argument("--rel-hidden-dim", type=int, default=128, help="hidden dim for relation net")  # 32 # 128
        parser.add_argument("--rel-layer", type=int, default=2, help="number of layers for relation net")
        parser.add_argument("--rel-edge-layer", type=int, default=2, help="number of layers for relation edge update")  # 3
        parser.add_argument("--rel-res", type=float, default=0, help="residual weight of mapper and relation")
        parser.add_argument("--batch_norm", type=int, default=0, help="batch_norm or not")
        parser.add_argument("--rel_edge", type=int, default=2, choices=[1, 2], help="rel edge dim")
        parser.add_argument("--rel_adj", type=str, default='sim', choices=['dist', 'sim'], help="edge update adjacent")
        parser.add_argument("--rel_act", type=str, default='sigmoid', choices=['sigmoid', 'softmax', 'none'],
                            help="edge update adjacent")
        parser.add_argument('--rel_node_concat', type=int, default=0, help='node concat or not')
        parser.add_argument("--rel-dropout", type=float, default=0.1, help="rel dropout")
        parser.add_argument("--rel-dropout2", type=float, default=0.1, help="rel dropout2")

        ## loss term
        ### adjacency reg
        parser.add_argument('--reg_adj', type=float, default=1, help='reg adj loss weight')

    # other
    parser.add_argument('--seed', type=int, default=seed, help="Seed for splitting the dataset.")
    parser.add_argument('--gpu_id', type=int, default=0, help="Choose the number of GPU.")
    parser.add_argument("--result_path", type=str, default=os.path.join(root_dir, 'results/'+method), help="result path")
    parser.add_argument("--eval_steps", type=int, default=eval_steps)
    parser.add_argument("--save-steps", type=int, default=save_model_iter, help="Training: Number of iterations between checkpoints")
    parser.add_argument("--support_valid", type=int, default=0)

    return parser

def get_args(root_dir='.', 
             is_save=True, 
             n_shot = 10,
             n_query = 16,
             meta_lr = 0.001,
             weight_decay = 5e-5,
             inner_lr = 0.05,
             epochs = 1000,
             eval_steps = 10,
             seed = 0,
             save_model_iter = 2000,
             method = 'maml'):
    parser = get_parser(root_dir,n_shot,n_query,meta_lr,weight_decay,inner_lr,epochs,eval_steps,seed,save_model_iter,method)
    args = parser.parse_args()

    args.rel_k= args.n_shot_train
    if args.pretrained:
        args.enc_layer = 5
        args.emb_dim =300
        args.dropout = 0.5 
    if  args.enc_layer<=3:
        args.emb_dim =200
        args.dropout = 0.1   

    if args.test_dataset == args.dataset:
        args.test_dataset = None
    if method == 'par' and args.map_layer<=0:
        args.map_dim = args.emb_dim
    args = init_trial_path(args,is_save)
    print(args)

    train_tasks, test_tasks = obatin_train_test_tasks(args.dataset)
    if args.test_dataset is not None:
        train_tasks_2, test_tasks_2 = obatin_train_test_tasks(args.test_dataset)
        train_tasks = train_tasks + test_tasks
        test_tasks = train_tasks_2 + test_tasks_2
    train_tasks=sorted(list(set(train_tasks)))
    test_tasks=sorted(list(set(test_tasks)))
    if args.run_task>=0:
        train_tasks=[args.run_task]
        test_tasks=[args.run_task]
    args.train_tasks = train_tasks
    args.test_tasks = test_tasks

    #random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        paddle.framework.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

    return args
