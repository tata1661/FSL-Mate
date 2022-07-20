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
from paddlefsl.utils.module import is_valid_module

# if is_valid_module("pahelix"):
#     raise ImportError(
#         f"can't import pahelix package, you can refer to "
#     )

from pahelix.model_zoo.pretrain_gnns_model import PretrainGNNModel

class All_Embedding(nn.Layer):
    """
    Implementation of Node/Edge Encoder
    """
    def __init__(self, num_1, num_2, embed_dim):
        super(All_Embedding, self).__init__()

        self.embed_1 = nn.Embedding(num_1, embed_dim, weight_attr=nn.initializer.XavierUniform())
        self.embed_2 = nn.Embedding(num_2, embed_dim, weight_attr=nn.initializer.XavierUniform())

    def forward(self, features):

        out_embed = self.embed_1(features['feature'][:, 0]) + self.embed_2(features['feature'][:, 1])
        return out_embed

class GIN(nn.Layer):
    """
    Implementation of a GIN(Graph Isomorphism Network) model.
    Import the model from pahelix.model_zoo
    """

    def __init__(self, args):
        super(GIN, self).__init__()
        config_encoder = {
            "atom_names": ["atomic_num", "chiral_tag"],
            "bond_names": ["bond_dir", "bond_type"],

            "residual": False,
            "dropout_rate": args.dropout,
            "gnn_type": "gin",
            "graph_norm": False,

            "embed_dim": args.emb_dim,
            "layer_num": args.enc_layer,
            "JK": args.JK,

            "readout": args.enc_pooling
        }
        self.mol_encoder = PretrainGNNModel(config_encoder)
        self.mol_encoder.atom_embedding = All_Embedding(args.num_node_type, args.num_node_tag, args.emb_dim)
        for i in range(args.enc_layer):
            self.mol_encoder.bond_embedding_list[i] = All_Embedding(args.num_edge_type, args.num_edge_direction, args.emb_dim)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            print('load pretrained model from', model_file)
            params = paddle.load(model_file)
            for i in params:
                params[i] = paddle.to_tensor(params[i])
            self.mol_encoder.load_dict(params)

    def forward(self, data):
        return self.mol_encoder(data.tensor())