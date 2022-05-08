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
import json
import numpy as np
import paddle
import random
import pgl.graph as G
from pahelix.datasets import InMemoryDataset


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    allowable_features = {
        'possible_atomic_num_list' : list(range(1, 119)),
        'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'possible_chirality_list' : [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list' : [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
        ],
        'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'possible_bonds' : [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'possible_bond_dirs' : [ # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]
    }
except:
    print('Error rdkit:')
    Chem, AllChem, allowable_features=None,None, None


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = np.array(atom_features_list, dtype='int64')

    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        for i in range(len(x)):
            edges_list.append((i, i))
            edge_features_list.append([4, 0])
        edge_index = np.array(edges_list, dtype='int64')

        edge_attr = np.array(edge_features_list, dtype='int64')

    else:   # mol has no bonds
        edge_index = np.array([[0,0]], dtype = 'int64')
        edge_attr = np.array([[4,0]], dtype = 'int64')

    data = {'x':x, 'edge_index':edge_index, 'edge_attr':edge_attr}

    return data


def load_dataset(dataset = 'tox21', type = 'train'):
    load_data = {}
    if type == 'train':
        tasks,_ = obatin_train_test_tasks(dataset)
    else:
        _,tasks = obatin_train_test_tasks(dataset)
    tasks=sorted(list(set(tasks)))
    for task in tasks:
        dataset_get = MoleculeDataset(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "../../raw_data")) + "/" + dataset + "/new/" + str(task + 1),
                                  dataset = dataset)
        load_data[task] = dataset_get
    return load_data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root


        super(MoleculeDataset, self).__init__()
        if not os.path.exists(self.root + "/new/data.pdparams"):
            self.read()
        self.smiles_list = paddle.load(self.root + "/new/smiles.pdparams")[1]
        data_list = paddle.load(self.root + "/new/data.pdparams")[1]
        self.data_list = [G.Graph(i['edge_index'],i['x'].shape[0], 
                        {'feature':i['x']}, {'feature':i['edge_attr']}) for i in data_list]
        for i in range(len(self.data_list)):
            self.data_list[i].y = data_list[i]['y']
        x = 0
        
    def read(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_dataset(self.root + "/raw/" + self.dataset + ".json")
            
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data['id'] = np.array([i])
                data['y'] = np.array(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_dataset(self.root + "/raw/" + self.dataset + ".json")
            
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data['id'] = np.array([i])
                data['y'] = np.array(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_dataset(self.root + "/raw/" + self.dataset + ".json")
            
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data['id'] = np.array([i])
                data['y'] = np.array(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_dataset(self.root + "/raw/" + self.dataset + ".json")
        
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    data['id'] = np.array([i])
                    data['y'] = np.array(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        else:
            raise ValueError('Invalid dataset name')

        self.smiles_list = data_smiles_list

        paddle.save({1:data_list}, self.root + "/new/data.pdparams")
        paddle.save({1:self.smiles_list}, self.root + "/new/smiles.pdparams")

    def get_train_data_sample(self, task = 0 , n_shot_train = 10, n_query = 16):
        s_data, q_data = self.sample_meta_datasets(self.dataset, task, n_shot_train, n_query)
        s_data_y = np.stack([i.y[0] for i in s_data.data_list])
        q_data_y = np.stack([i.y[0] for i in q_data.data_list])
        adapt_data = {'s_data': G.Graph.batch(s_data.data_list), 's_label': paddle.to_tensor(s_data_y), 'q_data': G.Graph.batch(q_data.data_list), 'q_label': paddle.to_tensor(q_data_y)}
        eval_data = { }
        return adapt_data, eval_data

    def get_test_data_sample(self, task = 0 , n_shot_test = 10, n_query = 16, update_step_test = 1):

        s_data, q_data, q_data_adapt = self.sample_test_datasets(self.dataset, task, n_shot_test, n_query, update_step_test)
            
        s_data_y = np.stack([i.y[0] for i in s_data.data_list])

        q_loader = q_data.get_data_loader(batch_size=n_query, shuffle=True, num_workers=1)
        q_loader_adapt = q_data_adapt.get_data_loader(batch_size=n_query, shuffle=True, num_workers=1)

        adapt_data = {'s_data': G.Graph.batch(s_data.data_list), 's_label': paddle.to_tensor(s_data_y), 'data_loader': q_loader_adapt}
        eval_data = {'s_data': G.Graph.batch(s_data.data_list), 's_label': paddle.to_tensor(s_data_y), 'data_loader': q_loader}

        return adapt_data, eval_data

    def sample_meta_datasets(self, dataset, task, n_shot, n_query):
        distri_list = obtain_distr_list(dataset)
        thresh = distri_list[task][0]

        neg_sample = sample_inds(range(0, thresh), n_shot)
        pos_sample = sample_inds(range(thresh, len(self)), n_shot)

        s_list = neg_sample + pos_sample

        l = [i for i in range(0, len(self)) if i not in s_list]
        random.shuffle(l)
        q_list = sample_inds(l, n_query)

        s_data = self[s_list]
        q_data = self[q_list]

        return s_data, q_data


    def sample_test_datasets(self, dataset, task, n_shot, n_query, update_step=1):
        distri_list = obtain_distr_list(dataset)
        thresh = distri_list[task][0]

        neg_sample = sample_inds(range(0, thresh), n_shot)
        pos_sample = sample_inds(range(thresh, len(self)), n_shot)

        s_list = neg_sample + pos_sample

        q_list = [i for i in range(0, len(self)) if i not in s_list]

        s_data = self[s_list]
        q_data = self[q_list]

        q_sample = sample_inds(q_list, update_step * n_query)
        q_data_adapt = self[q_sample]

        return s_data, q_data, q_data_adapt


def _load_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    with open(input_path) as json_file:
        binary_list = json.load(json_file)

    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels



def obatin_train_test_tasks(dataset):
    tox21_train_tasks = list(range(9))
    tox21_test_tasks = list(range(9, 12))
    sider_train_tasks = list(range(21))
    sider_test_tasks = list(range(21, 27))
    toxcast_drop_tasks = [343, 348, 349, 352, 354, 355, 356, 357, 358, 360, 361, 362, 364, 367, 368, 369, 370, 371, 372,
                          373, 374, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 387, 388, 389, 390, 391, 392, 393,
                          394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 406, 408, 409, 410, 411, 412, 413, 414,
                          415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 426, 428, 429, 430, 431, 432, 433, 434, 435,
                          436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 474, 475, 477,
                          480, 481, 482, 483]
    toxcast_train_tasks = [x for x in list(range(450)) if x not in toxcast_drop_tasks]
    toxcast_test_tasks = [x for x in list(range(450, 617)) if x not in toxcast_drop_tasks]
    muv_train_tasks = list(range(12))
    muv_test_tasks = list(range(12, 17))
    if dataset == "sider":
        return sider_train_tasks, sider_test_tasks
    elif dataset == "tox21":
        return tox21_train_tasks, tox21_test_tasks
    elif dataset == "muv":
        return muv_train_tasks, muv_test_tasks
    elif dataset == "toxcast":
        return toxcast_train_tasks, toxcast_test_tasks
    else:
        return None, None


def obtain_distr_list(dataset):
    if dataset == "sider":
        return [[684, 743], [431, 996], [1405, 22], [551, 876], [276, 1151], [430, 997], [129, 1298], [1176, 251],
                [403, 1024], [700, 727], [1051, 376], [135, 1292], [1104, 323], [1214, 213], [319, 1108], [542, 885],
                [109, 1318], [1174, 253], [421, 1006], [367, 1060], [411, 1016], [516, 911], [1302, 125], [768, 659],
                [439, 988], [123, 1304], [481, 946]]
    elif dataset == "tox21":
        return [[6956, 309], [6521, 237], [5781, 768], [5521, 300], [5400, 793], [6605, 350], [6264, 186], [4890, 942],
                [6808, 264], [6095, 372], [4892, 918], [6351, 423]]
    elif dataset == "muv":
        return [[14814, 27], [14705, 29], [14698, 30], [14593, 30], [14873, 29], [14572, 29], [14614, 30], [14383, 28],
                [14807, 29], [14654, 28], [14662, 29], [14615, 29], [14637, 30], [14681, 30], [14622, 29], [14745, 29],
                [14722, 24]]
    elif dataset == "toxcast":
        return [[1288, 443], [1434, 297], [859, 175], [991, 43], [791, 243], [734, 300], [589, 445], [972, 62],
                [943, 91], [955, 64], [906, 113], [903, 131], [1005, 29], [925, 94], [942, 77], [278, 25], [846, 188],
                [884, 135], [817, 217], [736, 283], [974, 60], [989, 45], [1013, 21], [792, 242], [784, 250], [283, 20],
                [962, 72], [930, 104], [837, 197], [823, 211], [259, 44], [254, 49], [249, 54], [264, 39], [248, 55],
                [245, 58], [244, 59], [248, 55], [248, 55], [259, 44], [260, 43], [280, 23], [271, 32], [283, 20],
                [281, 22], [3322, 90], [2789, 623], [3187, 225], [3368, 44], [3389, 23], [3371, 41], [3173, 239],
                [2966, 446], [3310, 102], [3025, 387], [3330, 82], [3375, 37], [3268, 144], [2867, 545], [3352, 60],
                [3051, 361], [3330, 82], [3163, 249], [2708, 704], [3346, 66], [3267, 145], [3086, 326], [3277, 135],
                [3379, 33], [2905, 507], [3196, 216], [2536, 876], [3375, 37], [3377, 35], [3391, 21], [2681, 731],
                [3071, 341], [3329, 83], [3185, 227], [3384, 28], [3310, 102], [3342, 70], [3339, 73], [3246, 166],
                [3359, 53], [3060, 84], [3300, 112], [3365, 47], [3369, 43], [3220, 192], [3261, 151], [3356, 56],
                [3384, 28], [3352, 60], [3183, 229], [3026, 386], [3377, 35], [3362, 50], [3282, 130], [3345, 67],
                [3356, 56], [2988, 424], [3068, 344], [3320, 92], [2938, 474], [3387, 25], [3348, 64], [3310, 102],
                [3386, 26], [3336, 76], [2688, 724], [3389, 23], [3327, 85], [3346, 66], [3375, 37], [3353, 59],
                [3359, 53], [3352, 60], [3382, 30], [3390, 22], [3288, 124], [3360, 52], [3361, 51], [3222, 190],
                [3354, 58], [3136, 276], [3132, 280], [3273, 139], [3255, 157], [3308, 104], [3356, 56], [2118, 1294],
                [3381, 31], [3198, 214], [3375, 37], [2858, 554], [3381, 31], [3018, 394], [3374, 38], [3173, 239],
                [3340, 72], [2479, 933], [3319, 93], [2879, 533], [3079, 333], [1767, 1645], [3389, 23], [2441, 971],
                [2952, 460], [3165, 247], [3334, 78], [3274, 138], [3387, 25], [3346, 66], [3355, 57], [3382, 30],
                [3034, 378], [3357, 55], [3382, 30], [3376, 36], [3388, 24], [3303, 109], [3313, 99], [2885, 527],
                [3368, 44], [2879, 533], [3312, 100], [3370, 42], [3379, 33], [3187, 225], [3377, 35], [3060, 352],
                [3346, 66], [3289, 123], [3383, 29], [3175, 237], [2949, 463], [3371, 41], [3288, 124], [3274, 138],
                [3373, 39], [3300, 112], [3392, 20], [2481, 931], [3387, 25], [3362, 50], [2640, 253], [3382, 30],
                [2951, 461], [3072, 340], [3324, 88], [1038, 401], [887, 552], [1234, 205], [1109, 330], [1058, 381],
                [1248, 191], [839, 600], [962, 477], [1377, 62], [1300, 139], [1118, 321], [1379, 60], [1085, 354],
                [1003, 436], [1021, 418], [1000, 439], [1056, 383], [1037, 402], [1417, 22], [1047, 392], [1014, 425],
                [1223, 216], [1092, 347], [1418, 21], [1052, 387], [1168, 271], [1055, 384], [1247, 192], [1248, 191],
                [1308, 131], [1145, 294], [1230, 209], [1209, 230], [1193, 246], [1238, 201], [1400, 39], [1199, 240],
                [1148, 291], [1229, 210], [1391, 48], [1293, 146], [1275, 164], [1356, 83], [1225, 214], [1407, 32],
                [1174, 265], [1417, 22], [1281, 158], [995, 444], [1411, 28], [1261, 178], [1404, 35], [1187, 252],
                [1365, 74], [1187, 252], [1218, 221], [1154, 285], [1401, 38], [1192, 247], [1416, 23], [1212, 227],
                [1141, 298], [1115, 324], [1415, 24], [1180, 259], [1414, 25], [1049, 390], [1205, 234], [1145, 294],
                [1114, 325], [1172, 267], [1015, 424], [1132, 307], [1417, 22], [1130, 309], [1418, 21], [1119, 320],
                [1409, 30], [1114, 325], [1043, 396], [1141, 298], [1343, 96], [1066, 373], [1145, 294], [1364, 75],
                [1202, 237], [1384, 55], [999, 440], [997, 442], [996, 443], [1036, 403], [1030, 409], [1393, 46],
                [1092, 347], [1396, 43], [1091, 348], [1206, 233], [1117, 322], [1364, 75], [876, 563], [1077, 362],
                [1001, 438], [1341, 98], [1376, 63], [1246, 193], [1019, 420], [1022, 417], [1143, 296], [1173, 266],
                [1376, 63], [1043, 396], [815, 624], [1107, 332], [1170, 269], [1027, 412], [298, 204], [315, 187],
                [367, 135], [363, 133], [421, 75], [464, 36], [422, 78], [453, 49], [430, 72], [406, 56], [349, 151],
                [381, 119], [346, 123], [355, 145], [281, 19], [277, 23], [175, 121], [185, 110], [198, 104],
                [168, 134], [147, 153], [218, 84], [128, 171], [138, 162], [120, 182], [178, 114], [178, 116],
                [251, 45], [271, 29], [274, 25], [258, 42], [251, 51], [234, 66], [172, 201], [274, 99], [142, 176],
                [66, 307], [22, 31], [220, 483], [168, 71], [105, 70], [39, 134], [86, 27], [35, 101], [76, 301],
                [38, 187], [37, 80], [75, 85], [49, 28], [23, 31], [74, 68], [90, 21], [72, 23], [80, 90], [42, 37],
                [99, 31], [43, 60], [80, 81], [59, 54], [135, 30], [196, 24], [55, 44], [37, 45], [55, 35], [70, 34],
                [71, 22], [58, 39], [53, 26], [80, 58], [112, 68], [92, 20], [65, 31], [63, 24], [54, 25], [51, 24],
                [76, 32], [29, 38], [88, 26], [69, 29], [42, 21], [130, 24], [56, 84], [42, 61], [50, 49], [56, 39],
                [31, 84], [42, 64], [57, 71], [76, 56], [52, 54], [74, 38], [23, 32], [50, 85], [43, 77], [36, 53],
                [37, 28], [45, 57], [54, 92], [62, 47], [66, 89], [35, 65], [40, 120], [46, 21], [34, 84], [20, 66],
                [30, 61], [31, 81], [38, 57], [38, 40], [61, 25], [32, 98], [53, 72], [21, 57], [33, 57], [49, 22],
                [26, 57], [43, 75], [32, 70], [49, 81], [85, 79], [47, 60], [75, 114], [34, 61], [41, 70], [43, 29],
                [44, 48], [41, 51], [40, 53], [25, 53], [42, 23], [66, 46], [57, 28], [57, 72], [57, 65], [36, 34],
                [912, 30], [25, 30], [41, 58], [26, 77], [51, 40], [31, 71], [35, 54], [41, 117], [42, 25], [43, 23],
                [24, 26], [37, 25], [53, 31], [132, 216], [115, 218], [924, 130], [108, 77], [98, 206], [116, 112],
                [194, 83], [896, 232], [131, 33], [197, 60], [119, 226], [304, 72], [600, 180], [194, 87], [403, 111],
                [230, 30], [144, 22], [168, 55], [740, 188], [139, 131], [76, 21], [38, 107], [50, 123], [26, 51],
                [49, 194], [68, 161], [64, 39], [39, 39], [52, 61], [52, 50], [1627, 145], [1621, 119], [1525, 208],
                [1614, 134], [1566, 108], [1536, 219], [1469, 199], [1536, 208], [1488, 178], [1588, 183], [1610, 148],
                [1419, 316], [1541, 141], [1553, 205], [1648, 89], [6818, 369], [6549, 638], [5573, 1614], [7101, 86],
                [7907, 24], [7543, 388], [7521, 410], [7889, 42], [7727, 204], [6756, 1175], [7332, 599], [7546, 385],
                [7081, 850], [6019, 1168], [7123, 808], [6658, 1273], [7881, 50], [7879, 52], [7880, 51], [7907, 24],
                [7882, 49], [7908, 23], [5137, 134], [7634, 297], [7463, 468], [7461, 470], [7631, 300], [7675, 256],
                [6901, 1030], [7731, 200], [6675, 1256], [7216, 715], [7093, 94], [7077, 110], [6854, 333], [6954, 233],
                [6827, 360], [6900, 287], [7003, 184], [6980, 207], [6231, 956], [6854, 333], [7601, 330], [7702, 229],
                [7429, 502], [7394, 537], [7473, 458], [7531, 400], [6892, 295], [6814, 373], [6577, 610], [7018, 169],
                [4415, 856], [5150, 121], [4968, 303], [6891, 296], [7083, 104], [6944, 243], [6738, 449], [6535, 652],
                [7067, 120], [7167, 20], [7000, 187], [6993, 194], [6746, 441], [7154, 33], [7907, 24], [7697, 234],
                [7577, 354], [7158, 29], [6639, 548], [7028, 159], [7893, 38], [6170, 1761], [6557, 630], [7010, 177],
                [7123, 64], [7117, 70], [6873, 314], [6909, 278], [7361, 570], [7460, 471], [7117, 814], [7537, 394],
                [7404, 527], [7130, 801], [6900, 1031], [7133, 798], [7252, 679], [7220, 711], [6972, 959], [7498, 433],
                [7273, 658], [7360, 571], [6982, 949], [7260, 671], [7577, 354], [7288, 643], [7048, 883], [7603, 328],
                [907, 114], [838, 196], [958, 63], [971, 50], [1001, 20], [940, 81], [906, 115], [909, 125], [977, 44],
                [900, 121], [964, 57], [974, 47], [910, 111], [981, 40], [986, 35], [962, 59], [938, 83], [892, 129]]
    elif dataset=='tox21train':
        return [[5586, 248], [5323, 190], [4718, 590], [4585, 210], [4444, 648], [5347, 297], [5174, 134], [4177, 717], [5542, 195], [5054, 282], [4060, 711], [5223, 278]]
    elif dataset == 'tox21valid':
        return [[693, 29], [602, 25], [531, 89], [469, 46], [483, 68], [632, 30], [546, 30], [360, 107], [634, 33], [520, 44], [413, 111], [574, 75]]
    elif dataset == 'tox21test':
        return [[677, 32], [596, 22], [532, 89], [467, 44], [473, 77], [626, 23], [544, 22], [353, 118], [632, 36], [521, 46], [419, 96], [554, 70]]
    else:
        return None


def sample_inds(data, size):
    len_data = len(data)
    if len_data >= size:
        return random.sample(data, size)
    else:
        return random.sample(data, len_data) + sample_inds(data, size - len_data)


def sample_meta_datasets(data, dataset, task, n_shot, n_query):
    distri_list = obtain_distr_list(dataset)
    thresh = distri_list[task][0]

    neg_sample = sample_inds(range(0, thresh), n_shot)
    pos_sample = sample_inds(range(thresh, len(data)), n_shot)

    s_list = neg_sample + pos_sample

    l = [i for i in range(0, len(data)) if i not in s_list]
    random.shuffle(l)
    q_list = sample_inds(l, n_query)

    s_data = data[s_list]
    q_data = data[q_list]

    return s_data, q_data


def sample_test_datasets(data, dataset, task, n_shot, n_query, update_step=1):
    distri_list = obtain_distr_list(dataset)
    thresh = distri_list[task][0]

    neg_sample = sample_inds(range(0, thresh), n_shot)
    pos_sample = sample_inds(range(thresh, len(data)), n_shot)

    s_list = neg_sample + pos_sample

    q_list = [i for i in range(0, len(data)) if i not in s_list]

    s_data = data[s_list]
    q_data = data[q_list]

    q_sample = sample_inds(q_list, update_step * n_query)
    q_data_adapt = data[q_sample]

    return s_data, q_data, q_data_adapt
