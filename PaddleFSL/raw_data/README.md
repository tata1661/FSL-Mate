# Raw Data

PaddleFSL has been evaluated on the following few-shot benchmark datasets. 

Users can download the datasets from the URLs provided below, put the dataset into the respective folder (default folder used by PaddleFSL is [raw_data](../raw_data/)), pre-process the data using scripts in [paddlefsl.dataset](../paddlefsl/datasets/), and 
run the example scripts in [examples](../examples/). 

## Computer Vision Datasets

- **Omniglot** ([B. M. Lake et al., 2015](http://www.sciencemag.org/content/350/6266/1332.short)), which can be
downloaded from [link](https://github.com/brendenlake/omniglot/raw/master/python/).

- **Mini-ImageNet** ([O. Vinyals et al., 2016](https://arxiv.org/abs/1606.04080)), which can be
downloaded from [link](https://drive.google.com/file/d/1LLUjwSUpWGSWizl3JZxd08V30_dIaRBx/view).

- **Tiered-ImageNet** ([M. Ren et al., 2018](https://arxiv.org/abs/1803.00676)), which can be
downloaded from [link](https://drive.google.com/file/d/1fQ6lI5pCnOEt9MHWdqFN1cdSU2SbMKzx/view).

- **CIFAR-FS** ([L. Bertinetto et al., 2018](https://arxiv.org/abs/1805.08136)), which can be
downloaded from [link](https://drive.google.com/file/d/1nN1u2ZeD0L90uG5Y_Ml6uQR6z-o6aBLL/view).

- **FC100** ([B. N. Oreshkin et al., 2018](https://arxiv.org/abs/1805.10123)), which can be
downloaded from [link](https://drive.google.com/file/d/18SPp-RLOL-nxxoHtkU8-n8OspDjMfhAH/view). 

- **CUB** ([W.-Y. Chen et al., 2019](https://arxiv.org/abs/1904.04232)), which can be
downloaded from [link](https://drive.google.com/file/d/1EiKOk6LAqlYwDJzUQRDUjGMsvUGBT1U8/view).

## Natural Language Processing Datasets

- **FewRel** ([X. Han et al., 2018](https://aclanthology.org/D18-1514.pdf)), which can be
downloaded from [link](https://github.com/thunlp/FewRel/tree/master/data).

- **FewGLUE** ([T. Schick et al., 2021](https://arxiv.org/abs/2001.07676)), which can be
downloaded from [link](https://github.com/THUDM/P-tuning/tree/main/FewGLUE_32dev).


- **FewCLUE** ([L. Xu et al., 2021](https://arxiv.org/abs/2107.07498)), which can be
downloaded from [link](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) or directly imported from paddlenlp.datasets. 

## Bioinformatics Datasets

- **Tox21, SIDER, MUV and ToxCast**
are provided by **MoleculeNet** ([Z. Wu et al., 2018](https://pubs.rsc.org/en/content/articlepdf/2018/sc/c7sc02664a)), which can be
downloaded from [link](https://drive.google.com/file/d/1K3c4iCFHEKUuDVSGBtBYr8EOegvIJulO/view).
Once downloaded, please unzip the file and put the resultant four folders corresponding to the four dataset into `<path to>/PaddleFSL/raw_data`.
