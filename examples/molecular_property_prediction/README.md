# Molecular Property Prediction Tasks

Here, we provide examples of applying PaddleFSL to few-shot molecular property prediction tasks. 

## Environment

To run the examples, please first install paddlehelix-1.0.1 or later versions as follows: 
```bash
pip install rdkit-pypi
pip install pgl
pip install paddlehelix
```
Or see [here](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/installation_guide.md) for details. 

## Datasets

We evaluate the performance on 4 benchmark datasets，including Tox21, SIDER, MUV and ToxCast provided by [MoleculeNet](https://pubs.rsc.org/en/content/articlepdf/2018/sc/c7sc02664a), which can be accessed as described in [raw_data/README.md](../../raw_data/README.md).

## Pretrained Model

We provide pretrained GIN model for molecular property prediction tasks, please download the pretrained model from [here](https://drive.google.com/file/d/1vKoEYBlCc6gviX8Tq0G8IgTuCKTrracT/view)
and put it under [molecular_property_prediction/utils](./utils/).

## Results
We provide results of using MAML [1] and PAR [2] below. 

### [MAML](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf?source=post_page---------------------------) 

|   Dataset    | Backbone | Way  |  Shot |      AUC-ROC      |
| :----------: | :---: | :--: | :---: | :---------------: |
|    Tox21     |  GIN  |  2   |   10  |   81.24 ± 0.58    |
|    Tox21     |  GIN  |  2   |   1   |   80.12 ± 0.52    |
|    SIDER     |  GIN  |  2   |   10  |   73.12 ± 1.03    |
|    SIDER     |  GIN  |  2   |   1   |   70.08 ± 0.76    |
|     MUV      |  GIN  |  2   |   10  |   67.98 ± 4.64    |
|     MUV      |  GIN  |  2   |   1   |   67.43 ± 1.89    |
|   ToxCast    |  GIN  |  2   |   10  |   71.56 ± 0.32    |
|   ToxCast    |  GIN  |  2   |   1   |   70.23 ± 1.06    |

### [PAR]((https://proceedings.neurips.cc/paper/2021/file/91bc333f6967019ac47b49ca0f2fa757-Paper.pdf))

|   Dataset    | Backbone  | Way  | Shot  |      AUC-ROC      |
| :----------: | :---: | :--: | :---: | :---------------: |
|    Tox21     |  GIN  |  2   |   10  |   82.42 ± 1.05    |
|    Tox21     |  GIN  |  2   |   1   |   81.76 ± 0.12    |
|    SIDER     |  GIN  |  2   |   10  |   75.46 ± 0.89    |
|    SIDER     |  GIN  |  2   |   1   |   72.94 ± 0.77    |
|     MUV      |  GIN  |  2   |   10  |   68.13 ± 0.51    |
|     MUV      |  GIN  |  2   |   1   |   70.47 ± 3.40    |
|   ToxCast    |  GIN  |  2   |   10  |   73.00 ± 0.35    |
|   ToxCast    |  GIN  |  2   |   1   |   72.50 ± 0.98    |

### Performance Obtained on Different Platforms

We also provide performance of MAML and PAR obtained on Tox21 when they are deployed on different platforms. 

|   Dataset    | Backbone | Way | Shot | Platform|Method|      AUC-ROC      |
| :----------: | :---: | :--: | :---: | :---: | :---:| :---------------: |
|    Tox21     |  GIN  |  2   |   10  | CPU   | MAML |   81.35 ± 0.73    |
|    Tox21     |  GIN  |  2   |   10  | GPU   | MAML |   81.24 ± 0.58    |
|    Tox21     |  GIN  |  2   |   10  |Cluster| MAML |   81.57 ± 1.03    |
|    Tox21     |  GIN  |  2   |   1   | CPU   | MAML |   79.88 ± 1.03    |
|    Tox21     |  GIN  |  2   |   1   | GPU   | MAML |   80.12 ± 0.52    |
|    Tox21     |  GIN  |  2   |   1   |Cluster| MAML |   79.93 ± 0.15    |
|    Tox21     |  GIN  |  2   |   10  | CPU   |  PAR |   83.01 ± 0.32    |
|    Tox21     |  GIN  |  2   |   10  | GPU   |  PAR |   82.42 ± 1.05    |
|    Tox21     |  GIN  |  2   |   10  |Cluster|  PAR |   82.71 ± 0.48    |
|    Tox21     |  GIN  |  2   |   1   | CPU   |  PAR |   81.78 ± 0.45    |
|    Tox21     |  GIN  |  2   |   1   | GPU   |  PAR |   81.76 ± 0.12    |
|    Tox21     |  GIN  |  2   |   1   |Cluster|  PAR |   82.06 ± 0.33    |


## References

1. *Model-agnostic meta-learning for fast adaptation of deep networks,* in ICML, 2017.
C. Finn, P. Abbeel, and S. Levine.

1. *Property-aware relation networks for few-shot molecular property prediction,* in NeurIPS, 2021.
Y. Wang, A. Abuduweili, Q. Yao, and D. Dou.