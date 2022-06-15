# Relation Classification Tasks

Here, we provide examples of applying PaddleFSL to few-shot relation classification tasks. 

## Datasets

We evaluate the performance on FewRel (1.0 version), which can be accessed as described in  [raw_data/README.md](../../raw_data/README.md).

## Results

We provide results of using ProtoNet [1] and Siamese [2] below. 
The exact model configuration and pretrained models can be downloaded from [here](https://drive.google.com/file/d/1Prig4u1gHZT7wA7UxJ5qaXUMeDaweXKA/view?usp=sharing), which can reproduce these results.


### [ProtoNet](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)

|  Dataset  | Word embedding | Model  | Way  | Shot  | Original paper |                        Other reports                         |     Ours     |
| :-------: | :------------: | :----: | :--: | :---: | :------------: | :----------------------------------------------------------: | :----------: |
| FewRel    |GloVE(dimension=50)|  CNN   |  5   |   1   |       --       | 69.20 ± 0.20([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 70.18 ± 0.35 |
| FewRel    |GloVE(dimension=50)|  CNN   |  5   |   5   |       --       | 84.79 ± 0.16([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 83.13 ± 0.42 |
| FewRel    |GloVE(dimension=50)|  CNN   |  10  |   1   |       --       | 56.44 ± 0.22([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 56.81 ± 0.15 |
| FewRel    |GloVE(dimension=50)|  CNN   |  10  |   5   |       --       | 75.55 ± 0.19([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 71.76 ± 0.31 |

### [Siamese](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)

|  Dataset  | Word embedding    |  Model | Way  | Shot  | Original paper |     Ours     |
| :-------: | :---------------: | :----: | :--: | :---: | :------------: | :----------: |
| FewRel    |GloVE(dimension=50)|  CNN   |  5   |   1   |       --       | 72.07 ± 0.38 |
| FewRel    |GloVE(dimension=50)|  CNN   |  5   |   5   |       --       | 80.05 ± 0.28 |
| FewRel    |GloVE(dimension=50)|  CNN   |  10  |   1   |       --       | 59.73 ± 0.25 |
| FewRel    |GloVE(dimension=50)|  CNN   |  10  |   5   |       --       | 69.41 ± 0.20 |

### [GNN](https://arxiv.org/pdf/1711.04043.pdf)

|  Dataset  | Word embedding    |  Model | Way  | Shot  | Original paper |     Ours     |
| :-------: | :---------------: | :----: | :--: | :---: | :------------: | :----------: |
| FewRel    |GloVE(dimension=50)|  CNN   |  5   |   1   |  66.2 ± 0.75   | 67.2 ± 0.60 |
| FewRel    |GloVE(dimension=50)|  CNN   |  5   |   5   |  81.3 ± 0.62   | 82.1 ± 0.48 |

## References

1. *Prototypical networks for few-shot learning,* in NeurIPS, 2017.
J. Snell, K. Swersky, and R. S. Zemel.

1. *Siamese neural networks for one-shot image recognition,* ICML deep learning workshop, 2015.
G. Koch, R. Zemel, and R. Salakhutdinov.
   
1. *Few-Shot Learning with Graph Neural Networks,* in ICLR, 2018.
V. Garcia and J. Bruna.