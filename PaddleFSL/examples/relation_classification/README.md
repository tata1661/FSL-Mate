# Relation Classification Tasks

For relation classification tasks, we currently provide implementation of two algorithms: ProtoNet and Siamese.

The performance is evaluated on [FewRel1.0](https://github.com/thunlp/FewRel), which is the benchmark dataset for few-shot relation classification.

For all the setting, configurations and hyper-parameters of every task, please see [PaddleFSL-Tasks](https://github.com/JeremyZhao1998/PaddleFSL-Tasks).

## ProtoNet

Prototypical Networks for Few-shot Learning. ([ProtoNet](https://arxiv.org/pdf/1703.05175.pdf))

|  Dataset  | Word embedding | Model  | Ways | Shots | Original paper |                        Other reports                         |     Ours     |
| :-------: | :------------: | :----: | :--: | :---: | :------------: | :----------------------------------------------------------: | :----------: |
| FewRel1.0 |    GloVe 50    | Conv1D |  5   |   1   |       --       | 69.20 ± 0.20([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 70.18 ± 0.35 |
| FewRel1.0 |    GloVe 50    | Conv1D |  5   |   5   |       --       | 84.79 ± 0.16([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 83.13 ± 0.42 |
| FewRel1.0 |    GloVe 50    | Conv1D |  10  |   1   |       --       | 56.44 ± 0.22([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 56.81 ± 0.15 |
| FewRel1.0 |    GloVe 50    | Conv1D |  10  |   5   |       --       | 75.55 ± 0.19([FewRel1.0](https://aclanthology.org/D18-1514.pdf)) | 71.76 ± 0.31 |

## Siamese

Siamese Nerual Network for One Shot Image Recognation. ([Siamese](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf))

|  Dataset  | Word embedding | Model  | Ways | Shots | Original paper |     Ours     |
| :-------: | :------------: | :----: | :--: | :---: | :------------: | :----------: |
| FewRel1.0 |    GloVe 50    | Conv1D |  5   |   1   |       --       | 72.07 ± 0.38 |
| FewRel1.0 |    GloVe 50    | Conv1D |  5   |   5   |       --       | 80.05 ± 0.28 |
| FewRel1.0 |    GloVe 50    | Conv1D |  10  |   1   |       --       | 59.73 ± 0.25 |
| FewRel1.0 |    GloVe 50    | Conv1D |  10  |   5   |       --       | 69.41 ± 0.20 |

