# Image Classification Tasks

For image classification tasks, we currently provide implementation of four algorithms: MAML, ANIL, ProtoNet and RelationNet.

For all the setting, configurations and hyper-parameters of every task, please see [PaddleFSL-Tasks](https://github.com/JeremyZhao1998/PaddleFSL-Tasks).

## MAML

Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks ([MAML](https://arxiv.org/pdf/1703.03400.pdf)).

Currently we only support first order approximate version of MAML, because paddle do not support second order gradients of several operations.

|     Dataset     | Model | Ways | Shots | Original paper |                        Other reports                         | Ours(approximate) |
| :-------------: | :---: | :--: | :---: | :------------: | :----------------------------------------------------------: | :---------------: |
|    Omniglot     |  MLP  |  5   |   1   |   89.7 ± 1.1   |       88.9<br>([learn2learn](http://learn2learn.net/))       |   88.88 ± 2.99    |
|    Omniglot     |  MLP  |  5   |   5   |   97.5 ± 0.6   |                              --                              |   97.50 ± 0.47    |
|    Omniglot     | Conv  |  5   |   1   |   98.7 ± 0.4   |      99.1<br/>([learn2learn](http://learn2learn.net/))       |   97.13 ± 1.25    |
|    Omniglot     | Conv  |  5   |   5   |   99.9 ± 0.1   | 99.9 ± 0.1<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) |   99.23 ± 0.40    |
|  Mini-ImageNet  | Conv  |  5   |   1   |  48.70 ± 1.84  |      48.3<br/>([learn2learn](http://learn2learn.net/))       |   49.81 ± 1.78    |
|  Mini-ImageNet  | Conv  |  5   |   5   |  63.11 ± 0.92  |      65.4<br/>([learn2learn](http://learn2learn.net/))       |   64.21 ± 1.33    |
|     CifarFS     | Conv  |  5   |   1   |       --       | 58.9 ± 1.9<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) |   57.06 ± 3.83    |
|     CifarFS     | Conv  |  5   |   5   |       --       |      76.6<br/>([learn2learn](http://learn2learn.net/))       |   72.24 ± 1.71    |
|      FC100      | Conv  |  5   |   1   |       --       |                              --                              |   37.63 ± 2.23    |
|      FC100      | Conv  |  5   |   5   |       --       |      49.0<br/>([learn2learn](http://learn2learn.net/))       |   49.14 ± 1.58    |
|      CubFS      | Conv  |  5   |   1   |       --       | 54.73 ± 0.97<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)﻿) |   53.31 ± 1.77    |
|      CubFS      | Conv  |  5   |   5   |       --       | 75.75 ± 0.76<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)) |   69.88 ± 1.47    |
| Tiered-ImageNet | Conv  |  5   |   1   |       --       |                              --                              |   49.00 ± 3.26    |
| Tiered-ImageNet | Conv  |  5   |   5   |       --       |                              --                              |   67.56 ± 1.80    |

## ANIL

Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML ([ANIL](https://arxiv.org/pdf/1909.09157.pdf)).

Currently we only support first order approximate version of ANIL, because paddle do not support second order gradients of several operations.

|     Dataset     | Model | Ways | Shots | Original paper |                   Other reports                   | Ours(approximate) |
| :-------------: | :---: | :--: | :---: | :------------: | :-----------------------------------------------: | :---------------: |
|    Omniglot     | Conv  |  5   |   1   |       --       |                        --                         |   96.06 ± 1.00    |
|    Omniglot     | Conv  |  5   |   5   |       --       |                        --                         |   98.74 ± 0.48    |
|  Mini-ImageNet  | Conv  |  5   |   1   |   46.7 ± 0.4   |                        --                         |   48.31 ± 2.83    |
|  Mini-ImageNet  | Conv  |  5   |   5   |   61.5 ± 0.5   |                        --                         |   62.38 ± 1.96    |
|     CifarFS     | Conv  |  5   |   1   |       --       |                        --                         |   56.19 ± 3.39    |
|     CifarFS     | Conv  |  5   |   5   |       --       | 68.3<br/>([learn2learn](http://learn2learn.net/)) |   68.60 ± 1.25    |
|      FC100      | Conv  |  5   |   1   |       --       |                        --                         |   40.69 ± 3.32    |
|      FC100      | Conv  |  5   |   5   |       --       | 47.6<br/>([learn2learn](http://learn2learn.net/)) |   48.01 ± 1.22    |
|      CubFS      | Conv  |  5   |   1   |       --       |                        --                         |   53.25 ± 2.18    |
|      CubFS      | Conv  |  5   |   5   |       --       |                        --                         |   69.09 ± 1.12    |
| Tiered-ImageNet | Conv  |  5   |   1   |       --       |                        --                         |   48.38 ± 2.46    |
| Tiered-ImageNet | Conv  |  5   |   5   |       --       |                        --                         |   65.69 ± 2.89    |

## ProtoNet

Prototypical Networks for Few-shot Learning ([ProtoNet](https://arxiv.org/pdf/1703.05175.pdf)).

|    Dataset    | Model | Ways | Shots | Original paper |                        Other reports                         |     Ours     |
| :-----------: | :---: | :--: | :---: | :------------: | :----------------------------------------------------------: | :----------: |
|   Omniglot    | Conv  |  5   |   1   |      98.8      | 98.5 ± 0.2<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 98.27 ± 0.13 |
|   Omniglot    | Conv  |  5   |   5   |      99.7      | 99.5 ± 0.1<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 99.37 ± 0.05 |
| Mini-ImageNet | Conv  |  5   |   1   |  49.42 ± 0.78  |      49.1<br/>([learn2learn](http://learn2learn.net/))       | 48.85 ± 0.42 |
| Mini-ImageNet | Conv  |  5   |   5   |  68.20 ± 0.66  |      66.5<br/>([learn2learn](http://learn2learn.net/))       | 66.87 ± 0.25 |
|    CifarFS    | Conv  |  5   |   1   |       --       | 55.5 ± 0.7<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 55.49 ± 0.21 |
|    CifarFS    | Conv  |  5   |   5   |       --       | 72.0 ± 0.6<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 72.10 ± 0.13 |
|     FC100     | Conv  |  5   |   1   |       --       |                              --                              | 35.90 ± 0.24 |
|     FC100     | Conv  |  5   |   5   |       --       | 51.1 ± 0.2<br/>([TADAM](https://arxiv.org/pdf/1805.10123.pdf)﻿) | 49.26 ± 0.25 |
|     CubFS     | Conv  |  5   |   1   |       --       | 50.46 ± 0.88<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)) | 51.31 ± 0.48 |
|     CubFS     | Conv  |  5   |   5   |       --       | 76.39 ± 0.64<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)) | 70.14 ± 0.19 |

## RelationNet

Learning to Compare: Relation Network for Few-Shot Learning ([RelationNet](https://arxiv.org/pdf/1711.06025v2.pdf)).

|    Dataset    | Model | Ways | Shots | Original paper |                        Other reports                         |     Ours     |
| :-----------: | :---: | :--: | :---: | :------------: | :----------------------------------------------------------: | :----------: |
|   Omniglot    | Conv  |  5   |   1   |   99.6 ± 0.2   |                              --                              | 98.02 ± 0.09 |
|   Omniglot    | Conv  |  5   |   5   |   99.8 ± 0.1   |                              --                              | 99.25 ± 0.05 |
| Mini-ImageNet | Conv  |  5   |   1   |  50.44 ± 0.82  | 50.4 ± 0.8<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 50.18 ± 0.46 |
| Mini-ImageNet | Conv  |  5   |   5   |  65.32 ± 0.70  | 65.3 ± 0.7<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 65.34 ± 0.41 |
|    CifarFS    | Conv  |  5   |   1   |       --       | 55.0 ± 1.0<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 55.84 ± 0.37 |
|    CifarFS    | Conv  |  5   |   5   |       --       | 69.3 ± 0.8<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) | 69.57 ± 0.30 |
|     FC100     | Conv  |  5   |   1   |       --       |                              --                              | 35.80 ± 0.18 |
|     FC100     | Conv  |  5   |   5   |       --       |                              --                              | 47.80 ± 0.24 |
|     CubFS     | Conv  |  5   |   1   |       --       | 62.34 ± 0.94<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)) | 57.40 ± 0.36 |
|     CubFS     | Conv  |  5   |   5   |       --       | 77.84 ± 0.68<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)) | 72.09 ± 0.31 |



