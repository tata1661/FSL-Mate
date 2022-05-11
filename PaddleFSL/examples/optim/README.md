# Image Classification Tasks

Here, we provide examples of applying PaddleFSL to few-shot image classification tasks which is similarity to example with [model_zoo](../image_classification/README.md).


## Datasets

We evaluate the performance on 5 benchmark datasets, including Omniglot, *mini*ImageNet, CIFAR-FS, FC100 and Tiered-ImageNet, which can be accessed as described in [raw_data/README.md](../../raw_data/README.md).


## Results

We provide results of using MAML [1], ANIL [2] below. The exact model configuration and pretrained models can be downloaded from [here](https://drive.google.com/file/d/1pmCI-8cwLsadG6JOcubufrQ2d4zpK9B-/view?usp=sharing), which can reproduce these results.

### [MAML](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf?source=post_page---------------------------)


|     Dataset     | Backbone | Way  | Shot | Original paper |                        Other reports                         | model zoo(first order) | Optim(first order) |
| :-------------: | :------: | :--: | :--: | :------------: | :----------------------------------------------------------: | :--------------------: | ------------------ |
|    Omniglot     |   MLP    |  5   |  1   |   89.7 ± 1.1   |       88.9<br>([learn2learn](http://learn2learn.net/))       |      88.88 ± 2.99      | --                 |
|    Omniglot     |   MLP    |  5   |  5   |   97.5 ± 0.6   |                              --                              |      97.50 ± 0.47      | --                 |
|    Omniglot     |   CNN    |  5   |  1   |   98.7 ± 0.4   |      99.1<br/>([learn2learn](http://learn2learn.net/))       |      97.13 ± 1.25      | 92.7               |
|    Omniglot     |   CNN    |  5   |  5   |   99.9 ± 0.1   | 99.9 ± 0.1<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) |      99.23 ± 0.40      | ***93.1***         |
| *mini*ImageNet  |   CNN    |  5   |  1   |  48.70 ± 1.84  |      48.3<br/>([learn2learn](http://learn2learn.net/))       |      49.81 ± 1.78      |                    |
| *mini*ImageNet  |   CNN    |  5   |  5   |  63.11 ± 0.92  |      65.4<br/>([learn2learn](http://learn2learn.net/))       |      64.21 ± 1.33      | --                 |
|    CIFAR-FS     |   CNN    |  5   |  1   |       --       | 58.9 ± 1.9<br/>([R2D2](https://arxiv.org/pdf/1805.08136.pdf)) |      57.06 ± 3.83      | 49.1               |
|    CIFAR-FS     |   CNN    |  5   |  5   |       --       |      76.6<br/>([learn2learn](http://learn2learn.net/))       |      72.24 ± 1.71      | --                 |
|      FC100      |   CNN    |  5   |  1   |       --       |                              --                              |      37.63 ± 2.23      | 30.2               |
|      FC100      |   CNN    |  5   |  5   |       --       |      49.0<br/>([learn2learn](http://learn2learn.net/))       |      49.14 ± 1.58      | --                 |
|       CUB       |   CNN    |  5   |  1   |       --       | 54.73 ± 0.97<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)) |      53.31 ± 1.77      | 20.7               |
|       CUB       |   CNN    |  5   |  5   |       --       | 75.75 ± 0.76<br/>([CloseLookFS](https://arxiv.org/pdf/1904.04232.pdf)) |      69.88 ± 1.47      | --                 |
| Tiered-ImageNet |   CNN    |  5   |  5   |       --       |                              --                              |      67.56 ± 1.80      | --                 |

### [ANIL](https://openreview.net/pdf?id=rkgMkCEtPB)

|    Dataset     | Backbone | Way  | Shot | Author Report |                   Other Report                    | model zoo(first order) | Optimizer(First Order) |
| :------------: | :------: | :--: | :--: | :-----------: | :-----------------------------------------------: | :--------------------: | ---------------------- |
|    Omniglot    |   CNN    |  5   |  1   |      --       |                        --                         |      96.06 ± 1.00      | 96.34 ± 1.98           |
|    Omniglot    |   CNN    |  5   |  5   |      --       |                        --                         |      98.74 ± 0.48      |                        |
| *mini*ImageNet |   CNN    |  5   |  1   |  46.7 ± 0.4   |                        --                         |      48.31 ± 2.83      | 45.31 ± 1.43           |
| *mini*ImageNet |   CNN    |  5   |  5   |  61.5 ± 0.5   |                        --                         |      62.38 ± 1.96      | 61.81 ± 1.2            |
|    CIFAR-FS    |   CNN    |  5   |  1   |      --       |                        --                         |      56.19 ± 3.39      | ***30.8 ± 2.5***       |
|    CIFAR-FS    |   CNN    |  5   |  5   |      --       | 68.3<br/>([learn2learn](http://learn2learn.net/)) |      68.60 ± 1.25      | 48.6                   |
|     FC100      |   CNN    |  5   |  1   |      --       |                        --                         |      40.69 ± 3.32      | 38.4 ± 1.3             |
|     FC100      |   CNN    |  5   |  5   |      --       | 47.6<br/>([learn2learn](http://learn2learn.net/)) |      48.01 ± 1.22      | 35.0                   |
|      CUB       |   CNN    |  5   |  1   |      --       |                        --                         |      53.25 ± 2.18      | --                     |
|      CUB       |   CNN    |  5   |  5   |      --       |                        --                         |      69.09 ± 1.12      | --                     |

