# PaddleFSL raw data

Raw data directory of the PaddleFSL(Paddle Few Shot Learning)

We provide the following datasets which are commly used in few-shot learning, as well as the pre-processing in paddlefsl.dataset:

### Computer vision, image classification task datasets:

- **Omniglot** ([B. M. Lake et al., 2015](http://www.sciencemag.org/content/350/6266/1332.short)), which is downloaded from

   (https://github.com/brendenlake/omniglot/raw/master/python/)

- **Mini-ImageNet** ([O. Vinyals et al., 2016](https://arxiv.org/abs/1606.04080)), which is downloaded from 

  (https://drive.google.com/file/d/1LLUjwSUpWGSWizl3JZxd08V30_dIaRBx/view)

- **Tiered-ImageNet** ([M. Ren et al., 2018](https://arxiv.org/abs/1803.00676)), which is downloaded from

  (https://drive.google.com/file/d/1fQ6lI5pCnOEt9MHWdqFN1cdSU2SbMKzx/view)

- **CIFAR-FS** ([L. Bertinetto et al., 2018](https://arxiv.org/abs/1805.08136)), which is downloaded from 

  (https://drive.google.com/file/d/1nN1u2ZeD0L90uG5Y_Ml6uQR6z-o6aBLL/view)

- **FC100** ([B. N. Oreshkin et al., 2018](https://arxiv.org/abs/1805.10123)), which is downloaded from 

  (https://drive.google.com/file/d/18SPp-RLOL-nxxoHtkU8-n8OspDjMfhAH/view)

- **CUB** ([W.-Y. Chen et al., 2019](https://arxiv.org/abs/1904.04232)) , which is downloaded from 

  (https://drive.google.com/file/d/1EiKOk6LAqlYwDJzUQRDUjGMsvUGBT1U8/view)

Natural language processing, relation classification task datasets:

- **FewRel1.0** ([Xu Han et al., 2018](https://aclanthology.org/D18-1514.pdf)), which is downloaded from

  (https://github.com/thunlp/FewRel/tree/master/data)

We use the train / val / test splits as mentioned in the respective papers. Note there is no specific split for Omniglot, thus we use random train / val / test splits.

Omniglot dataset and FewRel dataset can be automatically downloaded from github, while the ohters exceed the file limit, thus please download them from the website above and put them into the respective folder. Default folder used by paddlefsl is `<path to>/PaddleFSL/raw_data` .

### Usage:

```python
# Place mini-imagenet.tar.gz into this directory(<path to>/PaddleFSL/raw_data) or any path that you set.
train_dataset = paddlefsl.datasets.MiniImageNet(mode='train')
task = train_dataset.sample_task_set(ways=5, shots=5)
task.support_data  # numpy array of the images.
```

