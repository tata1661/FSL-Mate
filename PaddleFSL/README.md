# PaddleFSL
[![PyPI version](https://badge.fury.io/py/paddlefsl.svg)](https://pypi.org/project/paddlefsl/)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue)](https://github.com/tata1661/FSL-Mate/tree/master/PaddleFSL/CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-purple.svg)](https://opensource.org/licenses/Apache-2.0)

PaddleFSL is a Python library for few-shot learning (FSL) built upon PaddlePaddle 2.0.
- Provide various 
FSL solutions which are applicable to diverse applications. 
- Contain detailed 
annotations and tutorial examples, such that users can easily develop and compare different FSL solutions. 
- Can be easily deployed on various training platforms. 

![](https://github.com/tata1661/FSL-Mate/blob/master/PaddleFSL/art-paddlefsl.png)

## Overview

- **paddlefsl**: The proposed package for FSL.
  - **paddlefsl.backbones**: It provides representation learning models to extract sample representation, such as CNN, GNN and PLM.
  - **paddlefsl.datasets**: It provides publicly accessible benchmark few-shot datasets of diverse application scenarios, such as Omniglot, FewRel, FewGLUE and Tox21.
  - **paddlefsl.model_zoo**: It provides classical FSL models which particularly deal with the lack of labeled data, such as ProtoNet, MAML and PET.
  - **paddlefsl.task_sampler**: It provides APIs to split datasets into the typical N-way K-shot tasks.
  - **paddlefsl.utils**: It provides auxiliary APIs of PaddleFSL. 
- **examples**: It provides examples of applying PaddleFSL to diverse application scenarios, such as computer vision tasks, natural language tasks and bioinformatics tasks.
- **raw_data**: It provides guides to download and place the raw data files.
- **test**: It provides unit test files of functions and classes.


## Installation

To use our package, users should first install paddlepaddle-v2.0.0 or later versions, see https://www.paddlepaddle.org.cn/install .

First, install our required packages.

```bash
# Clone our repository.
git clone https://github.com/tata1661/FSL-Mate.git
# Install requirements.
cd FSL-Mate/PaddleFSL
python setup.py install
```

Second, update environment variables.

```bash
# Please edit env.sh to set the correct path of FSL-paddletoolkit directory.
# Then do:
source env.sh
# If you want to use our package frequently, you can add environment variables into .bashrc
cat env.sh >> ~/.bashrc
source ~/.bashrc
```

Finally, check whether the installation is successful.

```bash
# Start a python interpreter
python
>>> import paddlefsl
>>> paddlefsl.__version__
'1.1.0'
```

## Cite Us

Please cite our [paper](https://cse.hkust.edu.hk/~ywangcy/aux_file/PaddleFSL-2022.pdf) if you find PaddleFSL useful.
```
@misc{shen2022paddlefsl,
  title={PaddleFSL: A General Few-Shot Learning Toolbox in Python},
  author={Shen, Zhenqian and Wang, Yaqing and Xiong, Haoyi and Tian, Xin and Chen, Zeyu and Yao, Quanming and Dou, Dejing},
  year={2022},
  url={https://cse.hkust.edu.hk/~ywangcy/aux_file/PaddleFSL-2022.pdf},
  note={Available at: https://github.com/tata1661/FSL-Mate/tree/master/PaddleFSL}
}
```

## Contributing


PaddleFSL is mainly contributed by W Group led by [Yaqing Wang](https://cse.hkust.edu.hk/~ywangcy/). The full list of PaddleFSL contributors is [here](../CONTRIBUTING.md).


We also welcome and appreciate community contribution to improve PaddleFSL, such as introducing new datasets, models and algorithms, designing new features and fixing bugs. 
The codes can be contributed via pull requests and code review. 
Please also feel free to open an issue for feedbacks or advices. 


