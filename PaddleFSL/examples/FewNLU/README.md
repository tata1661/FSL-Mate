# Natural Language Understanding Tasks

Here, we provide examples of applying PaddleFSL to few-shot natural language understanding (FewNLU) tasks. 


## Environment

To run the examples, please first install paddlenlp-2.0.6 or later versions as follows: 

```bash
pip install paddlenlp -i https://pypi.org/simple
```
Or see [here](https://paddlenlp.readthedocs.io/zh/latest/get_started/installation.html) for details.

## Datasets

We evaluate the performance on 2 benchmark sets of datasets, including the FewGLUE [1] in English and the FewCLUE [2] in Chinese, which can be accessed as described in [raw_data/README.md](../../raw_data/README.md).


## Results

We provide results of using PET [1] and P-tuning [3] below. 

### Accuracy Obtained on dev32.jsonl of Four FewGLUE Datasets

| Model | Backbone   | CB     | BoolQ  | WiC    | RTE    | 
|------------|--------------------|--------|--------|--------|--------| 
| PET        | bert-large-uncased | 81.250 | 62.500 | 75.000 | 68.750 |
| P-tuning   | bert-large-uncased | 78.125 | 62.500 | 75.000 | 71.875 |

### Accuracy Obtained on test_public.json of Nine FewCLUE Datasets

| Model | Backbone | eprstmt  | bustm  | ocnli  | csldcp  | tnews  |  cluewsc | iflytek | csl | chid |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ | ------------ | ---------- |
| PET       | ERNIE1.0   | 86.88  | 61.90  | 36.90  | 61.10  | 56.51  | 55.02  | 50.31 | 59.72 | 41.35 
| P-tuning  | ERNIE1.0   | 83.28  | 63.43  | 35.36  | 60.54  | 50.02  | 54.51  | 50.14 | 54.93 | 41.16 |


## References

1. *It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners,* in NAACL, 2021.
T. Schick,and H. Schütze.

1. *GPT Understands, Too,* in ArXiv, 2021.
L. Xiao, Y. Zheng, Z. Du, M. Ding, Y. Qian, Z. Yang, and J. Tang.

1. *FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark,* in ArXiv, 2021.
L. Xu, X. Lu, C. Yuan, X. Zhang, H. Xu, H. Yuan, G. Wei, X. Pan, X. Tian, L. Qin, and H. Hai
