# FewCLUE Tasks

For FewCLUE tasks, we currently provide implementation of three algorithms: PET, P-Tuning and EFL. 

## Experiments
We report accuracy on test_public.json of 9 FewCLUE datasets.

| Algorithms | Pretrained Model | Score  | eprstmt  | bustm  | ocnli  | csldcp  | tnews  |  cluewsc | iflytek | csl | chid |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ | ------------ | ---------- |
| P-tuning  | ERNIE1.0  | 55.70 | 83.28  | 63.43  | 35.36  | 60.54  | 50.02  | 54.51  | 50.14 | 54.93 | 41.16 |
| EFL       | ERNIE1.0  | 54.47 | 84.10  | 60.10  | 35.12  | 56.61  | 56.57  | 53.59  | 46.37 | 61.21 | 36.56 |
| PET       | ERNIE1.0  | 56.63 | 86.88  | 61.90  | 36.90  | 61.10  | 56.51  | 55.02  | 50.31 | 59.72 | 41.35 |

## Algorithms
- [P-tuning](./p-tuning)
- [EFL](./efl)
- [PET](./pet)

## References

- [1] Liu, Xiao, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. “GPT Understands, Too.” ArXiv:2103.10385 [Cs], March 18, 2021. http://arxiv.org/abs/2103.10385.
- [2] Wang, Sinong, Han Fang, Madian Khabsa, Hanzi Mao, and Hao Ma. “Entailment as Few-Shot Learner.” ArXiv:2104.14690 [Cs], April 29, 2021. http://arxiv.org/abs/2104.14690.
- [3] Wang, S., Fang, H., Khabsa, M., Mao, H., and Ma, H., “Entailment as Few-Shot Learner”, ArXiv:2001.07676 [Cs], 2021. https://arxiv.org/abs/2001.0767
