# Few-Shot Papers

This repository contains few-shot learning (FSL) papers mentioned in our FSL survey published in ACM Computing Surveys (JCR Q1, CORE A*). 

For convenience, we also include public implementations of respective authors.

We will update this paper list to include new FSL papers periodically. 

## Citation 

Please cite our paper if you find it helpful.

```
@article{wang2020generalizing,
  title={Generalizing from a few examples: A survey on few-shot learning},
  author={Wang, Yaqing and Yao, Quanming and Kwok, James T and Ni, Lionel M},
  journal={ACM Computing Surveys},
  volume={53},
  number={3},
  pages={1--34},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```


## Content
1. [Survey](#Survey)
2. [Data](#Data)
3. [Model](#Model)
    1. [Multitask Learning](#Multitask-Learning)
    1. [Embedding/Metric Learning](#Embedding/Metric-Learning)
    1. [Learning with External Memory](#Learning-with-External-Memory)
    1. [Generative Modeling](#Generative-Modeling)
4. [Algorithm](#Algorithm)
    1. [Refining Existing Parameters](#Refining-Existing-Parameters)
    1. [Refining Meta-learned Parameters](#Refining-Meta-learned-Parameters)
    1. [Learning Search Steps](#Learning-Search-Steps)
5. [Applications](#Applications)
    1. [Computer Vision](#Computer-Vision)
    1. [Robotics](#Robotics)
    1. [Natural Language Processing](#Natural-Language-Processing)
    1. [Knowledge Graph](#Knowledge-Graph)
    1. [Acoustic Signal Processing](#Acoustic-Signal-Processing)
    1. [Recommendation](#Recommendation)
    1. [Others](#others)
6. [Theories](#Theories)
7. [Few-shot Learning and Zero-shot Learning](#Few-shot-Learning-and-Zero-shot-Learning)
8. [Variants of Few-shot Learning](#Variants-of-Few-shot-Learning)
9. [Datasets/Benchmarks](#Datasets/Benchmarks)
10. [Software Library](#Software-Library)


## [Survey](#content)

1. **Generalizing from a few examples: A survey on few-shot learning,** CSUR, 2020
*Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni.*
[paper](https://dl.acm.org/doi/10.1145/3386252?cid=99659542534)
[arXiv](https://arxiv.org/abs/1904.05046)

## [Data](#content)

1. **Learning from one example through shared densities on transforms,** in CVPR, 2000.
*E. G. Miller, N. E. Matsakis, and P. A. Viola.*
[paper](https://people.cs.umass.edu/~elm/papers/Miller_congealing.pdf)

1. **Domain-adaptive discriminative one-shot learning of gestures,** in ECCV, 2014.
*T. Pfister, J. Charles, and A. Zisserman.*
[paper](https://www.robots.ox.ac.uk/~vgg/publications/2014/Pfister14/pfister14.pdf)

1. **One-shot learning of scene locations via feature trajectory transfer,** in CVPR, 2016.
*R. Kwitt, S. Hegenbart, and M. Niethammer.* 
[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Kwitt_One-Shot_Learning_of_CVPR_2016_paper.pdf)

1. **Low-shot visual recognition by shrinking and hallucinating features,** in ICCV, 2017.
*B. Hariharan and R. Girshick.*
[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hariharan_Low-Shot_Visual_Recognition_ICCV_2017_paper.pdf)
[code](https://github.com/facebookresearch/low-shot-shrink-hallucinate)

1. **Improving one-shot learning through fusing side information,** arXiv preprint, 2017.
*Y.H.Tsai and R.Salakhutdinov.*
[paper](https://lld-workshop.github.io/2017/papers/LLD_2017_paper_31.pdf)

1. **Fast parameter adaptation for few-shot image captioning and visual question answering,** in ACM MM, 2018.
*X. Dong, L. Zhu, D. Zhang, Y. Yang, and F. Wu.* 
[paper](https://xuanyidong.com/resources/papers/ACM-MM-18-FPAIT.pdf)

1. **Exploit the unknown gradually: One-shot video-based person re-identification by stepwise learning,** in CVPR, 2018.
*Y. Wu, Y. Lin, X. Dong, Y. Yan, W. Ouyang, and Y. Yang.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf)

1. **Low-shot learning with large-scale diffusion,** in CVPR, 2018.
*M. Douze, A. Szlam, B. Hariharan, and H. Jégou.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Douze_Low-Shot_Learning_With_CVPR_2018_paper.pdf)

1. **Diverse few-shot text classification with multiple metrics,** in NAACL-HLT, 2018.
*M. Yu, X. Guo, J. Yi, S. Chang, S. Potdar, Y. Cheng, G. Tesauro, H. Wang, and B. Zhou.* 
[paper](https://www.aclweb.org/anthology/N18-1109.pdf)
[code](https://github.com/Gorov/DiverseFewShot_Amazon)

1. **Delta-encoder: An effective sample synthesis method for few-shot object recognition,** in NeurIPS, 2018.
*E. Schwartz, L. Karlinsky, J. Shtok, S. Harary, M. Marder, A. Kumar, R. Feris, R. Giryes, and A. Bronstein.*
[paper](https://papers.nips.cc/paper/7549-delta-encoder-an-effective-sample-synthesis-method-for-few-shot-object-recognition.pdf)

1. **Low-shot learning via covariance-preserving adversarial augmentation networks,** in NeurIPS, 2018.
*H. Gao, Z. Shou, A. Zareian, H. Zhang, and S. Chang.*
[paper](https://papers.nips.cc/paper/7376-low-shot-learning-via-covariance-preserving-adversarial-augmentation-networks.pdf)

1. **Learning to self-train for semi-supervised few-shot classification,** in NeurIPS, 2019.
*X. Li, Q. Sun, Y. Liu, S. Zheng, Q. Zhou, T.-S. Chua, and B. Schiele.*
[paper](https://papers.nips.cc/paper/9216-learning-to-self-train-for-semi-supervised-few-shot-classification.pdf)

1. **Few-shot learning with global class representations,** in ICCV, 2019.
*A. Li, T. Luo, T. Xiang, W. Huang, and L. Wang.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Few-Shot_Learning_With_Global_Class_Representations_ICCV_2019_paper.pdf)

1. **AutoAugment: Learning augmentation policies from data,** in CVPR, 2019.
*E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le.*
[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf)

1. **EDA: Easy data augmentation techniques for boosting performance on text classification tasks,** in EMNLP and IJCNLP, 2019.
*J. Wei and K. Zou.*
[paper](https://www.aclweb.org/anthology/D19-1670.pdf)

1. **LaSO: Label-set operations networks for multi-label few-shot learning,** in CVPR, 2019.
*A. Alfassy, L. Karlinsky, A. Aides, J. Shtok, S. Harary, R. Feris, R. Giryes, and A. M. Bronstein.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Alfassy_LaSO_Label-Set_Operations_Networks_for_Multi-Label_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/leokarlin/LaSO)

1. **Image deformation meta-networks for one-shot learning,** in CVPR, 2019.
*Z. Chen, Y. Fu, Y.-X. Wang, L. Ma, W. Liu, and M. Hebert.*
[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Image_Deformation_Meta-Networks_for_One-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/tankche1/IDeMe-Net)

1. **Spot and learn: A maximum-entropy patch sampler for few-shot image classification,** in CVPR, 2019.
*W.-H. Chu, Y.-J. Li, J.-C. Chang, and Y.-C. F. Wang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf)

1. **Adversarial feature hallucination networks for few-shot learning,** in CVPR, 2020.
*K. Li, Y. Zhang, K. Li, and Y. Fu.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Adversarial_Feature_Hallucination_Networks_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Instance credibility inference for few-shot learning,** in CVPR, 2020.
*Y. Wang, C. Xu, C. Liu, L. Zhang, and Y. Fu.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Instance_Credibility_Inference_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Diversity transfer network for few-shot learning,** in AAAI, 2020.
*M. Chen, Y. Fang, X. Wang, H. Luo, Y. Geng, X. Zhang, C. Huang, W. Liu, and B. Wang.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6628)
[code](https://github.com/Yuxin-CV/DTN)

1. **Neural snowball for few-shot relation learning,** in AAAI, 2020.
*T. Gao, X. Han, R. Xie, Z. Liu, F. Lin, L. Lin, and M. Sun.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6281)
[code](https://github.com/thunlp/Neural-Snowball)

1. **Associative alignment for few-shot image classification,** in ECCV, 2020.
*A. Afrasiyabi, J. Lalonde, and C. Gagné.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500018.pdf)
[code](https://github.com/ArmanAfrasiyabi/associative-alignment-fs)

1. **Information maximization for few-shot learning,** in NeurIPS, 2020.
*M. Boudiaf, I. Ziko, J. Rony, J. Dolz, P. Piantanida, and I. B. Ayed.*
[paper](https://proceedings.neurips.cc/paper/2020/file/196f5641aa9dc87067da4ff90fd81e7b-Paper.pdf)
[code](https://github.com/mboudiaf/TIM)

1. **Self-training for few-shot transfer across extreme task differences,** in ICLR, 2021.
*C. P. Phoo, and B. Hariharan.*
[paper](https://openreview.net/pdf?id=O3Y56aqpChA)

1. **Free lunch for few-shot learning: Distribution calibration,** in ICLR, 2021.
*S. Yang, L. Liu, and M. Xu.*
[paper](https://openreview.net/pdf?id=JWOiYxMG92s)
[code](https://github.com/ShuoYang-1998/ICLR2021-Oral_Distribution_Calibration)

1. **Parameterless transductive feature re-representation for few-shot learning,** in ICML, 2021.
*W. Cui, and Y. Guo;.*
[paper](http://proceedings.mlr.press/v139/cui21a/cui21a.pdf)

1. **Learning intact features by erasing-inpainting for few-shot classification,** in AAAI, 2021.
*J. Li, Z. Wang, and X. Hu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17021/16828)

1. **Variational feature disentangling for fine-grained few-shot classification,** in ICCV, 2021.
*J. Xu, H. Le, M. Huang, S. Athar, and D. Samaras.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Variational_Feature_Disentangling_for_Fine-Grained_Few-Shot_Classification_ICCV_2021_paper.pdf)

1. **Coarsely-labeled data for better few-shot transfer,** in ICCV, 2021.
*C. P. Phoo, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Phoo_Coarsely-Labeled_Data_for_Better_Few-Shot_Transfer_ICCV_2021_paper.pdf)

1. **Pseudo-loss confidence metric for semi-supervised few-shot learning,** in ICCV, 2021.
*K. Huang, J. Geng, W. Jiang, X. Deng, and Z. Xu.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Pseudo-Loss_Confidence_Metric_for_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Iterative label cleaning for transductive and semi-supervised few-shot learning,** in ICCV, 2021.
*M. Lazarou, T. Stathaki, and Y. Avrithis.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lazarou_Iterative_Label_Cleaning_for_Transductive_and_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Meta two-sample testing: Learning kernels for testing with limited data,** in NeurIPS, 2021.
*F. Liu, W. Xu, J. Lu, and D. J. Sutherland.*
[paper](https://proceedings.neurips.cc/paper/2021/file/2e6d9c6052e99fcdfa61d9b9da273ca2-Paper.pdf)

1. **Dynamic distillation network for cross-domain few-shot recognition with unlabeled data,** in NeurIPS, 2021.
*A. Islam, C.-F. Chen, R. Panda, L. Karlinsky, R. Feris, and R. Radke.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1d6408264d31d453d556c60fe7d0459e-Paper.pdf)

1. **Towards better understanding and better generalization of low-shot classification in histology images with contrastive learning,** in ICLR, 2022.
*J. Yang, H. Chen, J. Yan, X. Chen, and J. Yao.*
[paper](https://openreview.net/pdf?id=kQ2SOflIOVC)
[code](https://github.com/TencentAILabHealthcare/Few-shot-WSI)

1. **FlipDA: Effective and robust data augmentation for few-shot learning,** in ACL, 2022.
*J. Zhou, Y. Zheng, J. Tang, L. Jian, and Z. Yang.*
[paper](https://aclanthology.org/2022.acl-long.592.pdf)
[code](https://github.com/zhouj8553/flipda)

1. **PromDA: Prompt-based data augmentation for low-resource NLU tasks,** in ACL, 2022.
*Y. Wang, C. Xu, Q. Sun, H. Hu, C. Tao, X. Geng, and D. Jiang.*
[paper](https://aclanthology.org/2022.acl-long.292.pdf)
[code](https://github.com/garyyufei/promda)

1. **N-shot learning for augmenting task-oriented dialogue state tracking,** in Findings of ACL, 2022.
*I. T. Aksu, Z. Liu, M. Kan, and N. F. Chen.*
[paper](https://aclanthology.org/2022.findings-acl.131.pdf)

1. **Generating representative samples for few-shot classification,** in CVPR, 2022.
*J. Xu, and H. Le.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Generating_Representative_Samples_for_Few-Shot_Classification_CVPR_2022_paper.pdf)
[code](https://github.com/cvlab-stonybrook/)

1. **Semi-supervised few-shot learning via multi-factor clustering,** in CVPR, 2022.
*J. Ling, L. Liao, M. Yang, and J. Shuai.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ling_Semi-Supervised_Few-Shot_Learning_via_Multi-Factor_Clustering_CVPR_2022_paper.pdf)

1. **Information augmentation for few-shot node classification,** in IJCAI, 2022.
*Z. Wu, P. Zhou, G. Wen, Y. Wan, J. Ma, D. Cheng, and X. Zhu.*
[paper](https://www.ijcai.org/proceedings/2022/0500.pdf)

1. **Improving task-specific generalization in few-shot learning via adaptive vicinal risk minimization,** in NeurIPS, 2022.
*L.-K. Huang, and Y. Wei.*
[paper](https://openreview.net/pdf?id=fHUBa3gQno)

1. **An embarrassingly simple approach to semi-supervised few-shot learning,** in NeurIPS, 2022.
*X.-S. Wei, H.-Y. Xu, F. Zhang, Y. Peng, and W. Zhou.*
[paper](https://openreview.net/pdf?id=-3Pg7QNIF1S)

1. **FeLMi : Few shot learning with hard mixup,** in NeurIPS, 2022.
*A. Roy, A. Shah, K. Shah, P. Dhar, A. Cherian, and R. Chellappa.*
[paper](https://openreview.net/pdf?id=xpdaDM_B4D)
[code](https://github.com/aniket004/Felmi)

1. **Understanding cross-domain few-shot learning based on domain similarity and few-shot difficulty,** in NeurIPS, 2022.
*J. Oh, S. Kim, N. Ho, J.-H. Kim, H. Song, and S.-Y. Yun.*
[paper](https://openreview.net/pdf?id=rH-X09cB50f)
[code](https://github.com/sungnyun/understanding-cdfsl)

1. **Label hallucination for few-shot classification,** in AAAI, 2022.
*Y. Jian, and L. Torresani.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20659/20418)
[code](https://github.com/yiren-jian/LabelHalluc)

1. **STUNT: Few-shot tabular learning with self-generated tasks from unlabeled tables,** in ICLR, 2023.
*J. Nam, J. Tack, K. Lee, H. Lee, and J. Shin.*
[paper](https://openreview.net/pdf?id=_xlsjehDvlY)
[code](https://github.com/jaehyun513/STUNT)

1. **Unsupervised meta-learning via few-shot pseudo-supervised contrastive learning,** in ICLR, 2023.
*H. Jang, H. Lee, and J. Shin.*
[paper](https://openreview.net/pdf?id=TdTGGj7fYYJ)
[code](https://github.com/alinlab/PsCo)

1. **Progressive mix-up for few-shot supervised multi-source domain transfer,** in ICLR, 2023.
*R. Zhu, R. Zhu, X. Yu, and S. Li.*
[paper](https://openreview.net/pdf?id=H7M_5K5qKJV)
[code](https://github.com/ronghangzhu/P-Mixup)

1. **Cross-level distillation and feature denoising for cross-domain few-shot classification,** in ICLR, 2023.
*H. ZHENG, R. Wang, J. Liu, and A. Kanezaki.*
[paper](https://openreview.net/pdf?id=Kn-HA8DFik)
[code](https://gitee.com/mindspore/models/tree/master/research/cv/CLDFD)

1. **Tuning language models as training data generators for augmentation-enhanced few-shot learning,** in ICML, 2023.
*Y. Meng, M. Michalski, J. Huang, Y. Zhang, T. F. Abdelzaher, and J. Han.*
[paper](http://proceedings.mlr.press/v202/meng23b/meng23b.pdf)
[code](https://github.com/yumeng5/FewGen)

## [Model](#content)

### Multitask Learning

1. **Multi-task transfer methods to improve one-shot learning for multimedia event detection,** in BMVC, 2015.
*W. Yan, J. Yap, and G. Mori.*
[paper](http://www.bmva.org/bmvc/2015/papers/paper037/index.html)

1. **Label efficient learning of transferable representations across domains and tasks,** in NeurIPS, 2017.
*Z. Luo, Y. Zou, J. Hoffman, and L. Fei-Fei.*
[paper](https://papers.nips.cc/paper/6621-label-efficient-learning-of-transferable-representations-acrosss-domains-and-tasks.pdf)

1. **Few-shot adversarial domain adaptation,** in NeurIPS, 2017.
*S. Motiian, Q. Jones, S. Iranmanesh, and G. Doretto.*
[paper](https://papers.nips.cc/paper/7244-few-shot-adversarial-domain-adaptation)

1. **One-shot unsupervised cross domain translation,** in NeurIPS, 2018.
*S. Benaim and L. Wolf.* 
[paper](https://papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation.pdf)

1. **Multi-content GAN for few-shot font style transfer,** in CVPR, 2018. 
*S. Azadi, M. Fisher, V. G. Kim, Z. Wang, E. Shechtman, and T. Darrell.*
[paper](http://www.vovakim.com/papers/18_CVPRSpotlight_FontDropper.pdf)
[code](https://github.com/azadis/MC-GAN)

1. **Feature space transfer for data augmentation,** in CVPR, 2018.
*B. Liu, X. Wang, M. Dixit, R. Kwitt, and N. Vasconcelos.* 
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Feature_Space_Transfer_CVPR_2018_paper.pdf)

1. **Fine-grained visual categorization using meta-learning optimization with sample selection of auxiliary data,** in ECCV, 2018.
*Y. Zhang, H. Tang, and K. Jia.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yabin_Zhang_Fine-Grained_Visual_Categorization_ECCV_2018_paper.pdf)

1. **Few-shot charge prediction with discriminative legal attributes,** in COLING, 2018.
*Z. Hu, X. Li, C. Tu, Z. Liu, and M. Sun.*
[paper](https://www.aclweb.org/anthology/C18-1041.pdf)

1. **Boosting few-shot visual learning with self-supervision,** in ICCV, 2019.
*S. Gidaris, A. Bursuc, N. Komodakis, P. Pérez, and M. Cord.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gidaris_Boosting_Few-Shot_Visual_Learning_With_Self-Supervision_ICCV_2019_paper.pdf)

1. **When does self-supervision improve few-shot learning?,** in ECCV, 2020.
*J. Su, S. Maji, and B. Hariharan.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520630.pdf)

1. **Pareto self-supervised training for few-shot learning,** in CVPR, 2021.
*Z. Chen, J. Ge, H. Zhan, S. Huang, and D. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pareto_Self-Supervised_Training_for_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Bridging multi-task learning and meta-learning: Towards efficient training and effective adaptation,** in ICML, 2021.
*H. Wang, H. Zhao, and B. Li;.*
[paper](http://proceedings.mlr.press/v139/wang21ad/wang21ad.pdf)
[code](https://github.com/AI-secure/multi-task-learning)

1. **Task-level self-supervision for cross-domain few-shot learning,** in AAAI, 2022.
*W. Yuan, Z. Zhang, C. Wang, H. Song, Y. Xie, and L. Ma.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20230/19989)

21. **Improving few-shot generalization by exploring and exploiting auxiliary data,** in NeurIPS, 2023.
*A. Albalak, C. Raffel, and W. Y. Wang.*
[paper](https://openreview.net/attachment?id=JDnLXc4NOn&name=pdf)
[code](https://github.com/alon-albalak/FLAD)


### Embedding/Metric Learning

1. **Object classification from a single example utilizing class relevance metrics,** in NeurIPS, 2005.
*M. Fink.*
[paper](https://papers.nips.cc/paper/2576-object-classification-from-a-single-example-utilizing-class-relevance-metrics.pdf)

1. **Optimizing one-shot recognition with micro-set learning,** in CVPR, 2010.
*K. D. Tang, M. F. Tappen, R. Sukthankar, and C. H. Lampert.*
[paper](http://www.cs.ucf.edu/~mtappen/pubs/cvpr10_oneshot.pdf)

1. **Siamese neural networks for one-shot image recognition,** ICML deep learning workshop, 2015.
*G. Koch, R. Zemel, and R. Salakhutdinov.*
[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

1. **Matching networks for one shot learning,** in NeurIPS, 2016.
*O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al.* 
[paper](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)

1. **Learning feed-forward one-shot learners,** in NeurIPS, 2016.
*L. Bertinetto, J. F. Henriques, J. Valmadre, P. Torr, and A. Vedaldi.*
[paper](https://papers.nips.cc/paper/6068-learning-feed-forward-one-shot-learners.pdf)

1. **Few-shot learning through an information retrieval lens,** in NeurIPS, 2017.
*E. Triantafillou, R. Zemel, and R. Urtasun.*
[paper](https://papers.nips.cc/paper/6820-few-shot-learning-through-an-information-retrieval-lens.pdf)

1. **Prototypical networks for few-shot learning,** in NeurIPS, 2017.
*J. Snell, K. Swersky, and R. S. Zemel.*
[paper](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)
[code](https://github.com/jakesnell/prototypical-networks)

1. **Attentive recurrent comparators,** in ICML, 2017.
*P. Shyam, S. Gupta, and A. Dukkipati.*
[paper](http://proceedings.mlr.press/v70/shyam17a/shyam17a.pdf)

1. **Learning algorithms for active learning,** in ICML, 2017.
*P. Bachman, A. Sordoni, and A. Trischler.*
[paper](http://proceedings.mlr.press/v70/bachman17a.pdf)

1. **Active one-shot learning,** arXiv preprint, 2017.
*M. Woodward and C. Finn.*
[paper](https://arxiv.org/abs/1702.06559)

1. **Structured set matching networks for one-shot part labeling,** in CVPR, 2018.
*J. Choi, J. Krishnamurthy, A. Kembhavi, and A. Farhadi.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Structured_Set_Matching_CVPR_2018_paper.pdf)

1. **Low-shot learning from imaginary data,** in CVPR, 2018.
*Y.-X. Wang, R. Girshick, M. Hebert, and B. Hariharan.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Low-Shot_Learning_From_CVPR_2018_paper.pdf)

1. **Learning to compare: Relation network for few-shot learning,** in CVPR, 2018.
*F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. Torr, and T. M. Hospedales.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf)
[code](https://github.com/floodsung/LearningToCompare_FSL)

1. **Dynamic conditional networks for few-shot learning,** in ECCV, 2018.
*F. Zhao, J. Zhao, S. Yan, and J. Feng.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fang_Zhao_Dynamic_Conditional_Networks_ECCV_2018_paper.pdf)
[code](https://github.com/ZhaoJ9014/Dynamic-Conditional-Networks.PyTorch)

1. **TADAM: Task dependent adaptive metric for improved few-shot learning,** in NeurIPS, 2018.
*B. Oreshkin, P. R. López, and A. Lacoste.*
[paper](https://papers.nips.cc/paper/7352-tadam-task-dependent-adaptive-metric-for-improved-few-shot-learning.pdf)

1. **Meta-learning for semi-supervised few-shot classification,** in ICLR, 2018.
*M. Ren, S. Ravi, E. Triantafillou, J. Snell, K. Swersky, J. B. Tenen- baum, H. Larochelle, and R. S. Zemel.* 
[paper](https://openreview.net/forum?id=r1n5Osurf)
[code](https://github.com/renmengye/few-shot-ssl-public)

1. **Few-shot learning with graph neural networks,** in ICLR, 2018.
*V. G. Satorras and J. B. Estrach.*
[paper](https://openreview.net/pdf?id=HJcSzz-CZ)
[code](https://github.com/vgsatorras/few-shot-gnn)

1. **A simple neural attentive meta-learner,** in ICLR, 2018.
*N. Mishra, M. Rohaninejad, X. Chen, and P. Abbeel.*
[paper](https://openreview.net/forum?id=B1DmUzWAW)

1. **Meta-learning with differentiable closed-form solvers,** in ICLR, 2019.
*L. Bertinetto, J. F. Henriques, P. Torr, and A. Vedaldi.* 
[paper](https://openreview.net/forum?id=HyxnZh0ct7)

1. **Learning to propagate labels: Transductive propagation network for few-shot learning,** in ICLR, 2019.
*Y. Liu, J. Lee, M. Park, S. Kim, E. Yang, S. Hwang, and Y. Yang.*
[paper](https://openreview.net/forum?id=SyVuRiC5K7)
[code](https://github.com/csyanbin/TPN-pytorch)

1. **Multi-level matching and aggregation network for few-shot relation classification,** in ACL, 2019.
*Z.-X. Ye, and Z.-H. Ling.*
[paper](https://www.aclweb.org/anthology/P19-1277.pdf)

1. **Induction networks for few-shot text classification,** in EMNLP-IJCNLP, 2019.
*R. Geng, B. Li, Y. Li, X. Zhu, P. Jian, and J. Sun.*
[paper](https://www.aclweb.org/anthology/D19-1403.pdf)

1. **Hierarchical attention prototypical networks for few-shot text classification,** in EMNLP-IJCNLP, 2019.
*S. Sun, Q. Sun, K. Zhou, and T. Lv.*
[paper](https://www.aclweb.org/anthology/D19-1045.pdf)

1. **Cross attention network for few-shot classification,** in NeurIPS, 2019.
*R. Hou, H. Chang, B. Ma, S. Shan, and X. Chen.*
[paper](https://papers.nips.cc/paper/8655-cross-attention-network-for-few-shot-classification.pdf)

1. **Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes,** in NeurIPS, 2019.
*J. Requeima, J. Gordon, J. Bronskill, S. Nowozin, and R. E. Turner.*
[paper](https://proceedings.neurips.cc/paper/2019/file/1138d90ef0a0848a542e57d1595f58ea-Paper.pdf)
[code](https://github.com/cambridge-mlg/cnaps)

1. **Hybrid attention-based prototypical networks for noisy few-shot relation classification,** in AAAI, 2019.
*T. Gao, X. Han, Z. Liu, and M. Sun.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4604/4482)
[code](https://github.com/thunlp/HATT-Proto)

1. **Attention-based multi-context guiding for few-shot semantic segmentation,** in AAAI, 2019.
*T. Hu, P. Yang, C. Zhang, G. Yu, Y. Mu and C. G. M. Snoek.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4604/4482)

1. **Distribution consistency based covariance metric networks for few-shot learning,** in AAAI, 2019.
*W. Li, L. Wang, J. Xu, J. Huo, Y. Gao and J. Luo.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4885/4758)

1. **A dual attention network with semantic embedding for few-shot learning,** in AAAI, 2019.
*S. Yan, S. Zhang, and X. He.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4940/4813)

1. **TapNet: Neural network augmented with task-adaptive projection for few-shot learning,** in ICML, 2019.
*S. W. Yoon, J. Seo, and J. Moon.*
[paper](http://proceedings.mlr.press/v97/yoon19a/yoon19a.pdf)

1. **Prototype propagation networks (PPN) for weakly-supervised few-shot learning on category graph,** in IJCAI, 2019.
*L. Liu, T. Zhou, G. Long, J. Jiang, L. Yao, C. Zhang.*
[paper](https://www.ijcai.org/Proceedings/2019/0418.pdf)
[code](https://github.com/liulu112601/Prototype-Propagation-Net)

1. **Collect and select: Semantic alignment metric learning for few-shot learning,** in ICCV, 2019.
*F. Hao, F. He, J. Cheng, L. Wang, J. Cao, and D. Tao.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hao_Collect_and_Select_Semantic_Alignment_Metric_Learning_for_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **Transductive episodic-wise adaptive metric for few-shot learning,** in ICCV, 2019.
*L. Qiao, Y. Shi, J. Li, Y. Wang, T. Huang, and Y. Tian.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qiao_Transductive_Episodic-Wise_Adaptive_Metric_for_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **Few-shot learning with embedded class models and shot-free meta training,** in ICCV, 2019.
*A. Ravichandran, R. Bhotika, and S. Soatto.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ravichandran_Few-Shot_Learning_With_Embedded_Class_Models_and_Shot-Free_Meta_Training_ICCV_2019_paper.pdf)

1. **PARN: Position-aware relation networks for few-shot learning,** in ICCV, 2019.
*Z. Wu, Y. Li, L. Guo, and K. Jia.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_PARN_Position-Aware_Relation_Networks_for_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **PANet: Few-shot image semantic segmentation with prototype alignment,** in ICCV, 2019.
*K. Wang, J. H. Liew, Y. Zou, D. Zhou, and J. Feng.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf)
[code](https://github.com/kaixin96/PANet)

1. **RepMet: Representative-based metric learning for classification and few-shot object detection,** in CVPR, 2019.
*L. Karlinsky, J. Shtok, S. Harary, E. Schwartz, A. Aides, R. Feris, R. Giryes, and A. M. Bronstein.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karlinsky_RepMet_Representative-Based_Metric_Learning_for_Classification_and_Few-Shot_Object_Detection_CVPR_2019_paper.pdf)
[code](https://github.com/jshtok/RepMet)

1. **Edge-labeling graph neural network for few-shot learning,** in CVPR, 2019.
*J. Kim, T. Kim, S. Kim, and C. D. Yoo.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Edge-Labeling_Graph_Neural_Network_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

1. **Finding task-relevant features for few-shot learning by category traversal,** in CVPR, 2019.
*H. Li, D. Eigen, S. Dodge, M. Zeiler, and X. Wang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Finding_Task-Relevant_Features_for_Few-Shot_Learning_by_Category_Traversal_CVPR_2019_paper.pdf)
[code](https://github.com/Clarifai/few-shot-ctm)

1. **Revisiting local descriptor based image-to-class measure for few-shot learning,** in CVPR, 2019.
*W. Li, L. Wang, J. Xu, J. Huo, Y. Gao, and J. Luo.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/WenbinLee/DN4)

1. **TAFE-Net: Task-aware feature embeddings for low shot learning,** in CVPR, 2019.
*X. Wang, F. Yu, R. Wang, T. Darrell, and J. E. Gonzalez.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_TAFE-Net_Task-Aware_Feature_Embeddings_for_Low_Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/ucbdrive/tafe-net)

1. **Improved few-shot visual classification,** in CVPR, 2020.
*P. Bateni, R. Goyal, V. Masrani, F. Wood, and L. Sigal.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.pdf)

1. **Boosting few-shot learning with adaptive margin loss,** in CVPR, 2020.
*A. Li, W. Huang, X. Lan, J. Feng, Z. Li, and L. Wang.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Boosting_Few-Shot_Learning_With_Adaptive_Margin_Loss_CVPR_2020_paper.pdf)

1. **Adaptive subspaces for few-shot learning,** in CVPR, 2020.
*C. Simon, P. Koniusz, R. Nock, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **DPGN: Distribution propagation graph network for few-shot learning,** in CVPR, 2020.
*L. Yang, L. Li, Z. Zhang, X. Zhou, E. Zhou, and Y. Liu.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_DPGN_Distribution_Propagation_Graph_Network_for_Few-Shot_Learning_CVPR_2020_paper_check.pdf)

1. **Few-shot learning via embedding adaptation with set-to-set functions,** in CVPR, 2020.
*H.-J. Ye, H. Hu, D.-C. Zhan, and F. Sha.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf)
[code](https://github.com/Sha-Lab/FEAT)

1. **DeepEMD: Few-shot image classification with differentiable earth mover's distance and structured classifiers,** in CVPR, 2020.
*C. Zhang, Y. Cai, G. Lin, and C. Shen.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.pdf)
[code](https://github.com/icoz69/DeepEMD)

1. **Few-shot text classification with distributional signatures,** in ICLR, 2020.
*Y. Bao, M. Wu, S. Chang, and R. Barzilay.*
[paper](https://openreview.net/pdf?id=H1emfT4twB)
[code](https://github.com/YujiaBao/Distributional-Signatures)

1. **Learning task-aware local representations for few-shot learning,** in IJCAI, 2020.
*C. Dong, W. Li, J. Huo, Z. Gu, and Y. Gao.*
[paper](https://www.ijcai.org/Proceedings/2020/0100.pdf)

1. **SimPropNet: Improved similarity propagation for few-shot image segmentation,** in IJCAI, 2020.
*S. Gairola, M. Hemani, A. Chopra, and B. Krishnamurthy.*
[paper](https://www.ijcai.org/Proceedings/2020/0080.pdf)

1. **Asymmetric distribution measure for few-shot learning,** in IJCAI, 2020.
*W. Li, L. Wang, J. Huo, Y. Shi, Y. Gao, and J. Luo.*
[paper](https://www.ijcai.org/Proceedings/2020/0409.pdf)

1. **Transductive relation-propagation network for few-shot learning,** in IJCAI, 2020.
*Y. Ma, S. Bai, S. An, W. Liu, A. Liu, X. Zhen, and X. Liu.*
[paper](https://www.ijcai.org/Proceedings/2020/0112.pdf)

1. **Weakly supervised few-shot object segmentation using co-attention with visual and semantic embeddings,** in IJCAI, 2020.
*M. Siam, N. Doraiswamy, B. N. Oreshkin, H. Yao, and M. Jägersand.*
[paper](https://www.ijcai.org/Proceedings/2020/0120.pdf)

1. **Few-shot learning on graphs via super-classes based on graph spectral measures,** in ICLR, 2020.
*J. Chauhan, D. Nathani, and M. Kaul.*
[paper](https://openreview.net/pdf?id=Bkeeca4Kvr)

1. **SGAP-Net: Semantic-guided attentive prototypes network for few-shot human-object interaction recognition,** in AAAI, 2020.
*Z. Ji, X. Liu, Y. Pang, and X. Li.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6764)

1. **One-shot image classification by learning to restore prototypes,** in AAAI, 2020.
*W. Xue, and W. Wang.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6130)

1. **Negative margin matters: Understanding margin in few-shot classification,** in ECCV, 2020.
*B. Liu, Y. Cao, Y. Lin, Q. Li, Z. Zhang, M. Long, and H. Hu.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490426.pdf)
[code](https://github.com/bl0/negative-margin.few-shot)

1. **Prototype rectification for few-shot learning,** in ECCV, 2020.
*J. Liu, L. Song, and Y. Qin.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460715.pdf)

1. **Rethinking few-shot image classification: A good embedding is all you need?,** in ECCV, 2020.
*Y. Tian, Y. Wang, D. Krishnan, J. B. Tenenbaum, and P. Isola.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590256.pdf)
[code](https://github.com/WangYueFt/rfs/)

1. **SEN: A novel feature normalization dissimilarity measure for prototypical few-shot learning networks,** in ECCV, 2020.
*V. N. Nguyen, S. Løkse, K. Wickstrøm, M. Kampffmeyer, D. Roverso, and R. Jenssen.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680120.pdf)

1. **TAFSSL: Task-adaptive feature sub-space learning for few-shot classification,** in ECCV, 2020.
*M. Lichtenstein, P. Sattigeri, R. Feris, R. Giryes, and L. Karlinsky.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520511.pdf)

1. **Attentive prototype few-shot learning with capsule network-based embedding,** in ECCV, 2020.
*F. Wu, J. S.Smith, W. Lu, C. Pang, and B. Zhang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730239.pdf)

1. **Embedding propagation: Smoother manifold for few-shot classification,** in ECCV, 2020.
*P. Rodríguez, I. Laradji, A. Drouin, and A. Lacoste.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710120.pdf)
[code](https://github.com/ElementAI/embedding-propagation)

1. **Laplacian regularized few-shot learning,** in ICML, 2020.
*I. M. Ziko, J. Dolz, E. Granger, and I. B. Ayed.*
[paper](http://proceedings.mlr.press/v119/ziko20a/ziko20a.pdf)
[code](https://github.com/imtiazziko/LaplacianShot)

1. **TAdaNet: Task-adaptive network for graph-enriched meta-learning,** in KDD, 2020.
*Q. Suo, i. Chou, W. Zhong, and A. Zhang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403230)

1. **Concept learners for few-shot learning,** in ICLR, 2021.
*K. Cao, M. Brbic, and J. Leskovec.*
[paper](https://openreview.net/pdf?id=eJIJF3-LoZO)

1. **Reinforced attention for few-shot learning and beyond,** in CVPR, 2021.
*J. Hong, P. Fang, W. Li, T. Zhang, C. Simon, M. Harandi, and L. Petersson.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Reinforced_Attention_for_Few-Shot_Learning_and_Beyond_CVPR_2021_paper.pdf)

1. **Mutual CRF-GNN for few-shot learning,** in CVPR, 2021.
*S. Tang, D. Chen, L. Bai, K. Liu, Y. Ge, and W. Ouyang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Mutual_CRF-GNN_for_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Few-shot classification with feature map reconstruction networks,** in CVPR, 2021.
*D. Wertheimer, L. Tang, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.pdf)
[code](https://github.com/Tsingularity/FRN)

1. **ECKPN: Explicit class knowledge propagation network for transductive few-shot learning,** in CVPR, 2021.
*C. Chen, X. Yang, C. Xu, X. Huang, and Z. Ma.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_ECKPN_Explicit_Class_Knowledge_Propagation_Network_for_Transductive_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Exploring complementary strengths of invariant and equivariant representations for few-shot learning,** in CVPR, 2021.
*M. N. Rizve, S. Khan, F. S. Khan, and M. Shah.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Rizve_Exploring_Complementary_Strengths_of_Invariant_and_Equivariant_Representations_for_Few-Shot_CVPR_2021_paper.pdf)

1. **Rethinking class relations: Absolute-relative supervised and unsupervised few-shot learning,** in CVPR, 2021.
*H. Zhang, P. Koniusz, S. Jian, H. Li, and P. H. S. Torr.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Rethinking_Class_Relations_Absolute-Relative_Supervised_and_Unsupervised_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Unsupervised embedding adaptation via early-stage feature reconstruction for few-shot classification,** in ICML, 2021.
*D. H. Lee, and S. Chung.*
[paper](http://proceedings.mlr.press/v139/lee21d/lee21d.pdf)
[code](https://github.com/movinghoon/ESFR)

1. **Learning a few-shot embedding model with contrastive learning,** in AAAI, 2021.
*C. Liu, Y. Fu, C. Xu, S. Yang,  J. Li, C. Wang, and L. Zhang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17047/16854)

1. **Looking wider for better adaptive representation in few-shot learning,** in AAAI, 2021.
*J. Zhao, Y. Yang, X. Lin, J. Yang, and L. He.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17311/17118)

1. **Tailoring embedding function to heterogeneous few-shot tasks by global and local feature adaptors,** in AAAI, 2021.
*S. Lu, H. Ye, and D.-C. Zhan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17063/16870)

1. **Knowledge guided metric learning for few-shot text classification,** in NAACL-HLT, 2021.
*D. Sui, Y. Chen, B. Mao, D. Qiu, K. Liu, and J. Zhao.*
[paper](https://aclanthology.org/2021.naacl-main.261.pdf)

1. **Mixture-based feature space learning for few-shot image classification,** in ICCV, 2021.
*A. Afrasiyabi, J. Lalonde, and C. Gagné.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Afrasiyabi_Mixture-Based_Feature_Space_Learning_for_Few-Shot_Image_Classification_ICCV_2021_paper.pdf)

1. **Z-score normalization, hubness, and few-shot learning,** in ICCV, 2021.
*N. Fei, Y. Gao, Z. Lu, and T. Xiang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Fei_Z-Score_Normalization_Hubness_and_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Relational embedding for few-shot classification,** in ICCV, 2021.
*D. Kang, H. Kwon, J. Min, and M. Cho.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kang_Relational_Embedding_for_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/dahyun-kang/renet)

1. **Transductive few-shot classification on the oblique manifold,** in ICCV, 2021.
*G. Qi, H. Yu, Z. Lu, and S. Li.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Qi_Transductive_Few-Shot_Classification_on_the_Oblique_Manifold_ICCV_2021_paper.pdf)
[code](https://github.com/GuodongQi/FSL-OM)

1. **Curvature generation in curved spaces for few-shot learning,** in ICCV, 2021.
*Z. Gao, Y. Wu, Y. Jia, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Curvature_Generation_in_Curved_Spaces_for_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **On episodes, prototypical networks, and few-shot learning,** in NeurIPS, 2021.
*S. Laenen, and L. Bertinetto.*
[paper](https://proceedings.neurips.cc/paper/2021/file/cdfa4c42f465a5a66871587c69fcfa34-Paper.pdf)

1. **Few-shot learning as cluster-induced voronoi diagrams: A geometric approach,** in ICLR, 2022.
*C. Ma, Z. Huang, M. Gao, and J. Xu.*
[paper](https://openreview.net/pdf?id=6kCiVaoQdx9)
[code](https://github.com/horsepurve/DeepVoro)

1. **Few-shot learning with siamese networks and label tuning,** in ACL, 2022.
*T. Müller, G. Pérez-Torró, and M. Franco-Salvador.*
[paper](https://aclanthology.org/2022.acl-long.584.pdf)
[code](https://github.com/symanto-research/few-shot-learning-label-tuning)

1. **Learning to affiliate: Mutual centralized learning for few-shot classification,** in CVPR, 2022.
*Y. Liu, W. Zhang, C. Xiang, T. Zheng, D. Cai, and X. He.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Learning_To_Affiliate_Mutual_Centralized_Learning_for_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **Matching feature sets for few-shot image classification,** in CVPR, 2022.
*A. Afrasiyabi, H. Larochelle, J. Lalonde, and C. Gagné.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Afrasiyabi_Matching_Feature_Sets_for_Few-Shot_Image_Classification_CVPR_2022_paper.pdf)
[code](https://lvsn.github.io/SetFeat/)

1. **Joint distribution matters: Deep Brownian distance covariance for few-shot classification,** in CVPR, 2022.
*J. Xie, F. Long, J. Lv, Q. Wang, and P. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Joint_Distribution_Matters_Deep_Brownian_Distance_Covariance_for_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **CAD: Co-adapting discriminative features for improved few-shot classification,** in CVPR, 2022.
*P. Chikontwe, S. Kim, and S. H. Park.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chikontwe_CAD_Co-Adapting_Discriminative_Features_for_Improved_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **Ranking distance calibration for cross-domain few-shot learning,** in CVPR, 2022.
*P. Li, S. Gong, C. Wang, and Y. Fu.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Ranking_Distance_Calibration_for_Cross-Domain_Few-Shot_Learning_CVPR_2022_paper.pdf)

1. **EASE: Unsupervised discriminant subspace learning for transductive few-shot learning,** in CVPR, 2022.
*H. Zhu, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_EASE_Unsupervised_Discriminant_Subspace_Learning_for_Transductive_Few-Shot_Learning_CVPR_2022_paper.pdf)
[code](https://github.com/allenhaozhu/EASE)

1. **Cross-domain few-shot learning with task-specific adapters,** in CVPR, 2022.
*W. Li, X. Liu, and H. Bilen.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Few-Shot_Learning_With_Task-Specific_Adapters_CVPR_2022_paper.pdf)
[code](https://github.com/VICO-UoE/URL)

1. **Hyperbolic knowledge transfer with class hierarchy for few-shot learning,** in IJCAI, 2022.
*B. Zhang, H. Jiang, S. Feng, X. Li, Y. Ye, and R. Ye.*
[paper](https://www.ijcai.org/proceedings/2022/0517.pdf)

1. **Better embedding and more shots for few-shot learning,** in IJCAI, 2022.
*Z. Chi, Z. Wang, M. Yang, W. Guo, and X. Xu.*
[paper](https://www.ijcai.org/proceedings/2022/0398.pdf)

1. **A closer look at prototype classifier for few-shot image classification,** in NeurIPS, 2022.
*M. Hou, and I. Sato.*
[paper](https://openreview.net/pdf?id=U_hOegGGglw)

1. **Rethinking generalization in few-shot classification,** in NeurIPS, 2022.
*M. Hiller, R. Ma, M. Harandi, and T. Drummond.*
[paper](https://openreview.net/pdf?id=p_g2nHlMus)
[code](https://github.com/mrkshllr/FewTURE)

1. **DMN4: Few-shot learning via discriminative mutual nearest neighbor neural network,** in AAAI, 2022.
*Y. Liu, T. Zheng, J. Song, D. Cai, and X. He.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20076/19835)

1. **Hybrid graph neural networks for few-shot learning,** in AAAI, 2022.
*T. Yu, S. He, Y.-Z. Song, and T. Xiang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20226/19985)
[code](https://github.com/TianyuanYu/HGNN)

1. **Adaptive poincaré point to set distance for few-shot classification,** in AAAI, 2022.
*R. Ma, P. Fang, T. Drummond, and M. Harandi.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20087/19846)

1. **Hubs and hyperspheres: Reducing hubness and improving transductive few-shot learning with hyperspherical embeddings,** in CVPR, 2023.
*D. J. Trosten, R. Chakraborty, S. Løkse, K. K. Wickstrøm, R. Jenssen, and M. C. Kampffmeyer.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Trosten_Hubs_and_Hyperspheres_Reducing_Hubness_and_Improving_Transductive_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/uitml/noHub)

1. **Revisiting prototypical network for cross domain few-shot learning,** in CVPR, 2023.
*F. Zhou, P. Wang, L. Zhang, W. Wei, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Revisiting_Prototypical_Network_for_Cross_Domain_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/NWPUZhoufei/LDP-Net)

1. **Transductive few-shot learning with prototype-based label propagation by iterative graph refinement,** in CVPR, 2023.
*H. Zhu, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Transductive_Few-Shot_Learning_With_Prototype-Based_Label_Propagation_by_Iterative_Graph_CVPR_2023_paper.pdf)
[code](https://github.com/allenhaozhu/protoLP)

1. **Few-shot classification via ensemble learning with multi-order statistics,** in IJCAI, 2023.
*S. Yang, F. Liu, D. Chen, and J. Zhou.*
[paper](https://www.ijcai.org/proceedings/2023/0181.pdf)

1. **Few-sample feature selection via feature manifold learning,** in ICML, 2023.
*D. Cohen, T. Shnitzer, Y. Kluger, and R. Talmon.*
[paper](https://proceedings.mlr.press/v202/cohen23b/cohen23b.pdf)
[code](https://github.com/DavidCohen2/ManiFeSt)

1. **Interval bound interpolation for few-shot learning with few tasks,** in ICML, 2023.
*S. Datta, S. S. Mullick, A. Chakrabarty, and S. Das.*
[paper](https://proceedings.mlr.press/v202/datta23a/datta23a.pdf)
[code](https://github.com/SankhaSubhra/maml-ibp-ibi)

1. **A closer look at few-shot classification again,** in ICML, 2023.
*X. Luo, H. Wu, J. Zhang, L. Gao, J. Xu, and J. Song.*
[paper](https://proceedings.mlr.press/v202/luo23e/luo23e.pdf)
[code](https://github.com/Frankluox/CloserLookAgainFewShot)

1. **TART: Improved few-shot text classification using task-adaptive reference transformation,** in ACL, 2023.
*S. Lei, X. Zhang, J. He, F. Chen, and C.-T. Lu.*
[paper](https://aclanthology.org/2023.acl-long.617.pdf)
[code](https://github.com/slei109/TART)

51. **Class-aware patch embedding adaptation for few-shot image classification,** in ICCV, 2023.
*F. Hao, F. He, L. Liu, F. Wu, D. Tao, and J. Cheng.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Hao_Class-Aware_Patch_Embedding_Adaptation_for_Few-Shot_Image_Classification_ICCV_2023_paper.pdf)
[code](https://github.com/FushengHao/CPEA)

53. **Frequency guidance matters in few-shot learning,** in ICCV, 2023.
*H. Cheng, S. Yang, J. T. Zhou, L. Guo, and B. Wen.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_Frequency_Guidance_Matters_in_Few-Shot_Learning_ICCV_2023_paper.pdf)

14. **Prototypes-oriented transductive few-shot learning with conditional transport,** in ICCV, 2023.
*L. Tian, J. Feng, X. Chai, W. Chen, L. Wang, X. Liu, and B. Chen.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Tian_Prototypes-oriented_Transductive_Few-shot_Learning_with_Conditional_Transport_ICCV_2023_paper.pdf)
[code](https://github.com/RashLog/PUTM)

7. **Understanding few-shot learning: Measuring task relatedness and adaptation difficulty via attributes,** in NeurIPS, 2023.
*M. Hu, H. Chang, Z. Guo, B. Ma, S. Shan, and X. Chen.*
[paper](https://openreview.net/attachment?id=Pvgxecj5aS&name=pdf)
[code](https://github.com/hu-my/TaskAttributeDistance)

20. **DiffKendall: A novel approach for few-shot learning with differentiable Kendall's rank correlation,** in NeurIPS, 2023.
*K. Zheng, H. Zhang, and W. Huang.*
[paper](https://openreview.net/attachment?id=SVUQX1W7RL&name=pdf)

### Learning with External Memory

1. **Meta-learning with memory-augmented neural networks,** in ICML, 2016.
*A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap.*
[paper](http://proceedings.mlr.press/v48/santoro16.pdf)

1. **Few-shot object recognition from machine-labeled web images,** in CVPR, 2017.
*Z. Xu, L. Zhu, and Y. Yang.*
[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Few-Shot_Object_Recognition_CVPR_2017_paper.pdf)

1. **Learning to remember rare events,** in ICLR, 2017.
*Ł. Kaiser, O. Nachum, A. Roy, and S. Bengio.*
[paper](https://openreview.net/forum?id=SJTQLdqlg)

1. **Meta networks,** in ICML, 2017.
*T. Munkhdalai and H. Yu.* 
[paper](http://proceedings.mlr.press/v70/munkhdalai17a/munkhdalai17a.pdf)

1. **Memory matching networks for one-shot image recognition,** in CVPR, 2018.
*Q. Cai, Y. Pan, T. Yao, C. Yan, and T. Mei.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Memory_Matching_Networks_CVPR_2018_paper.pdf)

1. **Compound memory networks for few-shot video classification,** in ECCV, 2018.
*L. Zhu and Y. Yang.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.pdf)

1. **Memory, show the way: Memory based few shot word representation learning,** in EMNLP, 2018.
*J. Sun, S. Wang, and C. Zong.*
[paper](https://www.aclweb.org/anthology/D18-1173.pdf)

1. **Rapid adaptation with conditionally shifted neurons,** in ICML, 2018.
*T. Munkhdalai, X. Yuan, S. Mehri, and A. Trischler.*
[paper](http://proceedings.mlr.press/v80/munkhdalai18a/munkhdalai18a.pdf)

1. **Adaptive posterior learning: Few-shot learning with a surprise-based memory module,** in ICLR, 2019. 
*T. Ramalho and M. Garnelo.*
[paper](https://openreview.net/forum?id=ByeSdsC9Km)
[code](https://github.com/cogentlabs/apl)

1. **Coloring with limited data: Few-shot colorization via memory augmented networks,** in CVPR, 2019. 
*S. Yoo, H. Bahng, S. Chung, J. Lee, J. Chang, and J. Choo.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Coloring_With_Limited_Data_Few-Shot_Colorization_via_Memory_Augmented_Networks_CVPR_2019_paper.pdf)

1. **ACMM: Aligned cross-modal memory for few-shot image and sentence matching,** in ICCV, 2019. 
*Y. Huang, and L. Wang.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_ACMM_Aligned_Cross-Modal_Memory_for_Few-Shot_Image_and_Sentence_Matching_ICCV_2019_paper.pdf)

1. **Dynamic memory induction networks for few-shot text classification,** in ACL, 2020.
*R. Geng, B. Li, Y. Li, J. Sun, and X. Zhu.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.102.pdf)

1. **Few-shot visual learning with contextual memory and fine-grained calibration,** in IJCAI, 2020.
*Y. Ma, W. Liu, S. Bai, Q. Zhang, A. Liu, W. Chen, and X. Liu.*
[paper](https://www.ijcai.org/Proceedings/2020/0113.pdf)

1. **Learn from concepts: Towards the purified memory for few-shot learning,** in IJCAI, 2021.
*X. Liu, X. Tian, S. Lin, Y. Qu, L. Ma, W. Yuan, Z. Zhang, and Y. Xie.*
[paper](https://www.ijcai.org/proceedings/2021/0123.pdf)

1. **Prototype memory and attention mechanisms for few shot image generation,** in ICLR, 2022.
*T. Li, Z. Li, A. Luo, H. Rockwell, A. B. Farimani, and T. S. Lee.*
[paper](https://openreview.net/pdf?id=lY0-7bj0Vfz)
[code](https://github.com/Crazy-Jack/MoCA_release)

1. **Hierarchical variational memory for few-shot learning across domains,** in ICLR, 2022.
*Y. Du, X. Zhen, L. Shao, and C. G. M. Snoek.*
[paper](https://openreview.net/pdf?id=i3RI65sR7N)
[code](https://github.com/YDU-uva/HierMemory)

1. **Remember the difference: Cross-domain few-shot semantic segmentation via meta-memory transfer,** in CVPR, 2022.
*W. Wang, L. Duan, Y. Wang, Q. En, J. Fan, and Z. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Remember_the_Difference_Cross-Domain_Few-Shot_Semantic_Segmentation_via_Meta-Memory_Transfer_CVPR_2022_paper.pdf)

1. **Consistent prototype learning for few-shot continual relation extraction,** in ACL, 2023.
*X. Chen, H. Wu, and X. Shi.*
[paper](https://aclanthology.org/2023.acl-long.409.pdf)
[code](https://github.com/XiudiChen/ConPL)

4. **Few-shot generation via recalling brain-inspired episodic-semantic memory,** in NeurIPS, 2023.
*Z. Duan, L. Zhiyi, C. Wang, B. Chen, B. An, and M. Zhou.*
[paper](https://openreview.net/attachment?id=dxPcdEeQk9&name=pdf)


### Generative Modeling

1. **One-shot learning of object categories,** TPAMI, 2006.
*L. Fei-Fei, R. Fergus, and P. Perona.*
[paper](http://vision.stanford.edu/documents/Fei-FeiFergusPerona2006.pdf)

1. **Learning to learn with compound HD models,** in NeurIPS, 2011.
*A. Torralba, J. B. Tenenbaum, and R. R. Salakhutdinov.*
[paper](https://papers.nips.cc/paper/4474-learning-to-learn-with-compound-hd-models.pdf)

1. **One-shot learning with a hierarchical nonparametric bayesian model,** in ICML Workshop on Unsupervised and Transfer Learning, 2012.
*R. Salakhutdinov, J. Tenenbaum, and A. Torralba.*
[paper](http://proceedings.mlr.press/v27/salakhutdinov12a/salakhutdinov12a.pdf)

1. **Human-level concept learning through probabilistic program induction,** Science, 2015.
*B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum.*
[paper](https://web.mit.edu/cocosci/Papers/Science-2015-Lake-1332-8.pdf)

1. **One-shot generalization in deep generative models,** in ICML, 2016.
*D. Rezende, I. Danihelka, K. Gregor, and D. Wierstra.*
[paper](https://arxiv.org/pdf/1603.05106)

1. **One-shot video object segmentation,** in CVPR, 2017.
*S. Caelles, K.-K. Maninis, J. Pont-Tuset, L. Leal-Taixé, D. Cremers,
and L. Van Gool.*
[paper](http://zpascal.net/cvpr2017/Caelles_One-Shot_Video_Object_CVPR_2017_paper.pdf)

1. **Towards a neural statistician,** in ICLR, 2017.
*H. Edwards and A. Storkey.*
[paper](https://openreview.net/forum?id=HJDBUF5le)

1. **Extending a parser to distant domains using a few dozen partially annotated examples,** in ACL, 2018.
*V. Joshi, M. Peters, and M. Hopkins.*
[paper](https://www.aclweb.org/anthology/P18-1110.pdf)

1. **MetaGAN: An adversarial approach to few-shot learning,** in NeurIPS, 2018.
*R. Zhang, T. Che, Z. Ghahramani, Y. Bengio, and Y. Song.*
[paper](https://papers.nips.cc/paper/7504-metagan-an-adversarial-approach-to-few-shot-learning.pdf)

1. **Few-shot autoregressive density estimation: Towards learning to learn distributions,** in ICLR, 2018.
*S. Reed, Y. Chen, T. Paine, A. van den Oord, S. M. A. Eslami, D. Rezende, O. Vinyals, and N. de Freitas.* 
[paper](https://openreview.net/forum?id=r1wEFyWCW)

1. **The variational homoencoder: Learning to learn high capacity generative models from few examples,** in UAI, 2018.
*L. B. Hewitt, M. I. Nye, A. Gane, T. Jaakkola, and J. B. Tenenbaum.*
[paper](http://auai.org/uai2018/proceedings/papers/351.pdf)

1. **Meta-learning probabilistic inference for prediction,** in ICLR, 2019.
*J. Gordon, J. Bronskill, M. Bauer, S. Nowozin, and R. Turner.*
[paper](https://openreview.net/forum?id=HkxStoC5F7)

1. **Variational prototyping-encoder: One-shot learning with prototypical images,** in CVPR, 2019.
*J. Kim, T.-H. Oh, S. Lee, F. Pan, and I. S. Kweon.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Variational_Prototyping-Encoder_One-Shot_Learning_With_Prototypical_Images_CVPR_2019_paper.pdf)
[code](https://github.com/mibastro/VPE)

1. **Variational few-shot learning,** in ICCV, 2019.
*J. Zhang, C. Zhao, B. Ni, M. Xu, and X. Yang.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Variational_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **Infinite mixture prototypes for few-shot learning,** in ICML, 2019.
*K. Allen, E. Shelhamer, H. Shin, and J. Tenenbaum.*
[paper](http://proceedings.mlr.press/v97/allen19b/allen19b.pdf)

1. **Dual variational generation for low shot heterogeneous face recognition,** in NeurIPS, 2019.
*C. Fu, X. Wu, Y. Hu, H. Huang, and R. He.*
[paper](https://papers.nips.cc/paper/8535-dual-variational-generation-for-low-shot-heterogeneous-face-recognition.pdf)

1. **Bayesian meta sampling for fast uncertainty adaptation,** in ICLR, 2020.
*Z. Wang, Y. Zhao, P. Yu, R. Zhang, and C. Chen.*
[paper](https://openreview.net/pdf?id=Bkxv90EKPB)

1. **Empirical Bayes transductive meta-learning with synthetic gradients,** in ICLR, 2020.
*S. X. Hu, P. G. Moreno, Y. Xiao, X. Shen, G. Obozinski, N. D. Lawrence, and A. C. Damianou.*
[paper](https://openreview.net/pdf?id=Hkg-xgrYvH)

1. **Few-shot relation extraction via bayesian meta-learning on relation graphs,** in ICML, 2020.
*M. Qu, T. Gao, L. A. C. Xhonneux, and J. Tang.*
[paper](http://proceedings.mlr.press/v119/qu20a/qu20a.pdf)
[code](https://github.com/DeepGraphLearning/FewShotRE)

1. **Interventional few-shot learning,** in NeurIPS, 2020.
*Z. Yue, H. Zhang, Q. Sun, and X. Hua.*
[paper](https://proceedings.neurips.cc/paper/2020/file/1cc8a8ea51cd0adddf5dab504a285915-Paper.pdf)
[code](https://github.com/yue-zhongqi/ifsl)

1. **Bayesian few-shot classification with one-vs-each pólya-gamma augmented gaussian processes,** in ICLR, 2021.
*J. Snell, and R. Zemel.*
[paper](https://openreview.net/pdf?id=lgNx56yZh8a)

1. **Few-shot Bayesian optimization with deep kernel surrogates,** in ICLR, 2021.
*M. Wistuba, and J. Grabocka.*
[paper](https://openreview.net/pdf?id=bJxgv5C3sYc)

1. **A hierarchical transformation-discriminating generative model for few shot anomaly detection,** in ICCV, 2021.
*S. Sheynin, S. Benaim, and L. Wolf.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sheynin_A_Hierarchical_Transformation-Discriminating_Generative_Model_for_Few_Shot_Anomaly_Detection_ICCV_2021_paper.pdf)

1. **Reinforced few-shot acquisition function learning for Bayesian optimization,** in NeurIPS, 2021.
*B. Hsieh, P. Hsieh, and X. Liu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/3fab5890d8113d0b5a4178201dc842ad-Paper.pdf)

1. **GanOrCon: Are generative models useful for few-shot segmentation?,** in CVPR, 2022.
*O. Saha, Z. Cheng, and S. Maji.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Saha_GanOrCon_Are_Generative_Models_Useful_for_Few-Shot_Segmentation_CVPR_2022_paper.pdf)

1. **Few shot generative model adaption via relaxed spatial structural alignment,** in CVPR, 2022.
*J. Xiao, L. Li, C. Wang, Z. Zha, and Q. Huang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xiao_Few_Shot_Generative_Model_Adaption_via_Relaxed_Spatial_Structural_Alignment_CVPR_2022_paper.pdf)

1. **SCHA-VAE: Hierarchical context aggregation for few-shot generation,** in ICML, 2022.
*G. Giannone, and O. Winther.*
[paper](https://proceedings.mlr.press/v162/giannone22a/giannone22a.pdf)
[code](https://github.com/georgosgeorgos/hierarchical-few-shot-generative-models)

1. **Diversity vs. Recognizability: Human-like generalization in one-shot generative models,** in NeurIPS, 2022.
*V. Boutin, L. Singhal, X. Thomas, and T. Serre.*
[paper](https://openreview.net/pdf?id=DVfZKXSFW5m)
[code](https://github.com/serre-lab/diversity_vs_recognizability)

1. **Generalized one-shot domain adaptation of generative adversarial networks,** in NeurIPS, 2022.
*Z. Zhang, Y. Liu, C. Han, T. Guo, T. Yao, and T. Mei.*
[paper](https://openreview.net/pdf?id=mfxq7BrMfga)
[code](https://github.com/zhangzc21/Generalized-One-shot-GAN-Adaptation)

1. **Towards diverse and faithful one-shot adaption of generative adversarial networks,** in NeurIPS, 2022.
*Y. Zhang, m. Yao, Y. Wei, Z. Ji, J. Bai, and W. Zuo.*
[paper](https://openreview.net/pdf?id=IXoHxXIGpyV)
[code](https://github.com/YBYBZhang/DiFa)

1. **Few-shot cross-domain image generation via inference-time latent-code learning,** in ICLR, 2023.
*A. K. Mondal, P. Tiwary, P. Singla, and P. AP.*
[paper](https://openreview.net/pdf?id=sCYXJr3QJM8)
[code](https://github.com/arnabkmondal/GenDA)

1. **Adaptive IMLE for few-shot pretraining-free generative modelling,** in ICML, 2023.
*M. Aghabozorgi, S. Peng, and K. Li.*
[paper](https://proceedings.mlr.press/v202/aghabozorgi23a/aghabozorgi23a.pdf)
[code](https://github.com/mehranagh20/AdaIMLE)

1. **Diversity-enhancing generative network for few-shot hypothesis adaptation,** in ICML, 2023.
*R. Dong, F. Liu, H. Chi, T. Liu, M. Gong, G. Niu, M. Sugiyama, and B. Han.*
[paper](https://proceedings.mlr.press/v202/dong23d/dong23d.pdf)

1. **MetaModulation: Learning variational feature hierarchies for few-shot learning with fewer tasks,** in ICML, 2023.
*W. Sun, Y. Du, X. Zhen, F. Wang, L. Wang, and C. G. M. Snoek.*
[paper](https://proceedings.mlr.press/v202/sun23b/sun23b.pdf)
[code](https://github.com/lmsdss/MetaModulation)

6. **Revisiting logistic-softmax likelihood in bayesian meta-learning for few-shot classification,** in NeurIPS, 2023.
*T. Ke, H. Cao, Z. Ling, and F. Zhou.*
[paper](https://openreview.net/attachment?id=Z1W0u3Cr74&name=pdf)
[code](https://github.com/keanson/revisit-logistic-softmax)

1. **Human-like few-shot learning via bayesian reasoning over natural language,** in NeurIPS, 2023.
*and K. Ellis.*
[paper](https://openreview.net/attachment?id=dVnhdm9MIg&name=pdf)
[code](https://github.com/ellisk42/humanlike_fewshot_learning)

## [Algorithm](#content)

### Refining Existing Parameters

1. **Cross-generalization: Learning novel classes from a single example by feature replacement,** in CVPR, 2005. 
*E. Bart and S. Ullman.*
[paper](http://www.inf.tu-dresden.de/content/institutes/ki/is/HS_SS08_Papers/BartUllmanCVPR05.pdf)

1. **One-shot adaptation of supervised deep convolutional models,** in ICLR, 2013.
*J. Hoffman, E. Tzeng, J. Donahue, Y. Jia, K. Saenko, and T. Darrell.*
[paper](https://openreview.net/forum?id=tPCrkaLa9Y5ld)

1. **Learning to learn: Model regression networks for easy small sample learning,** in ECCV, 2016.
*Y.-X. Wang and M. Hebert.*
[paper](https://ri.cmu.edu/pub_files/2016/10/yuxiongw_eccv16_learntolearn.pdf)

1. **Learning from small sample sets by combining unsupervised meta-training with CNNs,** in NeurIPS, 2016.
*Y.-X. Wang and M. Hebert.*
[paper](https://papers.nips.cc/paper/6408-learning-from-small-sample-sets-by-combining-unsupervised-meta-training-with-cnns)

1. **Efficient k-shot learning with regularized deep networks,** in AAAI, 2018.
*D. Yoo, H. Fan, V. N. Boddeti, and K. M. Kitani.*
[paper](https://arxiv.org/abs/1710.02277)

1. **CLEAR: Cumulative learning for one-shot one-class image recognition,** in CVPR, 2018.
*J. Kozerawski and M. Turk.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kozerawski_CLEAR_Cumulative_LEARning_CVPR_2018_paper.pdf)

1. **Learning structure and strength of CNN filters for small sample size training,** in CVPR, 2018. 
*R. Keshari, M. Vatsa, R. Singh, and A. Noore.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Keshari_Learning_Structure_and_CVPR_2018_paper.pdf)

1. **Dynamic few-shot visual learning without forgetting,** in CVPR, 2018.
*S. Gidaris and N. Komodakis.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gidaris_Dynamic_Few-Shot_Visual_CVPR_2018_paper.pdf)
[code](https://github.com/gidariss/FewShotWithoutForgetting)

1. **Low-shot learning with imprinted weights,** in CVPR, 2018.
*H. Qi, M. Brown, and D. G. Lowe.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf)

1. **Neural voice cloning with a few samples,** in NeurIPS, 2018.
*S. Arik, J. Chen, K. Peng, W. Ping, and Y. Zhou.*
[paper](https://papers.nips.cc/paper/8206-neural-voice-cloning-with-a-few-samples.pdf)

1. **Text classification with few examples using controlled generalization,** in NAACL-HLT, 2019.
*A. Mahabal, J. Baldridge, B. K. Ayan, V. Perot, and D. Roth.* 
[paper](https://www.aclweb.org/anthology/N19-1319.pdf)

1. **Low shot box correction for weakly supervised object detection,** in IJCAI, 2019.
*T. Pan, B. Wang, G. Ding, J. Han, and J. Yong.*
[paper](https://www.ijcai.org/Proceedings/2019/0125.pdf)

1. **Diversity with cooperation: Ensemble methods for few-shot classification,** in ICCV, 2019.
*N. Dvornik, C. Schmid, and J. Mairal.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dvornik_Diversity_With_Cooperation_Ensemble_Methods_for_Few-Shot_Classification_ICCV_2019_paper.pdf)

1. **Few-shot image recognition with knowledge transfer,** in ICCV, 2019.
*Z. Peng, Z. Li, J. Zhang, Y. Li, G.-J. Qi, and J. Tang.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Few-Shot_Image_Recognition_With_Knowledge_Transfer_ICCV_2019_paper.pdf)

1. **Generating classification weights with gnn denoising autoencoders for few-shot learning,** in CVPR, 2019.
*S. Gidaris, and N. Komodakis.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gidaris_Generating_Classification_Weights_With_GNN_Denoising_Autoencoders_for_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/gidariss/wDAE_GNN_FewShot)

1. **Dense classification and implanting for few-shot learning,** in CVPR, 2019.
*Y. Lifchitz, Y. Avrithis, S. Picard, and A. Bursuc.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lifchitz_Dense_Classification_and_Implanting_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

1. **Few-shot adaptive faster R-CNN,** in CVPR, 2019.
*T. Wang, X. Zhang, L. Yuan, and J. Feng.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Few-Shot_Adaptive_Faster_R-CNN_CVPR_2019_paper.pdf)

1. **TransMatch: A transfer-learning scheme for semi-supervised few-shot learning,** in CVPR, 2020.
*Z. Yu, L. Chen, Z. Cheng, and J. Luo.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_TransMatch_A_Transfer-Learning_Scheme_for_Semi-Supervised_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Learning to select base classes for few-shot classification,** in CVPR, 2020.
*L. Zhou, P. Cui, X. Jia, S. Yang, and Q. Tian.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Learning_to_Select_Base_Classes_for_Few-Shot_Classification_CVPR_2020_paper.pdf)

1. **Few-shot NLG with pre-trained language model,** in ACL, 2020.
*Z. Chen, H. Eavani, W. Chen, Y. Liu, and W. Y. Wang.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.18.pdf)
[code](https://github.com/czyssrs/Few-Shot-NLG)

1. **Span-ConveRT: Few-shot span extraction for dialog with pretrained conversational representations,** in ACL, 2020.
*S. Coope, T. Farghly, D. Gerz, I. Vulic, and M. Henderson.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.11.pdf)

1. **Structural supervision improves few-shot learning and syntactic generalization in neural language models,** in EMNLP, 2020.
*E. Wilcox, P. Qian, R. Futrell, R. Kohita, R. Levy, and M. Ballesteros.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.375.pdf)
[code](https://github.com/wilcoxeg/fsl_invar)

1. **A baseline for few-shot image classification,** in ICLR, 2020.
*G. S. Dhillon, P. Chaudhari, A. Ravichandran, and S. Soatto.*
[paper](https://openreview.net/pdf?id=rylXBkrYDS)

1. **Cross-domain few-shot classification via learned feature-wise transformation,** in ICLR, 2020.
*H. Tseng, H. Lee, J. Huang, and M. Yang.*
[paper](https://openreview.net/pdf?id=SJl5Np4tPr)
[code](https://github.com/hytseng0509/CrossDomainFewShot)

1. **Graph few-shot learning via knowledge transfer,** in AAAI, 2020.
*H. Yao, C. Zhang, Y. Wei, M. Jiang, S. Wang, J. Huang, N. V. Chawla, and Z. Li.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6142)

1. **Context-Transformer: Tackling object confusion for few-shot detection,** in AAAI, 2020.
*Z. Yang, Y. Wang, X. Chen, J. Liu, and Y. Qiao.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6957)

1. **A broader study of cross-domain few-shot learning,** in ECCV, 2020.
*Y. Guo, N. C. Codella, L. Karlinsky, J. V. Codella, J. R. Smith, K. Saenko, T. Rosing, and R. Feris.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf)
[code](https://github.com/IBM/cdfsl-benchmark)

1. **Selecting relevant features from a multi-domain representation for few-shot classification,** in ECCV, 2020.
*N. Dvornik, C. Schmid, and J. Mairal.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550766.pdf)
[code](https://github.com/dvornikita/SUR)

1. **Prototype completion with primitive knowledge for few-shot learning,** in CVPR, 2021.
*B. Zhang, X. Li, Y. Ye, Z. Huang, and L. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Prototype_Completion_With_Primitive_Knowledge_for_Few-Shot_Learning_CVPR_2021_paper.pdf)
[code](https://github.com/zhangbq-research/Prototype_Completion_for_FSL)

1. **Partial is better than all: Revisiting fine-tuning strategy for few-shot learning,** in AAAI, 2021.
*Z. Shen, Z. Liu, J. Qin, M. Savvides, and K.-T. Cheng.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17155/16962)

1. **PTN: A poisson transfer network for semi-supervised few-shot learning,** in AAAI, 2021.
*H. Huang, J. Zhang, J. Zhang, Q. Wu, and C. Xu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16252/16059)

1. **A universal representation transformer layer for few-shot image classification,** in ICLR, 2021.
*L. Liu, W. L. Hamilton, G. Long, J. Jiang, and H. Larochelle.*
[paper](https://openreview.net/pdf?id=04cII6MumYV)

1. **Making pre-trained language models better few-shot learners,** in ACL-IJCNLP, 2021.
*T. Gao, A. Fisch, and D. Chen.*
[paper](https://aclanthology.org/2021.acl-long.295.pdf)
[code](https://github.com/princeton-nlp/LM-BFF)

1. **Self-supervised network evolution for few-shot classification,** in IJCAI, 2021.
*X. Tang, Z. Teng, B. Zhang, and J. Fan.*
[paper](https://www.ijcai.org/proceedings/2021/0419.pdf)

1. **Calibrate before use: Improving few-shot performance of language models,** in ICML, 2021.
*Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh.*
[paper](http://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)
[code](https://www.github.com/tonyzhaozh/few-shot-learning)

1. **Language models are few-shot learners,** in NeurIPS, 2020.
*T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei.*
[paper](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

1. **It's not just size that matters: Small language models are also few-shot learners,** in NAACL-HLT, 2021.
*T. Schick, and H. Schütze.*
[paper](https://aclanthology.org/2021.naacl-main.185.pdf)
[code](https://github.com/timoschick/pet)

1. **Self-training improves pre-training for few-shot learning in task-oriented dialog systems,** in EMNLP, 2021.
*F. Mi, W. Zhou, L. Kong, F. Cai, M. Huang, and B. Faltings.*
[paper](https://aclanthology.org/2021.emnlp-main.142.pdf)

1. **Few-shot intent detection via contrastive pre-training and fine-tuning,** in EMNLP, 2021.
*J. Zhang, T. Bui, S. Yoon, X. Chen, Z. Liu, C. Xia, Q. H. Tran, W. Chang, and P. S. Yu.*
[paper](https://aclanthology.org/2021.emnlp-main.144.pdf)
[code](https://github.com/jianguoz/Few-Shot-Intent-Detection)

1. **Avoiding inference heuristics in few-shot prompt-based finetuning,** in EMNLP, 2021.
*P. A. Utama, N. S. Moosavi, V. Sanh, and I. Gurevych.*
[paper](https://aclanthology.org/2021.emnlp-main.713.pdf)
[code](https://github.com/ukplab/emnlp2021-prompt-ft-heuristics)

1. **Constrained language models yield few-shot semantic parsers,** in EMNLP, 2021.
*R. Shin, C. H. Lin, S. Thomson, C. Chen, S. Roy, E. A. Platanios, A. Pauls, D. Klein, J. Eisner, and B. V. Durme.*
[paper](https://aclanthology.org/2021.emnlp-main.608.pdf)
[code](https://github.com/microsoft/semantic_parsing_with_constrained_lm)

1. **Revisiting self-training for few-shot learning of language model,** in EMNLP, 2021.
*Y. Chen, Y. Zhang, C. Zhang, G. Lee, R. Cheng, and H. Li.*
[paper](https://aclanthology.org/2021.emnlp-main.718.pdf)
[code](https://github.com/matthewcym/sflm)

1. **Language models are few-shot butlers,** in EMNLP, 2021.
*V. Micheli, and F. Fleuret.*
[paper](https://aclanthology.org/2021.emnlp-main.734.pdf)
[code](https://github.com/vmicheli/lm-butlers)

1. **FewshotQA: A simple framework for few-shot learning of question answering tasks using pre-trained text-to-text models,** in EMNLP, 2021.
*R. Chada, and P. Natarajan.*
[paper](https://aclanthology.org/2021.emnlp-main.491.pdf)

1. **TransPrompt: Towards an automatic transferable prompting framework for few-shot text classification,** in EMNLP, 2021.
*C. Wang, J. Wang, M. Qiu, J. Huang, and M. Gao.*
[paper](https://aclanthology.org/2021.emnlp-main.221.pdf)

1. **Meta distant transfer learning for pre-trained language models,** in EMNLP, 2021.
*C. Wang, H. Pan, M. Qiu, J. Huang, F. Yang, and Y. Zhang.*
[paper](https://aclanthology.org/2021.emnlp-main.768.pdf)

1. **STraTA: Self-training with task augmentation for better few-shot learning,** in EMNLP, 2021.
*T. Vu, M. Luong, Q. V. Le, G. Simon, and M. Iyyer.*
[paper](https://aclanthology.org/2021.emnlp-main.462.pdf)
[code](https://github.com/google-research/google-research)

1. **Few-shot image classification: Just use a library of pre-trained feature extractors and a simple classifier,** in ICCV, 2021.
*A. Chowdhury, M. Jiang, S. Chaudhuri, and C. Jermaine.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chowdhury_Few-Shot_Image_Classification_Just_Use_a_Library_of_Pre-Trained_Feature_ICCV_2021_paper.pdf)
[code](https://github.com/arjish/PreTrainedFullLibrary_FewShot)

1. **On the importance of distractors for few-shot classification,** in ICCV, 2021.
*R. Das, Y. Wang, and J. M. F. Moura.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Das_On_the_Importance_of_Distractors_for_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/quantacode/contrastive-finetuning)

1. **A multi-mode modulator for multi-domain few-shot classification,** in ICCV, 2021.
*Y. Liu, J. Lee, L. Zhu, L. Chen, H. Shi, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_A_Multi-Mode_Modulator_for_Multi-Domain_Few-Shot_Classification_ICCV_2021_paper.pdf)

1. **Universal representation learning from multiple domains for few-shot classification,** in ICCV, 2021.
*W. Li, X. Liu, and H. Bilen.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Universal_Representation_Learning_From_Multiple_Domains_for_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/VICO-UoE/URL)

1. **Boosting the generalization capability in cross-domain few-shot learning via noise-enhanced supervised autoencoder,** in ICCV, 2021.
*H. Liang, Q. Zhang, P. Dai, and J. Lu.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Boosting_the_Generalization_Capability_in_Cross-Domain_Few-Shot_Learning_via_Noise-Enhanced_ICCV_2021_paper.pdf)

1. **How fine-tuning allows for effective meta-learning,** in NeurIPS, 2021.
*K. Chua, Q. Lei, and J. D. Lee.*
[paper](https://proceedings.neurips.cc/paper/2021/file/4a533591763dfa743a13affab1a85793-Paper.pdf)

1. **Multimodal few-shot learning with frozen language models,** in NeurIPS, 2021.
*M. Tsimpoukelli, J. Menick, S. Cabi, S. M. A. Eslami, O. Vinyals, and F. Hill.*
[paper](https://proceedings.neurips.cc/paper/2021/file/01b7575c38dac42f3cfb7d500438b875-Paper.pdf)

1. **Grad2Task: Improved few-shot text classification using gradients for task representation,** in NeurIPS, 2021.
*J. Wang, K. Wang, F. Rudzicz, and M. Brudno.*
[paper](https://proceedings.neurips.cc/paper/2021/file/33a854e247155d590883b93bca53848a-Paper.pdf)

1. **True few-shot learning with language models,** in NeurIPS, 2021.
*E. Perez, D. Kiela, and K. Cho.*
[paper](https://proceedings.neurips.cc/paper/2021/file/5c04925674920eb58467fb52ce4ef728-Paper.pdf)

1. **POODLE: Improving few-shot learning via penalizing out-of-distribution samples,** in NeurIPS, 2021.
*D. Le, K. Nguyen, Q. Tran, R. Nguyen, and B. Hua.*
[paper](https://proceedings.neurips.cc/paper/2021/file/c91591a8d461c2869b9f535ded3e213e-Paper.pdf)

1. **TOHAN: A one-step approach towards few-shot hypothesis adaptation,** in NeurIPS, 2021.
*H. Chi, F. Liu, W. Yang, L. Lan, T. Liu, B. Han, W. Cheung, and J. Kwok.*
[paper](https://proceedings.neurips.cc/paper/2021/file/af5d5ef24881f3c3049a7b9bfe74d58b-Paper.pdf)

1. **Task affinity with maximum bipartite matching in few-shot learning,** in ICLR, 2022.
*C. P. Le, J. Dong, M. Soltani, and V. Tarokh.*
[paper](https://openreview.net/pdf?id=u2GZOiUTbt)

1. **Differentiable prompt makes pre-trained language models better few-shot learners,** in ICLR, 2022.
*N. Zhang, L. Li, X. Chen, S. Deng, Z. Bi, C. Tan, F. Huang, and H. Chen.*
[paper](https://openreview.net/pdf?id=ek9a0qIafW)
[code](https://github.com/zjunlp/DART)

1. **ConFeSS: A framework for single source cross-domain few-shot learning,** in ICLR, 2022.
*D. Das, S. Yun, and F. Porikli.*
[paper](https://openreview.net/pdf?id=zRJu6mU2BaE)

1. **Switch to generalize: Domain-switch learning for cross-domain few-shot classification,** in ICLR, 2022.
*Z. Hu, Y. Sun, and Y. Yang.*
[paper](https://openreview.net/pdf?id=H-iABMvzIc)

1. **LM-BFF-MS: Improving few-shot fine-tuning of language models based on multiple soft demonstration memory,** in ACL, 2022.
*E. Park, D. H. Jeon, S. Kim, I. Kang, and S. Na.*
[paper](https://aclanthology.org/2022.acl-short.34.pdf)
[code](https://github.com/judepark96/lm-bff-ms)

1. **Meta-learning via language model in-context tuning,** in ACL, 2022.
*Y. Chen, R. Zhong, S. Zha, G. Karypis, and H. He.*
[paper](https://aclanthology.org/2022.acl-long.53.pdf)
[code](https://github.com/yandachen/in-context-tuning)

1. **Few-shot tabular data enrichment using fine-tuned transformer architectures,** in ACL, 2022.
*A. Harari, and G. Katz.*
[paper](https://aclanthology.org/2022.acl-long.111.pdf)

1. **Noisy channel language model prompting for few-shot text classification,** in ACL, 2022.
*S. Min, M. Lewis, H. Hajishirzi, and L. Zettlemoyer.*
[paper](https://aclanthology.org/2022.acl-long.365.pdf)
[code](https://github.com/shmsw25/Channel-LM-Prompting)

1. **Prompt for extraction? PAIE: Prompting argument interaction for event argument extraction,** in ACL, 2022.
*Y. Ma, Z. Wang, Y. Cao, M. Li, M. Chen, K. Wang, and J. Shao.*
[paper](https://aclanthology.org/2022.acl-long.466.pdf)
[code](https://github.com/mayubo2333/paie)

1. **Are prompt-based models clueless?,** in ACL, 2022.
*P. Kavumba, R. Takahashi, and Y. Oda.*
[paper](https://aclanthology.org/2022.acl-long.166.pdf)

1. **Prototypical verbalizer for prompt-based few-shot tuning,** in ACL, 2022.
*G. Cui, S. Hu, N. Ding, L. Huang, and Z. Liu.*
[paper](https://aclanthology.org/2022.acl-long.483.pdf)
[code](https://github.com/thunlp/OpenPrompt)

1. **Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity,** in ACL, 2022.
*Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp.*
[paper](https://aclanthology.org/2022.acl-long.556.pdf)

1. **PPT: Pre-trained prompt tuning for few-shot learning,** in ACL, 2022.
*Y. Gu, X. Han, Z. Liu, and M. Huang.*
[paper](https://aclanthology.org/2022.acl-long.576.pdf)
[code](https://github.com/thu-coai/ppt)

1. **ASCM: An answer space clustered prompting method without answer engineering,** in Findings of ACL, 2022.
*Z. Wang, Y. Yang, Z. Xi, B. Ma, L. Wang, R. Dong, and A. Anwar.*
[paper](https://aclanthology.org/2022.findings-acl.193.pdf)
[code](https://github.com/miaomiao1215/ascm)

1. **Exploiting language model prompts using similarity measures: A case study on the word-in-context task,** in ACL, 2022.
*M. Tabasi, K. Rezaee, and M. T. Pilehvar.*
[paper](https://aclanthology.org/2022.acl-short.36.pdf)

1. **P-Tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks,** in ACL, 2022.
*X. Liu, K. Ji, Y. Fu, W. Tam, Z. Du, Z. Yang, and J. Tang.*
[paper](https://aclanthology.org/2022.acl-short.8.pdf)

1. **Cutting down on prompts and parameters: Simple few-shot learning with language models,** in Findings of ACL, 2022.
*R. L. L. IV, I. Balazevic, E. Wallace, F. Petroni, S. Singh, and S. Riedel.*
[paper](https://aclanthology.org/2022.findings-acl.222.pdf)
[code](https://github.com/ucinlp/null-prompts)

1. **Prompt-free and efficient few-shot learning with language models,** in ACL, 2022.
*R. K. Mahabadi, L. Zettlemoyer, J. Henderson, L. Mathias, M. Saeidi, V. Stoyanov, and M. Yazdani.*
[paper](https://aclanthology.org/2022.acl-long.254.pdf)
[code](https://github.com/facebookresearch/perfect)

1. **Pre-training to match for unified low-shot relation extraction,** in ACL, 2022.
*F. Liu, H. Lin, X. Han, B. Cao, and L. Sun.*
[paper](https://aclanthology.org/2022.acl-long.397.pdf)
[code](https://github.com/fc-liu/mcmn)

1. **Dual context-guided continuous prompt tuning for few-shot learning,** in Findings of ACL, 2022.
*J. Zhou, L. Tian, H. Yu, Z. Xiao, H. Su, and J. Zhou.*
[paper](https://aclanthology.org/2022.findings-acl.8.pdf)

1. **Cluster & tune: Boost cold start performance in text classification,** in ACL, 2022.
*E. Shnarch, A. Gera, A. Halfon, L. Dankin, L. Choshen, R. Aharonov, and N. Slonim.*
[paper](https://aclanthology.org/2022.acl-long.526.pdf)
[code](https://github.com/ibm/intermediate-training-using-clustering)

1. **Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference,** in CVPR, 2022.
*S. X. Hu, D. Li, J. Stühmer, M. Kim, and T. M. Hospedales.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Pushing_the_Limits_of_Simple_Pipelines_for_Few-Shot_Learning_External_CVPR_2022_paper.pdf)
[code](https://hushell.github.io/pmf/)

1. **HyperTransformer: Model generation for supervised and semi-supervised few-shot learning,** in ICML, 2022.
*A. Zhmoginov, M. Sandler, and M. Vladymyrov.*
[paper](https://proceedings.mlr.press/v162/zhmoginov22a/zhmoginov22a.pdf)
[code](https://github.com/google-research/google-research/tree/master/hypertransformer)

1. **Prompting ELECTRA: Few-shot learning with discriminative pre-trained models,** in EMNLP, 2022.
*M. Xia, M. Artetxe, J. Du, D. Chen, and V. Stoyanov.*
[paper](https://aclanthology.org/2022.emnlp-main.780.pdf)
[code](https://github.com/facebookresearch/ELECTRA-Fewshot-Learning)

1. **Continual training of language models for few-shot learning,** in EMNLP, 2022.
*Z. Ke, H. Lin, Y. Shao, H. Xu, L. Shu, and B. Liu.*
[paper](https://aclanthology.org/2022.emnlp-main.695.pdf)
[code](https://github.com/UIC-Liu-Lab/CPT)

1. **GPS: Genetic prompt search for efficient few-shot learning,** in EMNLP, 2022.
*H. Xu, Y. Chen, Y. Du, N. Shao, Y. Wang, H. Li, and Z. Yang.*
[paper](https://aclanthology.org/2022.emnlp-main.559.pdf)
[code](https://github.com/hwxu20/GPS)

1. **On measuring the intrinsic few-shot hardness of datasets,** in EMNLP, 2022.
*X. Zhao, S. Murty, and C. D. Manning.*
[paper](https://aclanthology.org/2022.emnlp-main.262.pdf)
[code](https://github.com/colinzhaoust/intrinsic_fewshot_hardness)

1. **AMAL: Meta knowledge-driven few-shot adapter learning,** in EMNLP, 2022.
*S. K. Hong, and T. Y. Jang.*
[paper](https://aclanthology.org/2022.emnlp-main.709.pdf)

1. **Flamingo: A visual language model for few-shot learning,** in NeurIPS, 2022.
*J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan.*
[paper](https://openreview.net/pdf?id=EbMuimAbPbs)

1. **Language models with image descriptors are strong few-shot video-language learners,** in NeurIPS, 2022.
*Z. Wang, M. Li, R. Xu, L. Zhou, J. Lei, X. Lin, S. Wang, Z. Yang, C. Zhu, D. Hoiem, S.-F. Chang, M. Bansal, and H. Ji.*
[paper](https://openreview.net/pdf?id=_LceCyuVcH)
[code](https://github.com/MikeWangWZHL/VidIL)

1. **Singular value fine-tuning: Few-shot segmentation requires few-parameters fine-tuning,** in NeurIPS, 2022.
*Y. Sun, Q. Chen, X. He, J. Wang, H. Feng, J. Han, E. Ding, J. Cheng, Z. Li, and J. Wang.*
[paper](https://openreview.net/pdf?id=LEqYZz7cZOI)
[code](https://github.com/syp2ysy/SVF)

1. **Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning,** in NeurIPS, 2022.
*H. Liu, D. Tam, M. Mohammed, J. Mohta, T. Huang, M. Bansal, and C. Raffel.*
[paper](https://openreview.net/pdf?id=rBCvMG-JsPd)
[code](https://github.com/r-three/t-few)

1. **Powering finetuning in few-shot learning: Domain-agnostic bias reduction with selected sampling,** in AAAI, 2022.
*R. Tao, H. Zhang, Y. Zheng, and M. Savvides.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20823/20582)

1. **SELECTION-INFERENCE: Exploiting large language models for interpretable logical reasoning,** in ICLR, 2023.
*A. Creswell, M. Shannahan, and I. Higgins.*
[paper](https://arxiv.org/pdf/2205.09712.pdf)

1. **Revisit finetuning strategy for few-shot learning to transfer the emdeddings,** in ICLR, 2023.
*H. Wang, T. Yue, X. Ye, Z. He, B. Li, and Y. Li.*
[paper](https://openreview.net/pdf?id=tXc-riXhmx)
[code](https://github.com/whzyf951620/ LinearProbingFinetuningFirthBias)

1. **Model ensemble instead of prompt fusion: A sample-specific knowledge transfer method for few-shot prompt tuning,** in ICLR, 2023.
*X. PENG, C. Xing, P. K. Choubey, C.-S. Wu, and C. Xiong.*
[paper](https://openreview.net/pdf?id=p0yrSRbN5Bu)

1. **Bidirectional language models are also few-shot learners,** in ICLR, 2023.
*A. Patel, B. Li, M. S. Rasooli, N. Constant, C. Raffel, and C. Callison-Burch.*
[paper](https://openreview.net/pdf?id=wCFB37bzud4)

1. **Prototypical calibration for few-shot learning of language models,** in ICLR, 2023.
*Z. Han, Y. Hao, L. Dong, Y. Sun, and F. Wei.*
[paper](https://openreview.net/pdf?id=nUsP9lFADUF)
[code](https://github.com/zhixhan/ProCa)

1. **Prompt, generate, then cache: Cascade of foundation models makes strong few-shot learners,** in CVPR, 2023.
*R. Zhang, X. Hu, B. Li, S. Huang, H. Deng, Y. Qiao, P. Gao, and H. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Prompt_Generate_Then_Cache_Cascade_of_Foundation_Models_Makes_Strong_CVPR_2023_paper.pdf)
[code](https://github.com/ZrrSkywalker/CaFo)

1. **Supervised masked knowledge distillation for few-shot transformers,** in CVPR, 2023.
*H. Lin, G. Han, J. Ma, S. Huang, X. Lin, and S.-F. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Supervised_Masked_Knowledge_Distillation_for_Few-Shot_Transformers_CVPR_2023_paper.pdf)
[code](https://github.com/HL-hanlin/SMKD)

1. **Boosting transductive few-shot fine-tuning with margin-based uncertainty weighting and probability regularization,** in CVPR, 2023.
*R. Tao, H. Chen, and M. Savvides.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tao_Boosting_Transductive_Few-Shot_Fine-Tuning_With_Margin-Based_Uncertainty_Weighting_and_Probability_CVPR_2023_paper.pdf)

1. **Hint-Aug: Drawing hints from foundation vision transformers towards boosted few-shot parameter-efficient tuning,** in CVPR, 2023.
*Z. Yu, S. Wu, Y. Fu, S. Zhang, and Y. C. Lin.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Hint-Aug_Drawing_Hints_From_Foundation_Vision_Transformers_Towards_Boosted_Few-Shot_CVPR_2023_paper.pdf)
[code](https://github.com/GATECH-EIC/Hint-Aug)

1. **ProD: Prompting-to-disentangle domain knowledge for cross-domain few-shot image classification,** in CVPR, 2023.
*T. Ma, Y. Sun, Z. Yang, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_ProD_Prompting-To-Disentangle_Domain_Knowledge_for_Cross-Domain_Few-Shot_Image_Classification_CVPR_2023_paper.pdf)

1. **Few-shot learning with visual distribution calibration and cross-modal distribution alignment,** in CVPR, 2023.
*R. Wang, H. Zheng, X. Duan, J. Liu, Y. Lu, T. Wang, S. Xu, and B. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Few-Shot_Learning_With_Visual_Distribution_Calibration_and_Cross-Modal_Distribution_Alignment_CVPR_2023_paper.pdf)
[code](https://gitee.com/mindspore/models/tree/master/research/cv)

1. **MetricPrompt: Prompting model as a relevance metric for few-shot text classification,** in KDD, 2023.
*H. Dong, W. Zhang, and W. Che.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599430)
[code](https://github.com/Dousia/MetricPrompt)

1. **Efficient training of language models using few-shot learning,** in ICML, 2023.
*S. J. Reddi, S. Miryoosefi, S. Karp, S. Krishnan, S. Kale, S. Kim, and S. Kumar.*
[paper](https://proceedings.mlr.press/v202/j-reddi23a/j-reddi23a.pdf)

1. **Multitask pre-training of modular prompt for chinese few-shot learning,** in ACL, 2023.
*T. Sun, Z. He, Q. Zhu, X. Qiu, and X. Huang.*
[paper](https://aclanthology.org/2023.acl-long.625.pdf)
[code](https://github.com/Hzfinfdu/MPMP)

1. **Cold-start data selection for better few-shot language model fine-tuning: A prompt-based uncertainty propagation approach,** in ACL, 2023.
*Y. Yu, R. Zhang, R. Xu, J. Zhang, J. Shen, and C. Zhang.*
[paper](https://aclanthology.org/2023.acl-long.141.pdf)
[code](https://github.com/yueyu1030/Patron)

1. **Instruction induction: From few examples to natural language task descriptions,** in ACL, 2023.
*O. Honovich, U. Shaham, S. R. Bowman, and O. Levy.*
[paper](https://aclanthology.org/2023.acl-long.108.pdf)
[code](https://github.com/orhonovich/instruction-induction)

1. **Few-shot adaptation works with unpredictable data,** in ACL, 2023.
*J. S. Chan, M. Pieler, J. Jao, J. Scheurer, and E. Perez.*
[paper](https://aclanthology.org/2023.acl-long.102.pdf)

1. **Hierarchical verbalizer for few-shot hierarchical text classification,** in ACL, 2023.
*K. Ji, Y. Lian, J. Gao, and B. Wang.*
[paper](https://aclanthology.org/2023.acl-long.164.pdf)
[code](https://github.com/1KE-JI/HierVerb)

16. **Black box few-shot adaptation for vision-language models,** in ICCV, 2023.
*Y. Ouali, A. Bulat, B. Matinez, and G. Tzimiropoulos.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.pdf)
[code](https://github.com/saic-fi/LFA)

42. **Read-only prompt optimization for vision-language few-shot learning,** in ICCV, 2023.
*D. Lee, S. Song, J. Suh, J. Choi, S. Lee, and H. J. Kim.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Read-only_Prompt_Optimization_for_Vision-Language_Few-shot_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/mlvlab/RPO)

40. **Not all features matter: Enhancing few-shot CLIP with adaptive prior refinement,** in ICCV, 2023.
*X. Zhu, R. Zhang, B. He, A. Zhou, D. Wang, B. Zhao, and P. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Not_All_Features_Matter_Enhancing_Few-shot_CLIP_with_Adaptive_Prior_ICCV_2023_paper.pdf)
[code](https://github.com/yangyangyang127/APE)

13. **One-shot generative domain adaptation,** in ICCV, 2023.
*C. Yang, Y. Shen, Z. Zhang, Y. Xu, J. Zhu, Z. Wu, and B. Zhou.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_One-Shot_Generative_Domain_Adaptation_ICCV_2023_paper.pdf)
[code](https://genforce.github.io/genda/)

21. **Smoothness similarity regularization for few-shot GAN adaptation,** in ICCV, 2023.
*V. Sushko, R. Wang, and J. Gall.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Sushko_Smoothness_Similarity_Regularization_for_Few-Shot_GAN_Adaptation_ICCV_2023_paper.pdf)

27. **Task-aware adaptive learning for cross-domain few-shot learning,** in ICCV, 2023.
*Y. Guo, R. Du, Y. Dong, T. Hospedales, Y. Song, and Z. Ma.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Task-aware_Adaptive_Learning_for_Cross-domain_Few-shot_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/PRIS-CV/TA2-Net)

13. **Defending pre-trained language models as few-shot learners against backdoor attacks,** in NeurIPS, 2023.
*Z. Xi, T. Du, C. Li, R. Pang, S. Ji, J. Chen, F. Ma, and T. Wang.*
[paper](https://openreview.net/attachment?id=GqXbfVmEPW&name=pdf)
[code](https://github.com/zhaohan-xi/PLM-prompt-defense)

14. **FD-Align: Feature discrimination alignment for fine-tuning pre-trained models in few-shot learning,** in NeurIPS, 2023.
*K. Song, H. Ma, B. Zou, H. Zhang, and W. Huang.*
[paper](https://openreview.net/attachment?id=shXnfALjuH&name=pdf)
[code](https://github.com/skingorz/FD-Align)

19. **Fairness-guided few-shot prompting for large language models,** in NeurIPS, 2023.
*H. Ma, C. Zhang, Y. Bian, L. Liu, Z. Zhang, P. Zhao, S. Zhang, H. Fu, Q. Hu, and B. Wu.*
[paper](https://openreview.net/attachment?id=D8oHQ2qSTj&name=pdf)

15. **Meta-Adapter: An online few-shot learner for vision-language model,** in NeurIPS, 2023.
*C. Cheng, L. Song, R. Xue, H. Wang, H. Sun, Y. Ge, and Y. Shan.*
[paper](https://openreview.net/attachment?id=Ts0d8PvTeB&name=pdf)

17. **Language models can improve event prediction by few-shot abductive reasoning,** in NeurIPS, 2023.
*X. Shi, S. Xue, K. Wang, F. Zhou, J. Y. Zhang, J. ZHOU, C. Tan, and H. Mei.*
[paper](https://openreview.net/attachment?id=aW9BqtRQkh&name=pdf)
[code](https://github.com/iLampard/lamp)

3. **ExPT: Synthetic pretraining for few-shot experimental design,** in NeurIPS, 2023.
*T. Nguyen, S. Agrawal, and A. Grover.*
[paper](https://openreview.net/attachment?id=7qfkImn0dL&name=pdf)
[code](https://github.com/tung-nd/ExPT)

25. **LoCoOp: Few-shot out-of-distribution detection via prompt learning,** in NeurIPS, 2023.
*A. Miyai, Q. Yu, G. Irie, and K. Aizawa.*
[paper](https://openreview.net/attachment?id=UjtiLdXGMC&name=pdf)
[code](https://github.com/AtsuMiyai/LoCoOp)

9. **Embroid: Unsupervised prediction smoothing can improve few-shot classification,** in NeurIPS, 2023.
*N. Guha, M. F. Chen, K. Bhatia, A. Mirhoseini, F. Sala, and C. Re.*
[paper](https://openreview.net/attachment?id=4iMpwAlza1&name=pdf)

18. **Domain re-modulation for few-shot generative domain adaptation,** in NeurIPS, 2023.
*Y. Wu, Z. Li, C. Wang, H. Zheng, S. Zhao, B. Li, and D. Tao.*
[paper](https://openreview.net/attachment?id=jown9RvYn7&name=pdf)
[code](https://github.com/wuyi2020/DoRM)

26. **Focus your attention when few-shot classification,** in NeurIPS, 2023.
*H. Wang, S. Jie, and Z. Deng.*
[paper](https://openreview.net/attachment?id=uFlE0qgtRO&name=pdf)
[code](https://github.com/Haoqing-Wang/FORT)

### Refining Meta-learned Parameters

1. **Model-agnostic meta-learning for fast adaptation of deep networks,** in ICML, 2017.
*C. Finn, P. Abbeel, and S. Levine.*
[paper](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf?source=post_page---------------------------)

1. **Bayesian model-agnostic meta-learning,** in NeurIPS, 2018.
*J. Yoon, T. Kim, O. Dia, S. Kim, Y. Bengio, and S. Ahn.*
[paper](https://papers.nips.cc/paper/7963-bayesian-model-agnostic-meta-learning.pdf)

1. **Probabilistic model-agnostic meta-learning,** in NeurIPS, 2018.
*C. Finn, K. Xu, and S. Levine.*
[paper](https://papers.nips.cc/paper/8161-probabilistic-model-agnostic-meta-learning.pdf)

1. **Gradient-based meta-learning with learned layerwise metric and subspace,** in ICML, 2018.
*Y. Lee and S. Choi.*
[paper](http://proceedings.mlr.press/v80/lee18a/lee18a.pdf)

1. **Recasting gradient-based meta-learning as hierarchical Bayes,** in ICLR, 2018.
*E. Grant, C. Finn, S. Levine, T. Darrell, and T. Griffiths.*
[paper](https://openreview.net/forum?id=BJ_UL-k0b)

1. **Few-shot human motion prediction via meta-learning,** in ECCV, 2018.
*L.-Y. Gui, Y.-X. Wang, D. Ramanan, and J. Moura.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangyan_Gui_Few-Shot_Human_Motion_ECCV_2018_paper.pdf)

1. **The effects of negative adaptation in model-agnostic meta-learning,** arXiv preprint, 2018.
*T. Deleu and Y. Bengio.*
[paper](http://metalearning.ml/2018/papers/metalearn2018_paper76.pdf)

1. **Unsupervised meta-learning for few-shot image classification,** in NeurIPS, 2019.
*S. Khodadadeh, L. Bölöni, and M. Shah.*
[paper](https://papers.nips.cc/paper/9203-unsupervised-meta-learning-for-few-shot-image-classification.pdf)

1. **Amortized bayesian meta-learning,** in ICLR, 2019.
*S. Ravi and A. Beatson.*
[paper](https://openreview.net/forum?id=rkgpy3C5tX)

1. **Meta-learning with latent embedding optimization,** in ICLR, 2019.
*A. A. Rusu, D. Rao, J. Sygnowski, O. Vinyals, R. Pascanu, S. Osindero, and R. Hadsell.* 
[paper](https://openreview.net/forum?id=BJgklhAcK7)
[code](https://github.com/deepmind/leo)

1. **LGM-Net: Learning to generate matching networks for few-shot learning,** in ICML, 2019.
*H. Li, W. Dong, X. Mei, C. Ma, F. Huang, and B.-G. Hu.*
[paper](http://proceedings.mlr.press/v97/li19c/li19c.pdf)
[code](https://github.com/likesiwell/LGM-Net/)

1. **Meta R-CNN: Towards general solver for instance-level low-shot learning,** in ICCV, 2019.
*X. Yan, Z. Chen, A. Xu, X. Wang, X. Liang, and L. Lin.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Meta_R-CNN_Towards_General_Solver_for_Instance-Level_Low-Shot_Learning_ICCV_2019_paper.pdf)

1. **Task agnostic meta-learning for few-shot learning,** in CVPR, 2019.
*M. A. Jamal, and G.-J. Qi.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jamal_Task_Agnostic_Meta-Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

1. **Meta-transfer learning for few-shot learning,** in CVPR, 2019.
*Q. Sun, Y. Liu, T.-S. Chua, and B. Schiele.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/yaoyao-liu/meta-transfer-learning)

1. **Meta-learning of neural architectures for few-shot learning,** in CVPR, 2020.
*T. Elsken, B. Staffler, J. H. Metzen, and F. Hutter.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Elsken_Meta-Learning_of_Neural_Architectures_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Attentive weights generation for few shot learning via information maximization,** in CVPR, 2020.
*Y. Guo, and N.-M. Cheung.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Attentive_Weights_Generation_for_Few_Shot_Learning_via_Information_Maximization_CVPR_2020_paper.pdf)

1. **Few-shot open-set recognition using meta-learning,** in CVPR, 2020.
*B. Liu, H. Kang, H. Li, G. Hua, and N. Vasconcelos.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Few-Shot_Open-Set_Recognition_Using_Meta-Learning_CVPR_2020_paper.pdf)

1. **Incremental few-shot object detection,** in CVPR, 2020.
*J.-M. Perez-Rua, X. Zhu, T. M. Hospedales, and T. Xiang.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Perez-Rua_Incremental_Few-Shot_Object_Detection_CVPR_2020_paper.pdf)

1. **Automated relational meta-learning,** in ICLR, 2020.
*H. Yao, X. Wu, Z. Tao, Y. Li, B. Ding, R. Li, and Z. Li.*
[paper](https://openreview.net/pdf?id=rklp93EtwH)

1. **Meta-learning with warped gradient descent,** in ICLR, 2020.
*S. Flennerhag, A. A. Rusu, R. Pascanu, F. Visin, H. Yin, and R. Hadsell.*
[paper](https://openreview.net/pdf?id=rkeiQlBFPB)

1. **Meta-learning without memorization,** in ICLR, 2020.
*M. Yin, G. Tucker, M. Zhou, S. Levine, and C. Finn.*
[paper](https://openreview.net/pdf?id=BklEFpEYwS)

1. **ES-MAML: Simple Hessian-free meta learning,** in ICLR, 2020.
*X. Song, W. Gao, Y. Yang, K. Choromanski, A. Pacchiano, and Y. Tang.*
[paper](https://openreview.net/pdf?id=S1exA2NtDB)

1. **Self-supervised tuning for few-shot segmentation,** in IJCAI, 2020.
*K. Zhu, W. Zhai, and Y. Cao.*
[paper](https://www.ijcai.org/Proceedings/2020/0142.pd)

1. **Multi-attention meta learning for few-shot fine-grained image recognition,** in IJCAI, 2020.
*Y. Zhu, C. Liu, and S. Jiang.*
[paper](https://www.ijcai.org/Proceedings/2020/0152.pdf)

1. **An ensemble of epoch-wise empirical Bayes for few-shot learning,** in ECCV, 2020.
*Y. Liu, B. Schiele, and Q. Sun.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610392.pdf)
[code](https://gitlab.mpi-klsb.mpg.de/yaoyaoliu/e3bm)

1. **Incremental few-shot meta-learning via indirect discriminant alignment,** in ECCV, 2020.
*Q. Liu, O. Majumder, A. Achille, A. Ravichandran, R. Bhotika, and S. Soatto.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520664.pdf)

1. **Model-agnostic boundary-adversarial sampling for test-time generalization in few-shot learning,** in ECCV, 2020.
*J. Kim, H. Kim, and G. Kim.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460579.pdf)
[code](https://github.com/jaekyeom/MABAS)

1. **Bayesian meta-learning for the few-shot setting via deep kernels,** in NeurIPS, 2020.
*M. Patacchiola, J. Turner, E. J. Crowley, M. O'Boyle, and A. J. Storkey.*
[paper](https://proceedings.neurips.cc/paper/2020/file/b9cfe8b6042cf759dc4c0cccb27a6737-Paper.pdf)
[code](https://github.com/BayesWatch/deep-kernel-transfer)

1. **OOD-MAML: Meta-learning for few-shot out-of-distribution detection and classification,** in NeurIPS, 2020.
*T. Jeong, and H. Kim.*
[paper](https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf)
[code](https://github.com/twj-KAIST/OOD-MAML)

1. **Unraveling meta-learning: Understanding feature representations for few-shot tasks,** in ICML, 2020.
*M. Goldblum, S. Reich, L. Fowl, R. Ni, V. Cherepanova, and T. Goldstein.*
[paper](http://proceedings.mlr.press/v119/goldblum20a/goldblum20a.pdf)
[code](https://github.com/goldblum/FeatureClustering)

1. **Node classification on graphs with few-shot novel labels via meta transformed network embedding,** in NeurIPS, 2020.
*L. Lan, P. Wang, X. Du, K. Song, J. Tao, and X. Guan.*
[paper](https://proceedings.neurips.cc/paper/2020/file/c055dcc749c2632fd4dd806301f05ba6-Paper.pdf)

1. **Adversarially robust few-shot learning: A meta-learning approach,** in NeurIPS, 2020.
*M. Goldblum, L. Fowl, and T. Goldstein.*
[paper](https://proceedings.neurips.cc/paper/2020/file/cfee398643cbc3dc5eefc89334cacdc1-Paper.pdf)
[code](https://github.com/goldblum/AdversarialQuerying)

1. **BOIL: Towards representation change for few-shot learning,** in ICLR, 2021.
*J. Oh, H. Yoo, C. Kim, and S. Yun.*
[paper](https://openreview.net/pdf?id=umIdUL8rMH)
[code](https://github.com/flennerhag/warpgrad)

1. **Few-shot open-set recognition by transformation consistency,** in CVPR, 2021.
*M. Jeong, S. Choi, and C. Kim.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jeong_Few-Shot_Open-Set_Recognition_by_Transformation_Consistency_CVPR_2021_paper.pdf)

1. **Improving generalization in meta-learning via task augmentation,** in ICML, 2021.
*H. Yao, L. Huang, L. Zhang, Y. Wei, L. Tian, J. Zou, J. Huang, and Z. Li.*
[paper](http://proceedings.mlr.press/v139/yao21b/yao21b.pdf)

1. **A representation learning perspective on the importance of train-validation splitting in meta-learning,** in ICML, 2021.
*N. Saunshi, A. Gupta, and W. Hu.*
[paper](http://proceedings.mlr.press/v139/saunshi21a/saunshi21a.pdf)
[code](https://github.com/nsaunshi/meta_tr_val_split)

1. **Data augmentation for meta-learning,** in ICML, 2021.
*R. Ni, M. Goldblum, A. Sharaf, K. Kong, and T. Goldstein.*
[paper](http://proceedings.mlr.press/v139/ni21a/ni21a.pdf)
[code](https://github.com/RenkunNi/MetaAug)

1. **Task cooperation for semi-supervised few-shot learning,** in AAAI, 2021.
*H. Ye, X. Li, and D.-C. Zhan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17277/17084)

1. **Conditional self-supervised learning for few-shot classification,** in IJCAI, 2021.
*Y. An, H. Xue, X. Zhao, and L. Zhang.*
[paper](https://www.ijcai.org/proceedings/2021/0295.pdf)

1. **Cross-domain few-shot classification via adversarial task augmentation,** in IJCAI, 2021.
*H. Wang, and Z.-H. Deng.*
[paper](https://www.ijcai.org/proceedings/2021/0149.pdf)
[code](https://github.com/Haoqing-Wang/CDFSL-ATA)

1. **DReCa: A general task augmentation strategy for few-shot natural language inference,** in NAACL-HLT, 2021.
*S. Murty, T. Hashimoto, and C. D. Manning.*
[paper](https://aclanthology.org/2021.naacl-main.88.pdf)

1. **MetaXL: Meta representation transformation for low-resource cross-lingual learning,** in NAACL-HLT, 2021.
*M. Xia, G. Zheng, S. Mukherjee, M. Shokouhi, G. Neubig, and A. H. Awadallah.*
[paper](https://aclanthology.org/2021.naacl-main.42.pdf)
[code](https://github.com/microsoft/MetaXL)

1. **Meta-learning with task-adaptive loss function for few-shot learning,** in ICCV, 2021.
*S. Baik, J. Choi, H. Kim, D. Cho, J. Min, and K. M. Lee.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Baik_Meta-Learning_With_Task-Adaptive_Loss_Function_for_Few-Shot_Learning_ICCV_2021_paper.pdf)
[code](https://github.com/baiksung/MeTAL)

1. **Meta-Baseline: Exploring simple meta-learning for few-shot learning,** in ICCV, 2021.
*Y. Chen, Z. Liu, H. Xu, T. Darrell, and X. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Meta-Baseline_Exploring_Simple_Meta-Learning_for_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **A lazy approach to long-horizon gradient-based meta-learning,** in ICCV, 2021.
*M. A. Jamal, L. Wang, and B. Gong.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jamal_A_Lazy_Approach_to_Long-Horizon_Gradient-Based_Meta-Learning_ICCV_2021_paper.pdf)

1. **Task-aware part mining network for few-shot learning,** in ICCV, 2021.
*J. Wu, T. Zhang, Y. Zhang, and F. Wu.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Task-Aware_Part_Mining_Network_for_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Binocular mutual learning for improving few-shot classification,** in ICCV, 2021.
*Z. Zhou, X. Qiu, J. Xie, J. Wu, and C. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Binocular_Mutual_Learning_for_Improving_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/zzqzzq/bml)

1. **Meta-learning with an adaptive task scheduler,** in NeurIPS, 2021.
*H. Yao, Y. Wang, Y. Wei, P. Zhao, M. Mahdavi, D. Lian, and C. Finn.*
[paper](https://proceedings.neurips.cc/paper/2021/file/3dc4876f3f08201c7c76cb71fa1da439-Paper.pdf)

1. **Memory efficient meta-learning with large images,** in NeurIPS, 2021.
*J. Bronskill, D. Massiceti, M. Patacchiola, K. Hofmann, S. Nowozin, and R. Turner.*
[paper](https://proceedings.neurips.cc/paper/2021/file/cc1aa436277138f61cda703991069eaf-Paper.pdf)

1. **EvoGrad: Efficient gradient-based meta-learning and hyperparameter optimization,** in NeurIPS, 2021.
*O. Bohdal, Y. Yang, and T. Hospedales.*
[paper](https://proceedings.neurips.cc/paper/2021/file/bac49b876d5dfc9cd169c22ef5178ca7-Paper.pdf)

1. **Towards enabling meta-learning from target models,** in NeurIPS, 2021.
*S. Lu, H. Ye, L. Gan, and D. Zhan.*
[paper](https://proceedings.neurips.cc/paper/2021/file/43baa6762fa81bb43b39c62553b2970d-Paper.pdf)

1. **The role of global labels in few-shot classification and how to infer them,** in NeurIPS, 2021.
*R. Wang, M. Pontil, and C. Ciliberto.*
[paper](https://proceedings.neurips.cc/paper/2021/file/e3b6fb0fd4df098162eede3313c54a8d-Paper.pdf)

1. **How to train your MAML to excel in few-shot classification,** in ICLR, 2022.
*H. Ye, and W. Chao.*
[paper](https://openreview.net/pdf?id=49h_IkpJtaE)
[code](https://github.com/Han-Jia/UNICORN-MAML)

1. **Meta-learning with fewer tasks through task interpolation,** in ICLR, 2022.
*H. Yao, L. Zhang, and C. Finn.*
[paper](https://openreview.net/pdf?id=ajXWF7bVR8d)
[code](https://github.com/huaxiuyao/MLTI)

1. **Continuous-time meta-learning with forward mode differentiation,** in ICLR, 2022.
*T. Deleu, D. Kanaa, L. Feng, G. Kerg, Y. Bengio, G. Lajoie, and P. Bacon.*
[paper](https://openreview.net/pdf?id=57PipS27Km)

1. **Bootstrapped meta-learning,** in ICLR, 2022.
*S. Flennerhag, Y. Schroecker, T. Zahavy, H. v. Hasselt, D. Silver, and S. Singh.*
[paper](https://openreview.net/pdf?id=b-ny3x071E5)

1. **Learning prototype-oriented set representations for meta-learning,** in ICLR, 2022.
*D. d. Guo, L. Tian, M. Zhang, M. Zhou, and H. Zha.*
[paper](https://openreview.net/pdf?id=WH6u2SvlLp4)

1. **Dynamic kernel selection for improved generalization and memory efficiency in meta-learning,** in CVPR, 2022.
*A. Chavan, R. Tiwari, U. Bamba, and D. K. Gupta.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Dynamic_Kernel_Selection_for_Improved_Generalization_and_Memory_Efficiency_in_CVPR_2022_paper.pdf)
[code](https://github.com/transmuteAI/MetaDOCK)

1. **What matters for meta-learning vision regression tasks?,** in CVPR, 2022.
*N. Gao, H. Ziesche, N. A. Vien, M. Volpp, and G. Neumann.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_What_Matters_for_Meta-Learning_Vision_Regression_Tasks_CVPR_2022_paper.pdf)
[code](https://github.com/boschresearch/what-matters-for-meta-learning)

1. **Multidimensional belief quantification for label-efficient meta-learning,** in CVPR, 2022.
*D. S. Pandey, and Q. Yu.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Pandey_Multidimensional_Belief_Quantification_for_Label-Efficient_Meta-Learning_CVPR_2022_paper.pdf)

1. **Few-shot node classification on attributed networks with graph meta-learning,** in SIGIR, 2022.
*Y. Liu, M. Li, X. Li, F. Giunchiglia, X. Feng, and R. Guan.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531978)

1. **The role of deconfounding in meta-learning,** in ICML, 2022.
*Y. Jiang, Z. Chen, K. Kuang, L. Yuan, X. Ye, Z. Wang, F. Wu, and Y. Wei.*
[paper](https://proceedings.mlr.press/v162/jiang22a/jiang22a.pdf)

1. **Stochastic deep networks with linear competing units for model-agnostic meta-learning,** in ICML, 2022.
*K. Kalais, and S. Chatzis.*
[paper](https://proceedings.mlr.press/v162/kalais22a/kalais22a.pdf)
[code](https://github.com/Kkalais/StochLWTA-ML)

1. **Efficient variance reduction for meta-learning,** in ICML, 2022.
*H. Yang, and J. T. Kwok.*
[paper](https://proceedings.mlr.press/v162/yang22g/yang22g.pdf)

1. **Subspace learning for effective meta-learning,** in ICML, 2022.
*W. Jiang, J. Kwok, and Y. Zhang.*
[paper](https://proceedings.mlr.press/v162/jiang22b/jiang22b.pdf)

1. **Robust meta-learning with sampling noise and label noise via Eigen-Reptile,** in ICML, 2022.
*D. Chen, L. Wu, S. Tang, X. Yun, B. Long, and Y. Zhuang.*
[paper](https://proceedings.mlr.press/v162/chen22aa/chen22aa.pdf)
[code](https://github.com/Anfeather/Eigen-Reptile)

1. **Attentional meta-learners for few-shot polythetic classification,** in ICML, 2022.
*B. J. Day, R. V. Torné, N. Simidjievski, and P. Lió.*
[paper](https://proceedings.mlr.press/v162/day22a/day22a.pdf)
[code](https://github.com/rvinas/polythetic_metalearning)

1. **PLATINUM: Semi-supervised model agnostic meta-learning using submodular mutual information,** in ICML, 2022.
*C. Li, S. Kothawade, F. Chen, and R. K. Iyer.*
[paper](https://proceedings.mlr.press/v162/li22k/li22k.pdf)
[code](https://github.com/Hugo101/PLATINUM)

1. **Finding meta winning ticket to train your MAML,** in KDD, 2022.
*D. Gao, Y. Xie, Z. Zhou, Z. Wang, Y. Li, and B. Ding.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539467)

1. **p-Meta: Towards on-device deep model adaptation,** in KDD, 2022.
*Z. Qu, Z. Zhou, Y. Tong, and L. Thiele.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539293)

1. **FAITH: Few-shot graph classification with hierarchical task graphs,** in IJCAI, 2022.
*S. Wang, Y. Dong, X. Huang, C. Chen, and J. Li.*
[paper](https://www.ijcai.org/proceedings/2022/0317.pdf)
[code](https://github.com/SongW-SW/FAITH)

1. **Meta-learning fast weight language models,** in EMNLP, 2022.
*K. Clark, K. Guu, M.-W. Chang, P. Pasupat, G. Hinton, and M. Norouzi.*
[paper](https://aclanthology.org/2022.emnlp-main.661.pdf)

1. **Understanding benign overfitting in gradient-based meta learning,** in NeurIPS, 2022.
*L. Chen, S. Lu, and T. Chen.*
[paper](https://openreview.net/pdf?id=oW4Zz0zlbFF)

1. **Meta-learning with self-improving momentum target,** in NeurIPS, 2022.
*J. Tack, J. Park, H. Lee, J. Lee, and J. Shin.*
[paper](https://github.com/jihoontack/SiMT)

1. **Adversarial task up-sampling for meta-learning,** in NeurIPS, 2022.
*Y. Wu, L.-K. Huang, and Y. Wei.*
[paper](https://openreview.net/pdf?id=pFqgUJxXXz)

1. **PAC prediction sets for meta-learning,** in NeurIPS, 2022.
*S. Park, E. Dobriban, I. Lee, and O. Bastani.*
[paper](https://openreview.net/pdf?id=s6ygs1UCOw1)

1. **A contrastive rule for meta-learning,** in NeurIPS, 2022.
*N. Zucchet, S. Schug, J. V. Oswald, D. Zhao, and J. Sacramento.*
[paper](https://openreview.net/pdf?id=NIJFp_n4MXt)
[code](https://github.com/smonsays/contrastive-meta-learning)

1. **On enforcing better conditioned meta-learning for rapid few-shot adaptation,** in NeurIPS, 2022.
*M. Hiller, M. Harandi, and T. Drummond.*
[paper](https://openreview.net/pdf?id=G6cJsOOx2R3)

1. **Conditional meta-learning of linear representations,** in NeurIPS, 2022.
*G. Denevi, m. pontil, and C. Ciliberto.*
[paper](https://openreview.net/pdf?id=0Uejkm1GB1U)

1. **Meta-ticket: Finding optimal subnetworks for few-shot learning within randomly initialized neural networks,** in NeurIPS, 2022.
*D. Chijiwa, S. Yamaguchi, A. Kumagai, and Y. Ida.*
[paper](https://openreview.net/pdf?id=Cr4_3ptitj)
[code](https://www.catalyzex.com/paper/arxiv:2205.15619/code)

1. **MetaNODE: Prototype optimization as a neural ODE for few-shot learning,** in AAAI, 2022.
*B. Zhang, X. Li, S. Feng, Y. Ye, and R. Ye.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20885/20644)

1. **A nested bi-level optimization framework for robust few shot learning,** in AAAI, 2022.
*K. Killamsetty, C. Li, C. Zhao, F. Chen, and R. K. Iyer.*
[paper](https://arxiv.org/pdf/2011.06782.pdf)

1. **Enhancing meta learning via multi-objective soft improvement functions,** in ICLR, 2023.
*R. Yu, W. Chen, X. Wang, and J. Kwok.*
[paper](https://openreview.net/pdf?id=hCmjBJeGXcu)

1. **Understanding train-validation split in meta-learning with neural networks,** in ICLR, 2023.
*X. Zuo, Z. Chen, H. Yao, Y. Cao, and Q. Gu.*
[paper](https://openreview.net/pdf?id=JVlyfHEEm0k)

1. **Bi-level meta-learning for few-shot domain generalization,** in CVPR, 2023.
*X. Qin, X. Song, and S. Jiang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Qin_Bi-Level_Meta-Learning_for_Few-Shot_Domain_Generalization_CVPR_2023_paper.pdf)

12. **SHOT: Suppressing the hessian along the optimization trajectory for gradient-based meta-learning,** in NeurIPS, 2023.
*J. Lee, J. Yoo, and N. Kwak.*
[paper](https://openreview.net/attachment?id=PXsqbAjpQd&name=pdf)
[code](https://github.com/JunHoo-Lee/SHOT)

11. **Meta-AdaM: An meta-learned adaptive optimizer with momentum for few-shot learning,** in NeurIPS, 2023.
*S. Sun, and H. Gao.*
[paper](https://openreview.net/attachment?id=d85pPNBHLt&name=pdf)

### Learning Search Steps

1. **Optimization as a model for few-shot learning,** in ICLR, 2017.
*S. Ravi and H. Larochelle.*
[paper](https://openreview.net/forum?id=rJY0-Kcll)
[code](https://github.com/twitter/meta-learning-lstm)

1. **Meta Navigator: Search for a good adaptation policy for few-shot learning,** in ICCV, 2021.
*C. Zhang, H. Ding, G. Lin, R. Li, C. Wang, and C. Shen.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Meta_Navigator_Search_for_a_Good_Adaptation_Policy_for_Few-Shot_ICCV_2021_paper.pdf)


## [Applications](#content)


### Computer Vision

1. **Learning robust visual-semantic embeddings,** in CVPR, 2017.
*Y.-H. Tsai, L.-K. Huang, and R. Salakhutdinov.*
[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tsai_Learning_Robust_Visual-Semantic_ICCV_2017_paper.pdf)

1. **One-shot action localization by learning sequence matching network,** in CVPR, 2018.
*H. Yang, X. He, and F. Porikli.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_One-Shot_Action_Localization_CVPR_2018_paper.pdf)

1. **Incremental few-shot learning for pedestrian attribute recognition,** in EMNLP, 2018.
*L. Xiang, X. Jin, G. Ding, J. Han, and L. Li.*
[paper](https://www.ijcai.org/Proceedings/2019/0543.pdf)

1. **Few-shot video-to-video synthesis,** in NeurIPS, 2019.
*T.-C. Wang, M.-Y. Liu, A. Tao, G. Liu, J. Kautz, and B. Catanzaro.*
[paper](https://papers.nips.cc/paper/8746-few-shot-video-to-video-synthesis.pdf)
[code](https://github.com/NVlabs/few-shot-vid2vid)

1. **Few-shot object detection via feature reweighting,** in ICCV, 2019.
*B. Kang, Z. Liu, X. Wang, F. Yu, J. Feng, and T. Darrell.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.pdf)
[code](https://github.com/bingykang/Fewshot_Detection)

1. **Few-shot unsupervised image-to-image translation,** in ICCV, 2019.
*M.-Y. Liu, X. Huang, A. Mallya, T. Karras, T. Aila, J. Lehtinen, and J. Kautz.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Few-Shot_Unsupervised_Image-to-Image_Translation_ICCV_2019_paper.pdf)
[code](https://github.com/NVlabs/FUNIT)

1. **Feature weighting and boosting for few-shot segmentation,** in ICCV, 2019.
*K. Nguyen, and S. Todorovic.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Feature_Weighting_and_Boosting_for_Few-Shot_Segmentation_ICCV_2019_paper.pdf)

1. **Few-shot adaptive gaze estimation,** in ICCV, 2019.
*S. Park, S. D. Mello, P. Molchanov, U. Iqbal, O. Hilliges, and J. Kautz.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf)

1. **AMP: Adaptive masked proxies for few-shot segmentation,** in ICCV, 2019.
*M. Siam, B. N. Oreshkin, and M. Jagersand.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Siam_AMP_Adaptive_Masked_Proxies_for_Few-Shot_Segmentation_ICCV_2019_paper.pdf)
[code](https://github.com/MSiam/AdaptiveMaskedProxies)

1. **Few-shot generalization for single-image 3D reconstruction via priors,** in ICCV, 2019.
*B. Wallace, and B. Hariharan.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wallace_Few-Shot_Generalization_for_Single-Image_3D_Reconstruction_via_Priors_ICCV_2019_paper.pdf)

1. **Few-shot adversarial learning of realistic neural talking head models,** in ICCV, 2019.
*E. Zakharov, A. Shysheya, E. Burkov, and V. Lempitsky.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zakharov_Few-Shot_Adversarial_Learning_of_Realistic_Neural_Talking_Head_Models_ICCV_2019_paper.pdf)
[code](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models)

1. **Pyramid graph networks with connection attentions for region-based one-shot semantic segmentation,** in ICCV, 2019.
*C. Zhang, G. Lin, F. Liu, J. Guo, Q. Wu, and R. Yao.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Pyramid_Graph_Networks_With_Connection_Attentions_for_Region-Based_One-Shot_Semantic_ICCV_2019_paper.pdf)

1. **Time-conditioned action anticipation in one shot,** in CVPR, 2019.
*Q. Ke, M. Fritz, and B. Schiele.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ke_Time-Conditioned_Action_Anticipation_in_One_Shot_CVPR_2019_paper.pdf)

1. **Few-shot learning with localization in realistic settings,** in CVPR, 2019.
*D. Wertheimer, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wertheimer_Few-Shot_Learning_With_Localization_in_Realistic_Settings_CVPR_2019_paper.pdf)
[code](https://github.com/daviswer/fewshotlocal)

1. **Improving few-shot user-specific gaze adaptation via gaze redirection synthesis,** in CVPR, 2019.
*Y. Yu, G. Liu, and J.-M. Odobez.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Improving_Few-Shot_User-Specific_Gaze_Adaptation_via_Gaze_Redirection_Synthesis_CVPR_2019_paper.pdf)

1. **CANet: Class-agnostic segmentation networks with iterative refinement and attentive few-shot learning,** in CVPR, 2019.
*C. Zhang, G. Lin, F. Liu, R. Yao, and C. Shen.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_CANet_Class-Agnostic_Segmentation_Networks_With_Iterative_Refinement_and_Attentive_Few-Shot_CVPR_2019_paper.pdf)
[code](https://github.com/icoz69/CaNet)

1. **Multi-level Semantic Feature Augmentation for One-shot Learning,** in TIP, 2019.
*Z. Chen, Y. Fu, Y. Zhang, Y.-G. Jiang, X. Xue, and L. Sigal.*
[paper](https://arxiv.org/abs/1804.05298)
[code](https://github.com/tankche1/Semantic-Feature-Augmentation-in-Few-shot-Learning)

1. **3FabRec: Fast few-shot face alignment by reconstruction,** in CVPR, 2020.
*B. Browatzki, and C. Wallraven.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Browatzki_3FabRec_Fast_Few-Shot_Face_Alignment_by_Reconstruction_CVPR_2020_paper.pdf)

1. **Few-shot video classification via temporal alignment,** in CVPR, 2020.
*K. Cao, J. Ji, Z. Cao, C.-Y. Chang, J. C. Niebles.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf)

1. **One-shot adversarial attacks on visual tracking with dual attention,** in CVPR, 2020.
*X. Chen, X. Yan, F. Zheng, Y. Jiang, S.-T. Xia, Y. Zhao, and R. Ji.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_One-Shot_Adversarial_Attacks_on_Visual_Tracking_With_Dual_Attention_CVPR_2020_paper.pdf)

1. **FGN: Fully guided network for few-shot instance segmentation,** in CVPR, 2020.
*Z. Fan, J.-G. Yu, Z. Liang, J. Ou, C. Gao, G.-S. Xia, and Y. Li.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_FGN_Fully_Guided_Network_for_Few-Shot_Instance_Segmentation_CVPR_2020_paper.pdf)

1. **CRNet: Cross-reference networks for few-shot segmentation,** in CVPR, 2020.
*W. Liu, C. Zhang, G. Lin, and F. Liu.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_CRNet_Cross-Reference_Networks_for_Few-Shot_Segmentation_CVPR_2020_paper.pdf)

1. **Revisiting pose-normalization for fine-grained few-shot recognition,** in CVPR, 2020.
*L. Tang, D. Wertheimer, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_Revisiting_Pose-Normalization_for_Fine-Grained_Few-Shot_Recognition_CVPR_2020_paper.pdf)

1. **Few-shot learning of part-specific probability space for 3D shape segmentation,** in CVPR, 2020.
*L. Wang, X. Li, and Y. Fang.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Few-Shot_Learning_of_Part-Specific_Probability_Space_for_3D_Shape_Segmentation_CVPR_2020_paper.pdf)

1. **Semi-supervised learning for few-shot image-to-image translation,** in CVPR, 2020.
*Y. Wang, S. Khan, A. Gonzalez-Garcia, J. van de Weijer, and F. S. Khan.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Semi-Supervised_Learning_for_Few-Shot_Image-to-Image_Translation_CVPR_2020_paper.pdf)

1. **Multi-domain learning for accurate and few-shot color constancy,** in CVPR, 2020.
*J. Xiao, S. Gu, and L. Zhang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xiao_Multi-Domain_Learning_for_Accurate_and_Few-Shot_Color_Constancy_CVPR_2020_paper.pdf)

1. **One-shot domain adaptation for face generation,** in CVPR, 2020.
*C. Yang, and S.-N. Lim.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_One-Shot_Domain_Adaptation_for_Face_Generation_CVPR_2020_paper.pdf)

1. **MetaPix: Few-shot video retargeting,** in ICLR, 2020.
*J. Lee, D. Ramanan, and R. Girdhar.*
[paper](https://openreview.net/pdf?id=SJx1URNKwH)

1. **Few-shot human motion prediction via learning novel motion dynamics,** in IJCAI, 2020.
*C. Zang, M. Pei, and Y. Kong.*
[paper](https://www.ijcai.org/Proceedings/2020/0118.pdf)

1. **Shaping visual representations with language for few-shot classification,** in ACL, 2020.
*J. Mu, P. Liang, and N. D. Goodman.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.436.pdf)

1. **MarioNETte: Few-shot face reenactment preserving identity of unseen targets,** in AAAI, 2020.
*S. Ha, M. Kersner, B. Kim, S. Seo, and D. Kim.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6721)

1. **One-shot learning for long-tail visual relation detection,** in AAAI, 2020.
*W. Wang, M. Wang, S. Wang, G. Long, L. Yao, G. Qi, and Y. Chen.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6904)
[code](https://github.com/Witt-Wang/oneshot)

1. **Differentiable meta-learning model for few-shot semantic segmentation,** in AAAI, 2020.
*P. Tian, Z. Wu, L. Qi, L. Wang, Y. Shi, and Y. Gao.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6887)

1. **Part-aware prototype network for few-shot semantic segmentation,** in ECCV, 2020.
*Y. Liu, X. Zhang, S. Zhang, and X. He.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540137.pdf)
[code](https://github.com/Xiangyi1996/PPNet-PyTorch)

1. **Prototype mixture models for few-shot semantic segmentation,** in ECCV, 2020.
*B. Yang, C. Liu, B. Li, J. Jiao, and Q. Ye.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530749.pdf)
[code](https://github.com/Yang-Bob/PMMs)

1. **Few-shot action recognition with permutation-invariant attention,** in ECCV, 2020.
*H. Zhang, L. Zhang, X. Qi, H. Li, P. H. S. Torr, and P. Koniusz.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf)

1. **Few-shot compositional font generation with dual memory,** in ECCV, 2020.
*J. Cha, S. Chun, G. Lee, B. Lee, S. Kim, and H. Lee.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640715.pdf)
[code](https://github.com/clovaai/dmfont)

1. **Few-shot object detection and viewpoint estimation for objects in the wild,** in ECCV, 2020.
*Y. Xiao, and R. Marlet.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620188.pdf)

1. **Few-shot scene-adaptive anomaly detection,** in ECCV, 2020.
*Y. Lu, F. Yu, M. K. K. Reddy, and Y. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500120.pdf)
[code](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection)

1. **Few-shot semantic segmentation with democratic attention networks,** in ECCV, 2020.
*H. Wang, X. Zhang, Y. Hu, Y. Yang, X. Cao, and X. Zhen.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580715.pdf)

1. **Few-shot single-view 3-D object reconstruction with compositional priors,** in ECCV, 2020.
*M. Michalkiewicz, S. Parisot, S. Tsogkas, M. Baktashmotlagh, A. Eriksson, and E. Belilovsky.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700613.pdf)

1. **COCO-FUNIT: Few-shot unsupervised image translation with a content conditioned style encoder,** in ECCV, 2020.
*K. Saito, K. Saenko, and M. Liu.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480392.pdf)
[code](https://nvlabs.github.io/COCO-FUNIT/)



1. **Multi-scale positive sample refinement for few-shot object detection,** in ECCV, 2020.
*J. Wu, S. Liu, D. Huang, and Y. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610443.pdf)
[code](https://github.com/jiaxi-wu/MPSR)

1. **Large-scale few-shot learning via multi-modal knowledge discovery,** in ECCV, 2020.
*S. Wang, J. Yue, J. Liu, Q. Tian, and M. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550715.pdf)

1. **Graph convolutional networks for learning with few clean and many noisy labels,** in ECCV, 2020.
*A. Iscen, G. Tolias, Y. Avrithis, O. Chum, and C. Schmid.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550290.pdf)

1. **Self-supervised few-shot learning on point clouds,** in NeurIPS, 2020.
*C. Sharma, and M. Kaul.*
[paper](https://proceedings.neurips.cc/paper/2020/file/50c1f44e426560f3f2cdcb3e19e39903-Paper.pdf)
[code](https://github.com/charusharma1991/SSL_PointClouds)

1. **Restoring negative information in few-shot object detection,** in NeurIPS, 2020.
*Y. Yang, F. Wei, M. Shi, and G. Li.*
[paper](https://proceedings.neurips.cc/paper/2020/file/240ac9371ec2671ae99847c3ae2e6384-Paper.pdf)
[code](https://github.com/yang-yk/NP-RepMet)

1. **Few-shot image generation with elastic weight consolidation,** in NeurIPS, 2020.
*Y. Li, R. Zhang, J. Lu, and E. Shechtman.*
[paper](https://proceedings.neurips.cc/paper/2020/file/b6d767d2f8ed5d21a44b0e5886680cb9-Paper.pdf)

1. **Few-shot visual reasoning with meta-analogical contrastive learning,** in NeurIPS, 2020.
*Y. Kim, J. Shin, E. Yang, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/c39e1a03859f9ee215bc49131d0caf33-Paper.pdf)

1. **CrossTransformers: spatially-aware few-shot transfer,** in NeurIPS, 2020.
*C. Doersch, A. Gupta, and A. Zisserman.*
[paper](https://proceedings.neurips.cc/paper/2020/file/fa28c6cdf8dd6f41a657c3d7caa5c709-Paper.pdf)

1. **Make one-shot video object segmentation efficient again,** in NeurIPS, 2020.
*T. Meinhardt, and L. Leal-Taixé.*
[paper](https://proceedings.neurips.cc/paper/2020/file/781397bc0630d47ab531ea850bddcf63-Paper.pdf)
[code](https://github.com/dvl-tum/e-osvos)

1. **Frustratingly simple few-shot object detection,** in ICML, 2020.
*X. Wang, T. E. Huang, J. Gonzalez, T. Darrell, and F. Yu.*
[paper](http://proceedings.mlr.press/v119/wang20j/wang20j.pdf)
[code](https://github.com/ucbdrive/few-shot-object-detection)

1. **Adversarial style mining for one-shot unsupervised domain adaptation,** in NeurIPS, 2020.
*Y. Luo, P. Liu, T. Guan, J. Yu, and Y. Yang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/781397bc0630d47ab531ea850bddcf63-Paper.pdf)
[code](https://github.com/RoyalVane/ASM)

1. **Disentangling 3D prototypical networks for few-shot concept learning,** in ICLR, 2021.
*M. Prabhudesai, S. Lal, D. Patil, H. Tung, A. W. Harley, and K. Fragkiadaki.*
[paper](https://openreview.net/pdf?id=-Lr-u0b42he)

1. **Learning normal dynamics in videos with meta prototype network,** in CVPR, 2021.
*H. Lv, C. Chen, Z. Cui, C. Xu, Y. Li, and J. Yang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lv_Learning_Normal_Dynamics_in_Videos_With_Meta_Prototype_Network_CVPR_2021_paper.pdf)
[code](https://github.com/ktr-hubrt/MPN/)

1. **Learning dynamic alignment via meta-filter for few-shot learning,** in CVPR, 2021.
*C. Xu, Y. Fu, C. Liu, C. Wang, J. Li, F. Huang, L. Zhang, and X. Xue.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Learning_Dynamic_Alignment_via_Meta-Filter_for_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Delving deep into many-to-many attention for few-shot video object segmentation,** in CVPR, 2021.
*H. Chen, H. Wu, N. Zhao, S. Ren, and S. He.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Delving_Deep_Into_Many-to-Many_Attention_for_Few-Shot_Video_Object_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/scutpaul/DANet)

1. **Adaptive prototype learning and allocation for few-shot segmentation,** in CVPR, 2021.
*G. Li, V. Jampani, L. Sevilla-Lara, D. Sun, J. Kim, and J. Kim.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Adaptive_Prototype_Learning_and_Allocation_for_Few-Shot_Segmentation_CVPR_2021_paper.pdf)
[code](https://git.io/ASGNet)

1. **FAPIS: A few-shot anchor-free part-based instance segmenter,** in CVPR, 2021.
*K. Nguyen, and S. Todorovic.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_FAPIS_A_Few-Shot_Anchor-Free_Part-Based_Instance_Segmenter_CVPR_2021_paper.pdf)

1. **FSCE: Few-shot object detection via contrastive proposal encoding,** in CVPR, 2021.
*B. Sun, B. Li, S. Cai, Y. Yuan, and C. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_FSCE_Few-Shot_Object_Detection_via_Contrastive_Proposal_Encoding_CVPR_2021_paper.pdf)
[code](https://github.com/MegviiDetection/FSCE)

1. **Few-shot 3D point cloud semantic segmentation,** in CVPR, 2021.
*N. Zhao, T. Chua, and G. H. Lee.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Few-Shot_3D_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/Na-Z/attMPTI)

1. **Generalized few-shot object detection without forgetting,** in CVPR, 2021.
*Z. Fan, Y. Ma, Z. Li, and J. Sun.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.pdf)

1. **Few-shot human motion transfer by personalized geometry and texture modeling,** in CVPR, 2021.
*Z. Huang, X. Han, J. Xu, and T. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Few-Shot_Human_Motion_Transfer_by_Personalized_Geometry_and_Texture_Modeling_CVPR_2021_paper.pdf)
[code](https://github.com/HuangZhiChao95/FewShotMotionTransfer)

1. **Labeled from unlabeled: Exploiting unlabeled data for few-shot deep HDR deghosting,** in CVPR, 2021.
*K. R. Prabhakar, G. Senthil, S. Agrawal, R. V. Babu, and R. K. S. S. Gorthi.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Prabhakar_Labeled_From_Unlabeled_Exploiting_Unlabeled_Data_for_Few-Shot_Deep_HDR_CVPR_2021_paper.pdf)

1. **Few-shot transformation of common actions into time and space,** in CVPR, 2021.
*P. Yang, P. Mettes, and C. G. M. Snoek.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Few-Shot_Transformation_of_Common_Actions_Into_Time_and_Space_CVPR_2021_paper.pdf)
[code](https://github.com/PengWan-Yang/few-shot-transformer)

1. **Temporal-relational CrossTransformers for few-shot action recognition,** in CVPR, 2021.
*T. Perrett, A. Masullo, T. Burghardt, M. Mirmehdi, and D. Damen.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Perrett_Temporal-Relational_CrossTransformers_for_Few-Shot_Action_Recognition_CVPR_2021_paper.pdf)

1. **pixelNeRF: Neural radiance fields from one or few images,** in CVPR, 2021.
*A. Yu, V. Ye, M. Tancik, and A. Kanazawa.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yu_pixelNeRF_Neural_Radiance_Fields_From_One_or_Few_Images_CVPR_2021_paper.pdf)
[code](https://alexyu.net/pixelnerf)

1. **Hallucination improves few-shot object detection,** in CVPR, 2021.
*W. Zhang, and Y. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Hallucination_Improves_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)

1. **Few-shot object detection via classification refinement and distractor retreatment,** in CVPR, 2021.
*Y. Li, H. Zhu, Y. Cheng, W. Wang, C. S. Teo, C. Xiang, P. Vadakkepat, and T. H. Lee.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Few-Shot_Object_Detection_via_Classification_Refinement_and_Distractor_Retreatment_CVPR_2021_paper.pdf)

1. **Dense relation distillation with context-aware aggregation for few-shot object detection,** in CVPR, 2021.
*H. Hu, S. Bai, A. Li, J. Cui, and L. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)
[code](https://github.com/hzhupku/DCNet)

1. **Few-shot segmentation without meta-learning: A good transductive inference is all you need? ,** in CVPR, 2021.
*M. Boudiaf, H. Kervadec, Z. I. Masud, P. Piantanida, I. B. Ayed, and J. Dolz.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.pdf)
[code](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation)

1. **Few-shot image generation via cross-domain correspondence,** in CVPR, 2021.
*U. Ojha, Y. Li, J. Lu, A. A. Efros, Y. J. Lee, E. Shechtman, and R. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ojha_Few-Shot_Image_Generation_via_Cross-Domain_Correspondence_CVPR_2021_paper.pdf)

1. **Self-guided and cross-guided learning for few-shot segmentation,** in CVPR, 2021.
*B. Zhang, J. Xiao, and T. Qin.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Self-Guided_and_Cross-Guided_Learning_for_Few-Shot_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/zbf1991/SCL)

1. **Anti-aliasing semantic reconstruction for few-shot semantic segmentation,** in CVPR, 2021.
*B. Liu, Y. Ding, J. Jiao, X. Ji, and Q. Ye.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Anti-Aliasing_Semantic_Reconstruction_for_Few-Shot_Semantic_Segmentation_CVPR_2021_paper.pdf)

1. **Beyond max-margin: Class margin equilibrium for few-shot object detection,** in CVPR, 2021.
*B. Li, B. Yang, C. Liu, F. Liu, R. Ji, and Q. Ye.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Beyond_Max-Margin_Class_Margin_Equilibrium_for_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)
[code](https://github.com/Bohao-Lee/CME)

1. **Incremental few-shot instance segmentation,** in CVPR, 2021.
*D. A. Ganea, B. Boom, and R. Poppe.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ganea_Incremental_Few-Shot_Instance_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/danganea/iMTFA)

1. **Scale-aware graph neural network for few-shot semantic segmentation,** in CVPR, 2021.
*G. Xie, J. Liu, H. Xiong, and L. Shao.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Scale-Aware_Graph_Neural_Network_for_Few-Shot_Semantic_Segmentation_CVPR_2021_paper.pdf)

1. **Semantic relation reasoning for shot-stable few-shot object detection,** in CVPR, 2021.
*C. Zhu, F. Chen, U. Ahmed, Z. Shen, and M. Savvides.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Semantic_Relation_Reasoning_for_Shot-Stable_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)

1. **Accurate few-shot object detection with support-query mutual guidance and hybrid loss,** in CVPR, 2021.
*L. Zhang, S. Zhou, J. Guan, and J. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Accurate_Few-Shot_Object_Detection_With_Support-Query_Mutual_Guidance_and_Hybrid_CVPR_2021_paper.pdf)

1. **Transformation invariant few-shot object detection,** in CVPR, 2021.
*A. Li, and Z. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Transformation_Invariant_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)

1. **MetaHTR: Towards writer-adaptive handwritten text recognition,** in CVPR, 2021.
*A. K. Bhunia, S. Ghose, A. Kumar, P. N. Chowdhury, A. Sain, and Y. Song.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhunia_MetaHTR_Towards_Writer-Adaptive_Handwritten_Text_Recognition_CVPR_2021_paper.pdf)

1. **What if we only use real datasets for scene text recognition? Toward scene text recognition with fewer labels,** in CVPR, 2021.
*J. Baek, Y. Matsui, and K. Aizawa.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf)
[code](https://github.com/ku21fan/STR-Fewer-Labels)

1. **Few-shot font generation with localized style representations and factorization,** in AAAI, 2021.
*S. Park, S. Chun, J. Cha, B. Lee, and H. Shim.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16340/16147)
[code](https://github.com/clovaai/lffont)

1. **Attributes-guided and pure-visual attention alignment for few-shot recognition,** in AAAI, 2021.
*S. Huang, M. Zhang, Y. Kang, and D. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16957/16764)
[code](https://github.com/bighuang624/AGAM)

1. **One-shot face reenactment using appearance adaptive normalization,** in AAAI, 2021.
*G. Yao, Y. Yuan, T. Shao, S. Li, S. Liu, Y. Liu, M. Wang, and K. Zhou.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16427/16234)

1. **FL-MSRE: A few-shot learning based approach to multimodal social relation extraction,** in AAAI, 2021.
*H. Wan, M. Zhang, J. Du, Z. Huang, Y. Yang, and J. Z. Pan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17639/17446)
[code](https://github.com/sysulic/FL-MSRE)

1. **StarNet: Towards weakly supervised few-shot object detection,** in AAAI, 2021.
*L. Karlinsky, J. Shtok, A. Alfassy, M. Lichtenstein, S. Harary, E. Schwartz, S. Doveh, P. Sattigeri, R. Feris, A. Bronstein, and R. Giryes.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16268/16075)
[code](https://github.com/leokarlin/StarNet)

1. **Progressive one-shot human parsing,** in AAAI, 2021.
*H. He, J. Zhang, B. Thuraisingham, and D. Tao.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16243/16050)
[code](https://github.com/Charleshhy/One-shot-Human-Parsing)

1. **Knowledge is power: Hierarchical-knowledge embedded meta-learning for visual reasoning in artistic domains,** in KDD, 2021.
*W. Zheng, L. Yan, C. Gou, and F.-Y. Wang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467285)

1. **MEDA: Meta-learning with data augmentation for few-shot text classification,** in IJCAI, 2021.
*P. Sun, Y. Ouyang, W. Zhang, and X.-Y. Dai.*
[paper](https://www.ijcai.org/proceedings/2021/0541.pdf)

1. **Learning implicit temporal alignment for few-shot video classification,** in IJCAI, 2021.
*S. Zhang, J. Zhou, and X. He.*
[paper](https://www.ijcai.org/proceedings/2021/0181.pdf)
[code](https://github.com/tonysy/PyAction)

1. **Few-shot neural human performance rendering from sparse RGBD videos,** in IJCAI, 2021.
*A. Pang, X. Chen, H. Luo, M. Wu, J. Yu, and L. Xu.*
[paper](https://www.ijcai.org/proceedings/2021/0130.pdf)

1. **Uncertainty-aware few-shot image classification,** in IJCAI, 2021.
*Z. Zhang, C. Lan, W. Zeng, Z. Chen, and S. Chan.*
[paper](https://www.ijcai.org/proceedings/2021/0471.pdf)

1. **Few-shot learning with part discovery and augmentation from unlabeled images,** in IJCAI, 2021.
*W. Chen, C. Si, W. Wang, L. Wang, Z. Wang, and T. Tan.*
[paper](https://www.ijcai.org/proceedings/2021/0313.pdf)

1. **Few-shot partial-label learning,** in IJCAI, 2021.
*Y. Zhao, G. Yu, L. Liu, Z. Yan, L. Cui, and C. Domeniconi.*
[paper](https://www.ijcai.org/proceedings/2021/0475.pdf)

1. **One-shot affordance detection,** in IJCAI, 2021.
*H. Luo, W. Zhai, J. Zhang, Y. Cao, and D. Tao.*
[paper](https://www.ijcai.org/proceedings/2021/0124.pdf)

1. **DeFRCN: Decoupled faster R-CNN for few-shot object detection,** in ICCV, 2021.
*L. Qiao, Y. Zhao, Z. Li, X. Qiu, J. Wu, and C. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiao_DeFRCN_Decoupled_Faster_R-CNN_for_Few-Shot_Object_Detection_ICCV_2021_paper.pdf)

1. **Learning meta-class memory for few-shot semantic segmentation,** in ICCV, 2021.
*Z. Wu, X. Shi, G. Lin, and J. Cai.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Meta-Class_Memory_for_Few-Shot_Semantic_Segmentation_ICCV_2021_paper.pdf)

1. **UVStyle-Net: Unsupervised few-shot learning of 3D style similarity measure for B-Reps,** in ICCV, 2021.
*P. Meltzer, H. Shayani, A. Khasahmadi, P. K. Jayaraman, A. Sanghi, and J. Lambourne.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Meltzer_UVStyle-Net_Unsupervised_Few-Shot_Learning_of_3D_Style_Similarity_Measure_for_ICCV_2021_paper.pdf)

1. **LoFGAN: Fusing local representations for few-shot image generation,** in ICCV, 2021.
*Z. Gu, W. Li, J. Huo, L. Wang, and Y. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_LoFGAN_Fusing_Local_Representations_for_Few-Shot_Image_Generation_ICCV_2021_paper.pdf)

1. **H3D-Net: Few-shot high-fidelity 3D head reconstruction,** in ICCV, 2021.
*E. Ramon, G. Triginer, J. Escur, A. Pumarola, J. Garcia, X. Giró-i-Nieto, and F. Moreno-Noguer.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ramon_H3D-Net_Few-Shot_High-Fidelity_3D_Head_Reconstruction_ICCV_2021_paper.pdf)

1. **Learned spatial representations for few-shot talking-head synthesis,** in ICCV, 2021.
*M. Meshry, S. Suri, L. S. Davis, and A. Shrivastava.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Meshry_Learned_Spatial_Representations_for_Few-Shot_Talking-Head_Synthesis_ICCV_2021_paper.pdf)

1. **Putting NeRF on a diet: Semantically consistent few-shot view synthesis,** in ICCV, 2021.
*A. Jain, M. Tancik, and P. Abbeel.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.pdf)

1. **Hypercorrelation squeeze for few-shot segmentation,** in ICCV, 2021.
*J. Min, D. Kang, and M. Cho.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Hypercorrelation_Squeeze_for_Few-Shot_Segmentation_ICCV_2021_paper.pdf)
[code](https://github.com/juhongm999/hsnet)

1. **Few-shot semantic segmentation with cyclic memory network,** in ICCV, 2021.
*G. Xie, H. Xiong, J. Liu, Y. Yao, and L. Shao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Few-Shot_Semantic_Segmentation_With_Cyclic_Memory_Network_ICCV_2021_paper.pdf)

1. **Simpler is better: Few-shot semantic segmentation with classifier weight transformer,** in ICCV, 2021.
*Z. Lu, S. He, X. Zhu, L. Zhang, Y. Song, and T. Xiang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Simpler_Is_Better_Few-Shot_Semantic_Segmentation_With_Classifier_Weight_Transformer_ICCV_2021_paper.pdf)
[code](https://github.com/zhiheLu/CWT-for-FSS)

1. **Unsupervised few-shot action recognition via action-appearance aligned meta-adaptation,** in ICCV, 2021.
*J. Patravali, G. Mittal, Y. Yu, F. Li, and M. Chen.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Patravali_Unsupervised_Few-Shot_Action_Recognition_via_Action-Appearance_Aligned_Meta-Adaptation_ICCV_2021_paper.pdf)

1. **Multiple heads are better than one: few-shot font generation with multiple localized experts,** in ICCV, 2021.
*S. Park, S. Chun, J. Cha, B. Lee, and H. Shim.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Multiple_Heads_Are_Better_Than_One_Few-Shot_Font_Generation_With_ICCV_2021_paper.pdf)
[code](https://github.com/clovaai/mxfont)

1. **Mining latent classes for few-shot segmentation,** in ICCV, 2021.
*L. Yang, W. Zhuo, L. Qi, Y. Shi, and Y. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Mining_Latent_Classes_for_Few-Shot_Segmentation_ICCV_2021_paper.pdf)
[code](https://github.com/LiheYoung/MiningFSS)

1. **Partner-assisted learning for few-shot image classification,** in ICCV, 2021.
*J. Ma, H. Xie, G. Han, S. Chang, A. Galstyan, and W. Abd-Almageed.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ma_Partner-Assisted_Learning_for_Few-Shot_Image_Classification_ICCV_2021_paper.pdf)

1. **Hierarchical graph attention network for few-shot visual-semantic learning,** in ICCV, 2021.
*C. Yin, K. Wu, Z. Che, B. Jiang, Z. Xu, and J. Tang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yin_Hierarchical_Graph_Attention_Network_for_Few-Shot_Visual-Semantic_Learning_ICCV_2021_paper.pdf)

1. **Video pose distillation for few-shot, fine-grained sports action recognition,** in ICCV, 2021.
*J. Hong, M. Fisher, M. Gharbi, and K. Fatahalian.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Hong_Video_Pose_Distillation_for_Few-Shot_Fine-Grained_Sports_Action_Recognition_ICCV_2021_paper.pdf)

1. **Universal-prototype enhancing for few-shot object detection,** in ICCV, 2021.
*A. Wu, Y. Han, L. Zhu, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Universal-Prototype_Enhancing_for_Few-Shot_Object_Detection_ICCV_2021_paper.pdf)
[code](https://github.com/amingwu/up-fsod)

1. **Query adaptive few-shot object detection with heterogeneous graph convolutional networks,** in ICCV, 2021.
*G. Han, Y. He, S. Huang, J. Ma, and S. Chang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Han_Query_Adaptive_Few-Shot_Object_Detection_With_Heterogeneous_Graph_Convolutional_Networks_ICCV_2021_paper.pdf)

1. **Few-shot visual relationship co-localization,** in ICCV, 2021.
*R. Teotia, V. Mishra, M. Maheshwari, and A. Mishra.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Teotia_Few-Shot_Visual_Relationship_Co-Localization_ICCV_2021_paper.pdf)
[code](https://github.com/vl2g/VRC)

1. **Shallow Bayesian meta learning for real-world few-shot recognition,** in ICCV, 2021.
*X. Zhang, D. Meng, H. Gouk, and T. M. Hospedales.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Shallow_Bayesian_Meta_Learning_for_Real-World_Few-Shot_Recognition_ICCV_2021_paper.pdf)
[code](https://github.com/open-debin/bayesian_mqda)

1. **Super-resolving cross-domain face miniatures by peeking at one-shot exemplar,** in ICCV, 2021.
*P. Li, X. Yu, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Super-Resolving_Cross-Domain_Face_Miniatures_by_Peeking_at_One-Shot_Exemplar_ICCV_2021_paper.pdf)

1. **Few-shot segmentation via cycle-consistent transformer,** in NeurIPS, 2021.
*G. Zhang, G. Kang, Y. Yang, and Y. Wei.*
[paper](https://proceedings.neurips.cc/paper/2021/file/b8b12f949378552c21f28deff8ba8eb6-Paper.pdf)

1. **Generalized and discriminative few-shot object detection via SVD-dictionary enhancement,** in NeurIPS, 2021.
*A. WU, S. Zhao, C. Deng, and W. Liu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/325995af77a0e8b06d1204a171010b3a-Paper.pdf)

1. **Re-ranking for image retrieval and transductive few-shot classification,** in NeurIPS, 2021.
*X. SHEN, Y. Xiao, S. Hu, O. Sbai, and M. Aubry.*
[paper](https://proceedings.neurips.cc/paper/2021/file/d9fc0cdb67638d50f411432d0d41d0ba-Paper.pdf)

1. **Neural view synthesis and matching for semi-supervised few-shot learning of 3D pose,** in NeurIPS, 2021.
*A. Wang, S. Mei, A. L. Yuille, and A. Kortylewski.*
[paper](https://proceedings.neurips.cc/paper/2021/file/3a61ed715ee66c48bacf237fa7bb5289-Paper.pdf)

1. **MetaAvatar: Learning animatable clothed human models from few depth images,** in NeurIPS, 2021.
*S. Wang, M. Mihajlovic, Q. Ma, A. Geiger, and S. Tang.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1680829293f2a8541efa2647a0290f88-Paper.pdf)

1. **Few-shot object detection via association and discrimination,** in NeurIPS, 2021.
*Y. Cao, J. Wang, Y. Jin, T. Wu, K. Chen, Z. Liu, and D. Lin.*
[paper](https://proceedings.neurips.cc/paper/2021/file/8a1e808b55fde9455cb3d8857ed88389-Paper.pdf)

1. **Rectifying the shortcut learning of background for few-shot learning,** in NeurIPS, 2021.
*X. Luo, L. Wei, L. Wen, J. Yang, L. Xie, Z. Xu, and Q. Tian.*
[paper](https://proceedings.neurips.cc/paper/2021/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf)

1. **D2C: Diffusion-decoding models for few-shot conditional generation,** in NeurIPS, 2021.
*A. Sinha, J. Song, C. Meng, and S. Ermon.*
[paper](https://proceedings.neurips.cc/paper/2021/file/682e0e796084e163c5ca053dd8573b0c-Paper.pdf)

1. **Few-shot backdoor attacks on visual object tracking,** in ICLR, 2022.
*Y. Li, H. Zhong, X. Ma, Y. Jiang, and S. Xia.*
[paper](https://openreview.net/pdf?id=qSV5CuSaK_a)
[code](https://github.com/HXZhong1997/FSBA)

1. **Temporal alignment prediction for supervised representation learning and few-shot sequence classification,** in ICLR, 2022.
*B. Su, and J. Wen.*
[paper](https://openreview.net/pdf?id=p3DKPQ7uaAi)
[code](https://github.com/BingSu12/TAP)

1. **Learning non-target knowledge for few-shot semantic segmentation,** in CVPR, 2022.
*Y. Liu, N. Liu, Q. Cao, X. Yao, J. Han, and L. Shao.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Learning_Non-Target_Knowledge_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf)

1. **Learning what not to segment: A new perspective on few-shot segmentation,** in CVPR, 2022.
*C. Lang, G. Cheng, B. Tu, and J. Han.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lang_Learning_What_Not_To_Segment_A_New_Perspective_on_Few-Shot_CVPR_2022_paper.pdf)
[code](https://github.com/chunbolang/BAM)

1. **Few-shot keypoint detection with uncertainty learning for unseen species,** in CVPR, 2022.
*C. Lu, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Few-Shot_Keypoint_Detection_With_Uncertainty_Learning_for_Unseen_Species_CVPR_2022_paper.pdf)

1. **XMP-Font: Self-supervised cross-modality pre-training for few-shot font generation,** in CVPR, 2022.
*W. Liu, F. Liu, F. Ding, Q. He, and Z. Yi.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_XMP-Font_Self-Supervised_Cross-Modality_Pre-Training_for_Few-Shot_Font_Generation_CVPR_2022_paper.pdf)

1. **Spatio-temporal relation modeling for few-shot action recognition,** in CVPR, 2022.
*A. Thatipelli, S. Narayan, S. Khan, R. M. Anwer, F. S. Khan, and B. Ghanem.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Thatipelli_Spatio-Temporal_Relation_Modeling_for_Few-Shot_Action_Recognition_CVPR_2022_paper.pdf)
[code](https://github.com/Anirudh257/strm)

1. **Attribute group editing for reliable few-shot image generation,** in CVPR, 2022.
*G. Ding, X. Han, S. Wang, S. Wu, X. Jin, D. Tu, and Q. Huang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Attribute_Group_Editing_for_Reliable_Few-Shot_Image_Generation_CVPR_2022_paper.pdf)
[code](https://github.com/UniBester/AGE)

1. **Few-shot backdoor defense using Shapley estimation,** in CVPR, 2022.
*J. Guan, Z. Tu, R. He, and D. Tao.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guan_Few-Shot_Backdoor_Defense_Using_Shapley_Estimation_CVPR_2022_paper.pdf)

1. **Hybrid relation guided set matching for few-shot action recognition,** in CVPR, 2022.
*X. Wang, S. Zhang, Z. Qing, M. Tang, Z. Zuo, C. Gao, R. Jin, and N. Sang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Hybrid_Relation_Guided_Set_Matching_for_Few-Shot_Action_Recognition_CVPR_2022_paper.pdf)
[code](https://hyrsm-cvpr2022.github.io/)

1. **Label, verify, correct: A simple few shot object detection method,** in CVPR, 2022.
*P. Kaul, W. Xie, and A. Zisserman.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kaul_Label_Verify_Correct_A_Simple_Few_Shot_Object_Detection_Method_CVPR_2022_paper.pdf)

1. **InfoNeRF: Ray entropy minimization for few-shot neural volume rendering,** in CVPR, 2022.
*M. Kim, S. Seo, and B. Han.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_InfoNeRF_Ray_Entropy_Minimization_for_Few-Shot_Neural_Volume_Rendering_CVPR_2022_paper.pdf)

1. **A closer look at few-shot image generation,** in CVPR, 2022.
*Y. Zhao, H. Ding, H. Huang, and N. Cheung.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_A_Closer_Look_at_Few-Shot_Image_Generation_CVPR_2022_paper.pdf)
[code](https://github.com/mseitzer/pytorch-fid)

1. **Motion-modulated temporal fragment alignment network for few-shot action recognition,** in CVPR, 2022.
*J. Wu, T. Zhang, Z. Zhang, F. Wu, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Motion-Modulated_Temporal_Fragment_Alignment_Network_for_Few-Shot_Action_Recognition_CVPR_2022_paper.pdf)

1. **Kernelized few-shot object detection with efficient integral aggregation,** in CVPR, 2022.
*S. Zhang, L. Wang, N. Murray, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Kernelized_Few-Shot_Object_Detection_With_Efficient_Integral_Aggregation_CVPR_2022_paper.pdf)
[code](https://github.com/ZS123-lang/KFSOD)

1. **FS6D: Few-shot 6D pose estimation of novel objects,** in CVPR, 2022.
*Y. He, Y. Wang, H. Fan, J. Sun, and Q. Chen.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_FS6D_Few-Shot_6D_Pose_Estimation_of_Novel_Objects_CVPR_2022_paper.pdf)

1. **Look closer to supervise better: One-shot font generation via component-based discriminator,** in CVPR, 2022.
*Y. Kong, C. Luo, W. Ma, Q. Zhu, S. Zhu, N. Yuan, and L. Jin.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_Look_Closer_To_Supervise_Better_One-Shot_Font_Generation_via_Component-Based_CVPR_2022_paper.pdf)

1. **Generalized few-shot semantic segmentation,** in CVPR, 2022.
*Z. Tian, X. Lai, L. Jiang, S. Liu, M. Shu, H. Zhao, and J. Jia.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tian_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf)
[code](https://github.com/dvlab-research/GFS-Seg)

1. **Dynamic prototype convolution network for few-shot semantic segmentation,** in CVPR, 2022.
*J. Liu, Y. Bao, G. Xie, H. Xiong, J. Sonke, and E. Gavves.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Dynamic_Prototype_Convolution_Network_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf)

1. **OSOP: A multi-stage one shot object pose estimation framework,** in CVPR, 2022.
*I. Shugurov, F. Li, B. Busam, and S. Ilic.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shugurov_OSOP_A_Multi-Stage_One_Shot_Object_Pose_Estimation_Framework_CVPR_2022_paper.pdf)

1. **Semantic-aligned fusion transformer for one-shot object detection,** in CVPR, 2022.
*Y. Zhao, X. Guo, and Y. Lu.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Semantic-Aligned_Fusion_Transformer_for_One-Shot_Object_Detection_CVPR_2022_paper.pdf)

1. **OnePose: One-shot object pose estimation without CAD models,** in CVPR, 2022.
*J. Sun, Z. Wang, S. Zhang, X. He, H. Zhao, G. Zhang, and X. Zhou.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_OnePose_One-Shot_Object_Pose_Estimation_Without_CAD_Models_CVPR_2022_paper.pdf)
[code](https://zju3dv.github.io/onepose/)

1. **Few-shot object detection with fully cross-transformer,** in CVPR, 2022.
*G. Han, J. Ma, S. Huang, L. Chen, and S. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Few-Shot_Object_Detection_With_Fully_Cross-Transformer_CVPR_2022_paper.pdf)

1. **Learning to memorize feature hallucination for one-shot image generation,** in CVPR, 2022.
*Y. Xie, Y. Fu, Y. Tai, Y. Cao, J. Zhu, and C. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Learning_To_Memorize_Feature_Hallucination_for_One-Shot_Image_Generation_CVPR_2022_paper.pdf)

1. **Few-shot font generation by learning fine-grained local styles,** in CVPR, 2022.
*L. Tang, Y. Cai, J. Liu, Z. Hong, M. Gong, M. Fan, J. Han, J. Liu, E. Ding, and J. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Few-Shot_Font_Generation_by_Learning_Fine-Grained_Local_Styles_CVPR_2022_paper.pdf)

1. **Balanced and hierarchical relation learning for one-shot object detection,** in CVPR, 2022.
*H. Yang, S. Cai, H. Sheng, B. Deng, J. Huang, X. Hua, Y. Tang, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Balanced_and_Hierarchical_Relation_Learning_for_One-Shot_Object_Detection_CVPR_2022_paper.pdf)

1. **Few-shot head swapping in the wild,** in CVPR, 2022.
*C. Shu, H. Wu, H. Zhou, J. Liu, Z. Hong, C. Ding, J. Han, J. Liu, E. Ding, and J. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shu_Few-Shot_Head_Swapping_in_the_Wild_CVPR_2022_paper.pdf)

1. **Integrative few-shot learning for classification and segmentation,** in CVPR, 2022.
*D. Kang, and M. Cho.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_Integrative_Few-Shot_Learning_for_Classification_and_Segmentation_CVPR_2022_paper.pdf)

1. **Attribute surrogates learning and spectral tokens pooling in transformers for few-shot learning,** in CVPR, 2022.
*Y. He, W. Liang, D. Zhao, H. Zhou, W. Ge, Y. Yu, and W. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Attribute_Surrogates_Learning_and_Spectral_Tokens_Pooling_in_Transformers_for_CVPR_2022_paper.pdf)
[code](https://github.com/StomachCold/HCTransformers)

1. **Task discrepancy maximization for fine-grained few-shot classification,** in CVPR, 2022.
*S. Lee, W. Moon, and J. Heo.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Task_Discrepancy_Maximization_for_Fine-Grained_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **Channel importance matters in few-shot image classification,** in ICML, 2022.
*X. Luo, J. Xu, and Z. Xu.*
[paper](https://proceedings.mlr.press/v162/luo22c/luo22c.pdf)

1. **Long-short term cross-transformer in compressed domain for few-shot video classification,** in IJCAI, 2022.
*W. Luo, Y. Liu, B. Li, W. Hu, Y. Miao, and Y. Li.*
[paper](https://www.ijcai.org/proceedings/2022/0174.pdf)

1. **HifiHead: One-shot high fidelity neural head synthesis with 3D control,** in IJCAI, 2022.
*F. Zhu, J. Zhu, W. Chu, Y. Tai, Z. Xie, X. Huang, and C. Wang.*
[paper](https://www.ijcai.org/proceedings/2022/0244.pdf)
[code](https://github.com/TencentYoutuResearch/HeadSynthesis-HifHea)

1. **Iterative few-shot semantic segmentation from image label text,** in IJCAI, 2022.
*H. Wang, L. Liu, W. Zhang, J. Zhang, Z. Gan, Y. Wang, C. Wang, and H. Wang.*
[paper](https://www.ijcai.org/proceedings/2022/0193.pdf)
[code](https://github.com/Whileherham/IMR-HSNet)

1. **Beyond the prototype: Divide-and-conquer proxies for few-shot segmentation,** in IJCAI, 2022.
*C. Lang, B. Tu, G. Cheng, and J. Han.*
[paper](https://www.ijcai.org/proceedings/2022/0143.pdf)
[code](https://github.com/chunbolang/DCP)

1. **CATrans: Context and affinity transformer for few-shot segmentation,** in IJCAI, 2022.
*S. Zhang, T. Wu, S. Wu, and G. Guo.*
[paper](https://www.ijcai.org/proceedings/2022/0231.pdf)

1. **Masked feature generation network for few-shot learning,** in IJCAI, 2022.
*Y. Yu, D. Zhang, and Z. Ji.*
[paper](https://www.ijcai.org/proceedings/2022/0513.pdf)

1. **Decoupling classifier for boosting few-shot object detection and instance segmentation,** in NeurIPS, 2022.
*B.-B. Gao, X. Chen, Z. Huang, C. Nie, J. Liu, J. Lai, G. JIANG, X. Wang, and C. Wang.*
[paper](https://openreview.net/pdf?id=dVXO3Orjmxk)
[code](https://csgaobb.github.io/Projects/DCFS)

1. **Searching for better spatio-temporal alignment in few-shot action recognition,** in NeurIPS, 2022.
*Y. Cao, X. Su, Q. Tang, S. You, X. Lu, and C. Xu.*
[paper](https://openreview.net/pdf?id=IlYS1pLa9y)

1. **Feature-proxy transformer for few-shot segmentation,** in NeurIPS, 2022.
*J.-W. Zhang, Y. Sun, Y. Yang, and W. Chen.*
[paper](https://openreview.net/pdf?id=hBaI5MY0CBz)
[code](https://github.com/Jarvis73/FPTrans)

1. **Intermediate prototype mining transformer for few-shot semantic segmentation,** in NeurIPS, 2022. 
*Y. liu, N. Liu, X. Yao, J. Han,*
[paper](https://openreview.net/pdf?id=NyAJzgHLAr)
[code](https://github.com/LIUYUANWEI98/IPMT)

1. **OnePose++: Keypoint-free one-shot object pose estimation without CAD models,** in NeurIPS, 2022.
*X. He, J. Sun, Y. Wang, D. Huang, H. Bao, and X. Zhou.*
[paper](https://openreview.net/pdf?id=BZ92dxDS3tO)
[code](https://zju3dv.github.io/onepose_plus_plus/)

1. **Mask matching transformer for few-shot segmentation,** in NeurIPS, 2022.
*S. Jiao, G. Zhang, S. Navasardyan, L. Chen, Y. Zhao, Y. Wei, and H. Shi.*
[paper](https://openreview.net/pdf?id=zt4xNo0lF8W)
[code](https://github.com/Picsart-AI-Research/Mask-Matching-Transformer)

1. **Learning dense object descriptors from multiple views for low-shot category generalization,** in NeurIPS, 2022.
*S. Stojanov, N. A. Thai, Z. Huang, and J. M. Rehg.*
[paper](https://openreview.net/pdf?id=KJemAi9fymT)
[code](https://github.com/rehg-lab/dope_selfsup)

1. **Pose adaptive dual mixup for few-shot single-view 3D reconstruction,** in AAAI, 2022.
*T. Y. Cheng, H.-R. Yang, N. Trigoni, H.-T. Chen, and T.-L. Liu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19920/19679)

1. **Meta faster R-CNN: Towards accurate few-shot object detection with attentive feature alignment,** in AAAI, 2022.
*G. Han, S. Huang, J. Ma, Y. He, and S.-F. Chang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19959/19718)
[code](https://github.com/GuangxingHan/Meta-Faster-R-CNN)

1. **TA2N: Two-stage action alignment network for few-shot action recognition,** in AAAI, 2022.
*S. Li, H. Liu, R. Qian, Y. Li, J. See, M. Fei, X. Yu, and W. Lin.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20029/19788)

1. **Learning from the target: Dual prototype network for few shot semantic segmentation,** in AAAI, 2022.
*B. Mao, X. Zhang, L. Wang, Q. Zhang, S. Xiang, and C. Pan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20090/19849)

1. **OA-FSUI2IT: A novel few-shot cross domain object detection framework with object-aware few-shot unsupervised image-to-image translation,** in AAAI, 2022.
*L. Zhao, Y. Meng, and L. Xu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20253)

1. **When facial expression recognition meets few-shot learning: A joint and alternate learning framework,** in AAAI, 2022.
*X. Zou, Y. Yan, J.-H. Xue, S. Chen, and H. Wang.*
[paper](https://arxiv.org/abs/2201.06781)

1. **Dual attention networks for few-shot fine-grained recognition,** in AAAI, 2022.
*S.-L. Xu, F. Zhang, X.-S. Wei, and J. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20196)

1. **Inferring prototypes for multi-label few-shot image classification with word vector guided attention,** in AAAI, 2022.
*K. Yan, C. Zhang, J. Hou, P. Wang, Z. Bouraoui, S. Jameel, and S. Schockaert.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20205/19964)

1. **Analogy-forming transformers for few-shot 3D parsing,** in ICLR, 2023.
*N. Gkanatsios, M. Singh, Z. Fang, S. Tulsiani, and K. Fragkiadaki.*
[paper](https://openreview.net/pdf?id=SRIQZTh0IK)
[code](http://analogicalnets.github.io)

1. **Suppressing the heterogeneity: A strong feature extractor for few-shot segmentation,** in ICLR, 2023.
*Z. Hu, Y. Sun, and Y. Yang.*
[paper](https://openreview.net/pdf?id=CGuvK3U09LH)

1. **Universal few-shot learning of dense prediction tasks with visual token matching,** in ICLR, 2023.
*D. Kim, J. Kim, S. Cho, C. Luo, and S. Hong.*
[paper](https://openreview.net/pdf?id=88nT0j5jAn)
[code](https://github.com/GitGyun/visual_token_matching)

1. **Meta learning to bridge vision and language models for multimodal few-shot learning,** in ICLR, 2023.
*I. Najdenkoska, X. Zhen, and M. Worring.*
[paper](https://openreview.net/pdf?id=3oWo92cQyxL)
[code](https://github.com/ivonajdenkoska/multimodal-meta-learn)

1. **Few-shot semantic image synthesis with class affinity transfer,** in CVPR, 2023.
*M. Careil, J. Verbeek, and S. Lathuilière.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Careil_Few-Shot_Semantic_Image_Synthesis_With_Class_Affinity_Transfer_CVPR_2023_paper.pdf)

1. **Semantic prompt for few-shot image recognition,** in CVPR, 2023.
*W. Chen, C. Si, Z. Zhang, L. Wang, Z. Wang, and T. Tan.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Semantic_Prompt_for_Few-Shot_Image_Recognition_CVPR_2023_paper.pdf)

1. **ViewNet: A novel projection-based backbone with view pooling for few-shot point cloud classification,** in CVPR, 2023.
*J. Chen, M. Yang, and S. Velipasalar.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_ViewNet_A_Novel_Projection-Based_Backbone_With_View_Pooling_for_Few-Shot_CVPR_2023_paper.pdf)

1. **Meta-tuning loss functions and data augmentation for few-shot object detection,** in CVPR, 2023.
*B. Demirel, O. B. Baran, and R. G. Cinbis.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Demirel_Meta-Tuning_Loss_Functions_and_Data_Augmentation_for_Few-Shot_Object_Detection_CVPR_2023_paper.pdf)

1. **Few-shot geometry-aware keypoint localization,** in CVPR, 2023.
*X. He, G. Bharaj, D. Ferman, H. Rhodin, and P. Garrido.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/He_Few-Shot_Geometry-Aware_Keypoint_Localization_CVPR_2023_paper.pdf)
[code](https://xingzhehe.github.io/FewShot3DKP/)



1. **AsyFOD: An asymmetric adaptation paradigm for few-shot domain adaptive object detection,** in CVPR, 2023.
*Y. Gao, K.-Y. Lin, J. Yan, Y. Wang, and W.-S. Zheng.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object_CVPR_2023_paper.pdf)
[code](https://github.com/Hlings/AsyFOD)

1. **NIFF: Alleviating forgetting in generalized few-shot object detection via neural instance feature forging,** in CVPR, 2023.
*K. Guirguis, J. Meier, G. Eskandar, M. Kayser, B. Yang, and J. Beyerer.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guirguis_NIFF_Alleviating_Forgetting_in_Generalized_Few-Shot_Object_Detection_via_Neural_CVPR_2023_paper.pdf)

1. **A strong baseline for generalized few-shot semantic segmentation,** in CVPR, 2023.
*S. Hajimiri, M. Boudiaf, I. B. Ayed, and J. Dolz.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hajimiri_A_Strong_Baseline_for_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2023_paper.pdf)
[code](https://github.com/sinahmr/DIaM)

1. **StyleAdv: Meta style adversarial training for cross-domain few-shot learning,** in CVPR, 2023.
*Y. Fu, Y. Xie, Y. Fu, and Y.-G. Jiang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_StyleAdv_Meta_Style_Adversarial_Training_for_Cross-Domain_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/lovelyqian/StyleAdv-CDFSL)

1. **BlendFields: Few-shot example-driven facial modeling,** in CVPR, 2023.
*K. Kania, S. J. Garbin, A. Tagliasacchi, V. Estellers, K. M. Yi, J. Valentin, T. Trzcinski, and M. Kowalski.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kania_BlendFields_Few-Shot_Example-Driven_Facial_Modeling_CVPR_2023_paper.pdf)

1. **Learning orthogonal prototypes for generalized few-shot semantic segmentation,** in CVPR, 2023.
*S. Liu, Y. Zhang, Z. Qiu, H. Xie, Y. Zhang, and T. Yao.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Learning_Orthogonal_Prototypes_for_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2023_paper.pdf)

1. **DiGeo: Discriminative geometry-aware learning for generalized few-shot object detection,** in CVPR, 2023.
*J. Ma, Y. Niu, J. Xu, S. Huang, G. Han, and S.-F. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_DiGeo_Discriminative_Geometry-Aware_Learning_for_Generalized_Few-Shot_Object_Detection_CVPR_2023_paper.pdf)
[code](https://github.com/Phoenix-V/DiGeo)

1. **Hierarchical dense correlation distillation for few-shot segmentation,** in CVPR, 2023.
*B. Peng, Z. Tian, X. Wu, C. Wang, S. Liu, J. Su, and J. Jia.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Peng_Hierarchical_Dense_Correlation_Distillation_for_Few-Shot_Segmentation_CVPR_2023_paper.pdf)
[code](https://github.com/Pbihao/HDMNet)

1. **Rethinking the correlation in few-shot segmentation: A buoys view,** in CVPR, 2023.
*Y. Wang, R. Sun, and T. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Rethinking_the_Correlation_in_Few-Shot_Segmentation_A_Buoys_View_CVPR_2023_paper.pdf)

1. **CF-Font: Content fusion for few-shot font generation,** in CVPR, 2023.
*C. Wang, M. Zhou, T. Ge, Y. Jiang, H. Bao, and W. Xu.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_CF-Font_Content_Fusion_for_Few-Shot_Font_Generation_CVPR_2023_paper.pdf)
[code](https://github.com/wangchi95/CF-Font)

1. **MoLo: Motion-augmented long-short contrastive learning for few-shot action recognition,** in CVPR, 2023.
*X. Wang, S. Zhang, Z. Qing, C. Gao, Y. Zhang, D. Zhao, and N. Sang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MoLo_Motion-Augmented_Long-Short_Contrastive_Learning_for_Few-Shot_Action_Recognition_CVPR_2023_paper.pdf)
[code](https://github.com/alibaba-mmai-research/MoLo)

1. **Active exploration of multimodal complementarity for few-shot action recognition,** in CVPR, 2023.
*Y. Wanyan, X. Yang, C. Chen, and C. Xu.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wanyan_Active_Exploration_of_Multimodal_Complementarity_for_Few-Shot_Action_Recognition_CVPR_2023_paper.pdf)

1. **Generating features with increased crop-related diversity for few-shot object detection,** in CVPR, 2023.
*J. Xu, H. Le, and D. Samaras.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Generating_Features_With_Increased_Crop-Related_Diversity_for_Few-Shot_Object_Detection_CVPR_2023_paper.pdf)

1. **SMAE: Few-shot learning for HDR deghosting with saturation-aware masked autoencoders,** in CVPR, 2023.
*Q. Yan, S. Zhang, W. Chen, H. Tang, Y. Zhu, J. Sun, L. V. Gool, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_SMAE_Few-Shot_Learning_for_HDR_Deghosting_With_Saturation-Aware_Masked_Autoencoders_CVPR_2023_paper.pdf)

1. **MIANet: Aggregating unbiased instance and general information for few-shot semantic segmentation,** in CVPR, 2023.
*Y. Yang, Q. Chen, Y. Feng, and T. Huang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_MIANet_Aggregating_Unbiased_Instance_and_General_Information_for_Few-Shot_Semantic_CVPR_2023_paper.pdf)
[code](https://github.com/Aldrich2y/MIANet)

1. **FreeNeRF: Improving few-shot neural rendering with free frequency regularization,** in CVPR, 2023.
*J. Yang, M. Pavone, and Y. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_FreeNeRF_Improving_Few-Shot_Neural_Rendering_With_Free_Frequency_Regularization_CVPR_2023_paper.pdf)
[code](https://github.com/Jiawei-Yang/FreeNeRF)

1. **Exploring incompatible knowledge transfer in few-shot image generation,** in CVPR, 2023.
*Y. Zhao, C. Du, M. Abdollahzadeh, T. Pang, M. Lin, S. Yan, and N.-M. Cheung.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Exploring_Incompatible_Knowledge_Transfer_in_Few-Shot_Image_Generation_CVPR_2023_paper.pdf)
[code](https://github.com/yunqing-me/RICK)

1. **Where is my spot? few-shot image generation via latent subspace optimization,** in CVPR, 2023.
*C. Zheng, B. Liu, H. Zhang, X. Xu, and S. He.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_Where_Is_My_Spot_Few-Shot_Image_Generation_via_Latent_Subspace_CVPR_2023_paper.pdf)
[code](https://github.com/chansey0529/LSO)

1. **Distilling self-supervised vision transformers for weakly-supervised few-shot classification & segmentation,** in CVPR, 2023.
*D. Kang, P. Koniusz, M. Cho, and N. Murray.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Distilling_Self-Supervised_Vision_Transformers_for_Weakly-Supervised_Few-Shot_Classification__Segmentation_CVPR_2023_paper.pdf)

1. **FGNet: Towards filling the intra-class and inter-class gaps for few-shot segmentation,** in IJCAI, 2023.
*Y. Zhang, W. Yang, and S. Wang.*
[paper](https://www.ijcai.org/proceedings/2023/0194.pdf)
[code](https://github.com/YXZhang979/FGNet)

1. **Clustered-patch element connection for few-shot learning,** in IJCAI, 2023.
*J. Lai, S. Yang, J. Zhou, W. Wu, X. Chen, J. Liu, B.-B. Gao, and C. Wang.*
[paper](https://www.ijcai.org/proceedings/2023/0110.pdf)

1. **GeCoNeRF: Few-shot neural radiance fields via geometric consistency,** in ICML, 2023.
*M. Kwak, J. Song, and S. Kim.*
[paper](https://proceedings.mlr.press/v202/kwak23a/kwak23a.pdf)
[code](https://github.com/KU-CVLAB/GeCoNeRF)

2. **Few-shot common action localization via cross-attentional fusion of context and temporal dynamics,** in ICCV, 2023.
*J. Lee, M. Jain, and S. Yun.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Few-Shot_Common_Action_Localization_via_Cross-Attentional_Fusion_of_Context_and_ICCV_2023_paper.pdf)

6. **StyleDomain: Efficient and lightweight parameterizations of StyleGAN for one-shot and few-shot domain adaptation,** in ICCV, 2023.
*A. Alanov, V. Titov, M. Nakhodnov, and D. Vetrov.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Alanov_StyleDomain_Efficient_and_Lightweight_Parameterizations_of_StyleGAN_for_One-shot_and_ICCV_2023_paper.pdf)
[code](https://github.com/AIRI-Institute/StyleDomain)

7. **FlipNeRF: Flipped reflection rays for few-shot novel view synthesis,** in ICCV, 2023.
*S. Seo, Y. Chang, and N. Kwak.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Seo_FlipNeRF_Flipped_Reflection_Rays_for_Few-shot_Novel_View_Synthesis_ICCV_2023_paper.pdf)

12. **Few-shot physically-aware articulated mesh generation via hierarchical deformation,** in ICCV, 2023.
*X. Liu, B. Wang, H. Wang, and L. Yi.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Few-Shot_Physically-Aware_Articulated_Mesh_Generation_via_Hierarchical_Deformation_ICCV_2023_paper.pdf)
[code](meowuu7.github.io/few-arti-obj-gen)

17. **SparseNeRF: Distilling depth ranking for few-shot novel view synthesis,** in ICCV, 2023.
*G. Wang, Z. Chen, C. C. Loy, and Z. Liu.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SparseNeRF_Distilling_Depth_Ranking_for_Few-shot_Novel_View_Synthesis_ICCV_2023_paper.pdf)
[code](https://sparsenerf.github.io/)

18. **Few-shot video classification via representation fusion and promotion learning,** in ICCV, 2023.
*H. Xia, K. Li, M. R. Min, and Z. Ding.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Few-Shot_Video_Classification_via_Representation_Fusion_and_Promotion_Learning_ICCV_2023_paper.pdf)

25. **Augmenting and aligning snippets for few-shot video domain adaptation,** in ICCV, 2023.
*Y. Xu, J. Yang, Y. Zhou, Z. Chen, M. Wu, and X. Li.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Augmenting_and_Aligning_Snippets_for_Few-Shot_Video_Domain_Adaptation_ICCV_2023_paper.pdf)
[code](https://github.com/xuyu0010/SSA2lign)

30. **One-shot recognition of any material anywhere using contrastive learning with physics-based rendering,** in ICCV, 2023.
*M. S. Drehwald, S. Eppel, J. Li, H. Hao, and A. Aspuru-Guzik.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Drehwald_One-Shot_Recognition_of_Any_Material_Anywhere_Using_Contrastive_Learning_with_ICCV_2023_paper.pdf)
[code](https://github.com/ZuseZ4/MatSim-Dataset-Generator-Scripts-And-Neural-net)


31. **FS-DETR: Few-shot detection transformer with prompting and without re-training,** in ICCV, 2023.
*A. Bulat, R. Guerrero, B. Martinez, and G. Tzimiropoulos.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bulat_FS-DETR_Few-Shot_DEtection_TRansformer_with_Prompting_and_without_Re-Training_ICCV_2023_paper.pdf)

32. **Confidence-based visual dispersal for few-shot unsupervised domain adaptation,** in ICCV, 2023.
*Y. Xiong, H. Chen, Z. Lin, S. Zhao, and G. Ding.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiong_Confidence-based_Visual_Dispersal_for_Few-shot_Unsupervised_Domain_Adaptation_ICCV_2023_paper.pdf)
[code](https://github.com/Bostoncake/C-VisDiT)

33. **CDFSL-V: Cross-domain few-shot learning for videos,** in ICCV, 2023.
*S. Samarasinghe, M. N. Rizve, N. Kardan, and M. Shah.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Samarasinghe_CDFSL-V_Cross-Domain_Few-Shot_Learning_for_Videos_ICCV_2023_paper.pdf)
[code](https://github.com/Sarinda251/CDFSL-V)

34. **Generalized few-shot point cloud segmentation via geometric words,** in ICCV, 2023.
*Y. Xu, C. Hu, N. Zhao, and G. H. Lee.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Generalized_Few-Shot_Point_Cloud_Segmentation_via_Geometric_Words_ICCV_2023_paper.pdf)
[code](https://github.com/Pixie8888/GFS-3DSeg_GWs)

35. **Invariant training 2D-3D joint hard samples for few-shot point cloud recognition,** in ICCV, 2023.
*X. Yi, J. Deng, Q. Sun, X. Hua, J. Lim, and H. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yi_Invariant_Training_2D-3D_Joint_Hard_Samples_for_Few-Shot_Point_Cloud_ICCV_2023_paper.pdf)
[code](https://github.com/yxymessi/InvJoint)

36. **CIRI: Curricular inactivation for residue-aware one-shot video inpainting,** in ICCV, 2023.
*W. Zheng, C. Xu, X. Xu, W. Liu, and S. He.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_CIRI_Curricular_Inactivation_for_Residue-aware_One-shot_Video_Inpainting_ICCV_2023_paper.pdf)
[code](https://github.com/Arise-zwy/CIRI)

37. **s-Adaptive decoupled prototype for few-shot object detection,** in ICCV, 2023.
*J. Du, S. Zhang, Q. Chen, H. Le, Y. Sun, Y. Ni, J. Wang, B. He, and J. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Du_s-Adaptive_Decoupled_Prototype_for_Few-Shot_Object_Detection_ICCV_2023_paper.pdf)

39. **Parallel attention interaction network for few-shot skeleton-based action recognition,** in ICCV, 2023.
*X. Liu, S. Zhou, L. Wang, and G. Hua.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Parallel_Attention_Interaction_Network_for_Few-Shot_Skeleton-Based_Action_Recognition_ICCV_2023_paper.pdf)

41. **Robust one-shot face video re-enactment using hybrid latent spaces of StyleGAN2,** in ICCV, 2023.
*T. Oorloff, and Y. Yacoob.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Oorloff_Robust_One-Shot_Face_Video_Re-enactment_using_Hybrid_Latent_Spaces_of_ICCV_2023_paper.pdf)
[code](https://trevineoorloff.github.io/FaceVideoReenactment_HybridLatents.io/)

43. **Informative data mining for one-shot cross-domain semantic segmentation,** in ICCV, 2023.
*Y. Wang, J. Liang, J. Xiao, S. Mei, Y. Yang, and Z. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Informative_Data_Mining_for_One-Shot_Cross-Domain_Semantic_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/yxiwang/IDM)

44. **The euclidean space is evil: Hyperbolic attribute editing for few-shot image generation,** in ICCV, 2023.
*L. Li, Y. Zhang, and S. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_The_Euclidean_Space_is_Evil_Hyperbolic_Attribute_Editing_for_Few-shot_ICCV_2023_paper.pdf)
[code](https://github.com/lingxiao-li/HAE)

45. **Few shot font generation via transferring similarity guided global style and quantization local style,** in ICCV, 2023.
*W. Pan, A. Zhu, X. Zhou, B. K. Iwana, and S. Li.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Few_Shot_Font_Generation_Via_Transferring_Similarity_Guided_Global_Style_ICCV_2023_paper.pdf)
[code](https://github.com/awei669/VQ-Font)

46. **Boosting few-shot action recognition with graph-guided hybrid matching,** in ICCV, 2023.
*J. Xing, M. Wang, Y. Ruan, B. Chen, Y. Guo, B. Mu, G. Dai, J. Wang, and Y. Liu.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xing_Boosting_Few-shot_Action_Recognition_with_Graph-guided_Hybrid_Matching_ICCV_2023_paper.pdf)
[code](https://github.com/jiazheng-xing/GgHM)

47. **MSI: Maximize support-set information for few-shot segmentation,** in ICCV, 2023.
*S. Moon, S. S. Sohn, H. Zhou, S. Yoon, V. Pavlovic, M. H. Khan, and M. Kapadia.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Moon_MSI_Maximize_Support-Set_Information_for_Few-Shot_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/moonsh/MSI-Maximize-Support-Set-Information-ICCV2023)

49. **FastRecon: Few-shot industrial anomaly detection via fast feature reconstruction,** in ICCV, 2023.
*Z. Fang, X. Wang, H. Li, J. Liu, Q. Hu, and J. Xiao.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)
[code](https://github.com/FzJun26th/FastRecon)

50. **Self-calibrated cross attention network for few-shot segmentation,** in ICCV, 2023.
*Q. Xu, W. Zhao, G. Lin, and C. Long.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Self-Calibrated_Cross_Attention_Network_for_Few-Shot_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/Sam1224/SCCAN)

52. **Multi-grained temporal prototype learning for few-shot video object segmentation,** in ICCV, 2023.
*N. Liu, K. Nan, W. Zhao, Y. Liu, X. Yao, S. Khan, H. Cholakkal, R. M. Anwer, J. Han, and F. S. Khan.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Multi-grained_Temporal_Prototype_Learning_for_Few-shot_Video_Object_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/nankepan/VIPMT)

1. **HyperReenact: One-shot reenactment via jointly learning to refine and retarget faces,** in ICCV, 2023.
*S. Bounareli, C. Tzelepis, V. Argyriou, I. Patras, and G. Tzimiropoulos.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bounareli_HyperReenact_One-Shot_Reenactment_via_Jointly_Learning_to_Refine_and_Retarget_ICCV_2023_paper.pdf)
[code](https://github.com/StelaBou/HyperReenact)

11. **General image-to-image translation with one-shot image guidance,** in ICCV, 2023.
*B. Cheng, Z. Liu, Y. Peng, and Y. Lin.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_General_Image-to-Image_Translation_with_One-Shot_Image_Guidance_ICCV_2023_paper.pdf)
[code](https://github.com/CrystalNeuro/visual-concept-translator)

22. **ActorsNeRF: Animatable few-shot human rendering with generalizable NeRFs,** in ICCV, 2023.
*J. Mu, S. Sang, N. Vasconcelos, and X. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Mu_ActorsNeRF_Animatable_Few-shot_Human_Rendering_with_Generalizable_NeRFs_ICCV_2023_paper.pdf)
[code](https://jitengmu.github.io/ActorsNeRF/)

23. **One-shot implicit animatable avatars with model-based priors,** in ICCV, 2023.
*Y. Huang, H. Yi, W. Liu, H. Wang, B. Wu, W. Wang, B. Lin, D. Zhang, and D. Cai.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_One-shot_Implicit_Animatable_Avatars_with_Model-based_Priors_ICCV_2023_paper.pdf)
[code](https://huangyangyi.github.io/ELICIT)

24. **Preface: A data-driven volumetric prior for few-shot ultra high-resolution face synthesis,** in ICCV, 2023.
*M. C. Bühler, K. Sarkar, T. Shah, G. Li, D. Wang, L. Helminger, S. Orts-Escolano, D. Lagun, O. Hilliges, T. Beeler, and A. Meka.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Buhler_Preface_A_Data-driven_Volumetric_Prior_for_Few-shot_Ultra_High-resolution_Face_ICCV_2023_paper.pdf)

48. **DINAR: Diffusion inpainting of neural textures for one-shot human avatars,** in ICCV, 2023.
*D. Svitov, D. Gudkov, R. Bashirov, and V. Lempitsky.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Svitov_DINAR_Diffusion_Inpainting_of_Neural_Textures_for_One-Shot_Human_Avatars_ICCV_2023_paper.pdf)

20. **Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation,** in ICCV, 2023.
*J. Z. Wu, Y. Ge, X. Wang, S. W. Lei, Y. Gu, Y. Shi, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.pdf)
[code](https://tuneavideo.github.io)

10. **Phasic content fusing diffusion model with directional distribution consistency for few-shot model adaption,** in ICCV, 2023.
*T. Hu, J. Zhang, L. Liu, R. Yi, S. Kou, H. Zhu, X. Chen, Y. Wang, C. Wang, and L. Ma.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Phasic_Content_Fusing_Diffusion_Model_with_Directional_Distribution_Consistency_for_ICCV_2023_paper.pdf)
[code](https://github.com/sjtuplayer/few-shot-diffusion)

5. **Prototypical variational autoencoder for 3D few-shot object detection,** in NeurIPS, 2023.
*W. Tang, B. YANG, X. Li, Y. Liu, P. Heng, and C. Fu.*
[paper](https://openreview.net/attachment?id=fljrZsJ2I8&name=pdf)

22. **Generalizable one-shot 3D neural head avatar,** in NeurIPS, 2023.
*X. Li, S. D. Mello, S. Liu, K. Nagano, U. Iqbal, and J. Kautz.*
[paper](https://openreview.net/attachment?id=95q46MpBGZ&name=pdf)
[code](https://research.nvidia.com/labs/lpr/one-shot-avatar/)

24. **Focus on query: Adversarial mining transformer for few-shot segmentation,** in NeurIPS, 2023.
*Y. Wang, N. Luo, and T. Zhang.*
[paper](https://openreview.net/attachment?id=hxJu0386if&name=pdf)
[code](https://github.com/Wyxdm/AMNet)

### Robotics

1. **Towards one shot learning by imitation for humanoid robots,** in ICRA, 2010.
*Y. Wu and Y. Demiris.*
[paper](https://spiral.imperial.ac.uk/bitstream/10044/1/12669/4/icra2010.pdf)

1. **Learning manipulation actions from a few demonstrations,** in ICRA, 2013.
*N. Abdo, H. Kretzschmar, L. Spinello, and C. Stachniss.*
[paper](https://ieeexplore.ieee.org/document/6630734)

1. **Learning assistive strategies from a few user-robot interactions: Model-based reinforcement learning approach,** in ICRA, 2016. 
*M. Hamaya, T. Matsubara, T. Noda, T. Teramae, and J. Morimoto.*
[paper](https://ieeexplore.ieee.org/document/7487509)

1. **One-shot imitation learning,** in NeurIPS, 2017.
*Y. Duan, M. Andrychowicz, B. Stadie, J. Ho, J. Schneider, I. Sutskever, P. Abbeel, and W. Zaremba.*
[paper](https://papers.nips.cc/paper/6709-one-shot-imitation-learning.pdf)

1. **Meta-learning language-guided policy learning,** in ICLR, 2019.
*J. D. Co-Reyes, A. Gupta, S. Sanjeev, N. Altieri, J. DeNero, P. Abbeel, and S. Levine.*
[paper](https://openreview.net/forum?id=HkgSEnA5KQ)

1. **Meta reinforcement learning with autonomous inference of subtask dependencies,** in ICLR, 2020.
*S. Sohn, H. Woo, J. Choi, and H. Lee.*
[paper](https://openreview.net/pdf?id=HkgsWxrtPB)

1. **Watch, try, learn: Meta-learning from demonstrations and rewards,** in ICLR, 2020.
*A. Zhou, E. Jang, D. Kappler, A. Herzog, M. Khansari, P. Wohlhart, Y. Bai, M. Kalakrishnan, S. Levine, and C. Finn.*
[paper](https://openreview.net/pdf?id=SJg5J6NtDr)

1. **Few-shot Bayesian imitation learning with logical program policies,** in AAAI, 2020.
*T. Silver, K. R. Allen, A. K. Lew, L. P. Kaelbling, and J. Tenenbaum.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6587)

1. **One solution is not all you need: Few-shot extrapolation via structured MaxEnt RL,** in NeurIPS, 2020.
*S. Kumar, A. Kumar, S. Levine, and C. Finn.*
[paper](https://proceedings.neurips.cc/paper/2020/file/5d151d1059a6281335a10732fc49620e-Paper.pdf)

1. **Bowtie networks: Generative modeling for joint few-shot recognition and novel-view synthesis,** in ICLR, 2021.
*Z. Bao, Y. Wang, and M. Hebert.*
[paper](https://openreview.net/pdf?id=ESG-DMKQKsD)

1. **Demonstration-conditioned reinforcement learning for few-shot imitation,** in ICML, 2021.
*C. R. Dance, J. Perez, and T. Cachet.*
[paper](http://proceedings.mlr.press/v139/dance21a/dance21a.pdf)

1. **Hierarchical few-shot imitation with skill transition models,** in ICLR, 2022.
*K. Hakhamaneshi, R. Zhao, A. Zhan, P. Abbeel, and M. Laskin.*
[paper](https://openreview.net/pdf?id=xKZ4K0lTj_)

1. **Prompting decision transformer for few-shot policy generalization,** in ICML, 2022.
*M. Xu, Y. Shen, S. Zhang, Y. Lu, D. Zhao, J. B. Tenenbaum, and C. Gan.*
[paper](https://proceedings.mlr.press/v162/xu22g/xu22g.pdf)
[code](https://github.com/mxu34/prompt-dt)

1. **Stage conscious attention network (SCAN): A demonstration-conditioned policy for few-shot imitation,** in AAAI, 2022.
*J.-F. Yeh, C.-M. Chung, H.-T. Su, Y.-T. Chen, and W. H. Hsu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20868/20627)

1. **Online prototype alignment for few-shot policy transfer,** in ICML, 2023.
*Q. Yi, R. Zhang, S. Peng, J. Guo, Y. Gao, K. Yuan, R. Chen, S. Lan, X. Hu, Z. Du, X. Zhang, Q. Guo, and Y. Chen.*
[paper](https://proceedings.mlr.press/v202/yi23b/yi23b.pdf)
[code](https://github.com/albertcity/OP)

54. **LLM-planner: Few-shot grounded planning for embodied agents with large language models,** in ICCV, 2023.
*C. H. Song, J. Wu, C. Washington, B. M. Sadler, W. Chao, and Y. Su.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Song_LLM-Planner_Few-Shot_Grounded_Planning_for_Embodied_Agents_with_Large_Language_ICCV_2023_paper.pdf)
[code](https://osu-nlp-group.github.io/LLM-Planner/)

16. **Where2Explore: Few-shot affordance learning for unseen novel categories of articulated objects,** in NeurIPS, 2023.
*C. Ning, R. Wu, H. Lu, K. Mo, and H. Dong.*
[paper](https://openreview.net/attachment?id=QLllDwizVd&name=pdf)

### Natural Language Processing

1. **High-risk learning: Acquiring new word vectors from tiny data,** in EMNLP, 2017.
*A. Herbelot and M. Baroni.*
[paper](https://www.aclweb.org/anthology/D17-1030.pdf)

1. **Few-shot representation learning for out-of-vocabulary words,** in ACL, 2019.
*Z. Hu, T. Chen, K.-W. Chang, and Y. Sun.*
[paper](https://www.aclweb.org/anthology/P19-1402.pdf)

1. **Learning to customize model structures for few-shot dialogue generation tasks,** in ACL, 2020.
*Y. Song, Z. Liu, W. Bi, R. Yan, and M. Zhang.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.517.pdf)

1. **Few-shot slot tagging with collapsed dependency transfer and label-enhanced task-adaptive projection network,** in ACL, 2020.
*Y. Hou, W. Che, Y. Lai, Z. Zhou, Y. Liu, H. Liu, and T. Liu.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.128.pdf)

1. **Meta-reinforced multi-domain state generator for dialogue systems,** in ACL, 2020.
*Y. Huang, J. Feng, M. Hu, X. Wu, X. Du, and S. Ma.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.636.pdf)

1. **Universal natural language processing with limited annotations: Try few-shot textual entailment as a start,** in EMNLP, 2020.
*W. Yin, N. F. Rajani, D. Radev, R. Socher, and C. Xiong.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.660.pdf)
[code](https://github.com/salesforce/UniversalFewShotNLP)

1. **Simple and effective few-shot named entity recognition with structured nearest neighbor learning,** in EMNLP, 2020.
*Y. Yang, and A. Katiyar.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.516.pdf)
[code](https://github.com/asappresearch/structshot)

1. **Discriminative nearest neighbor few-shot intent detection by transferring natural language inference,** in EMNLP, 2020.
*J. Zhang, K. Hashimoto, W. Liu, C. Wu, Y. Wan, P. Yu, R. Socher, and C. Xiong.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.411.pdf)
[code](https://github.com/salesforce/DNNC-few-shot-intent)

1. **Few-shot learning for opinion summarization,** in EMNLP, 2020.
*A. Bražinskas, M. Lapata, and I. Titov.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.337.pdf)
[code](https://github.com/abrazinskas/FewSum)

1. **Few-shot complex knowledge base question answering via meta reinforcement learning,** in EMNLP, 2020.
*Y. Hua, Y. Li, G. Haffari, G. Qi, and T. Wu.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.469.pdf)
[code](https://github.com/DevinJake/MRL-CQA)

1. **Self-supervised meta-learning for few-shot natural language classification tasks,** in EMNLP, 2020.
*T. Bansal, R. Jha, T. Munkhdalai, and A. McCallum.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.38.pdf)
[code](https://github.com/iesl/metanlp)

1. **Uncertainty-aware self-training for few-shot text classification,** in NeurIPS, 2020.
*S. Mukherjee, and A. Awadallah.*
[paper](https://proceedings.neurips.cc/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf)
[code](https://github.com/microsoft/UST)

1. **Learning to extrapolate knowledge: Transductive few-shot out-of-graph link prediction,** in NeurIPS, 2020:.
*J. Baek, D. B. Lee, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/0663a4ddceacb40b095eda264a85f15c-Paper.pdf)
[code](https://github.com/JinheonBaek/GEN)

1. **MetaNER: Named entity recognition with meta-learning,** in WWW, 2020.
*J. Li, S. Shang, and L. Shao.*
[paper](https://dl.acm.org/doi/10.1145/3366423.3380127)

1. **Conditionally adaptive multi-task learning: Improving transfer learning in NLP using fewer parameters & less data,** in ICLR, 2021.
*J. Pilault, A. E. hattami, and C. Pal.*
[paper](https://openreview.net/pdf?id=de11dbHzAMF)
[code](https://github.com/CAMTL/CA-MTL)

1. **Revisiting few-sample BERT fine-tuning,** in ICLR, 2021.
*T. Zhang, F. Wu, A. Katiyar, K. Q. Weinberger, and Y. Artzi.*
[paper](https://openreview.net/pdf?id=cO1IH43yUF)
[code](https://pytorch.org/docs/1.4.0/_modules/torch/optim/adamw.html)

1. **Few-shot conversational dense retrieval,** in SIGIR, 2021.
*S. Yu, Z. Liu, C. Xiong, T. Feng, and Z. Liu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462856)
[code](https://github.com/thunlp/ConvDR)

1. **Few-shot language coordination by modeling theory of mind,** in ICML, 2021.
*H. Zhu, G. Neubig, and Y. Bisk.*
[paper](http://proceedings.mlr.press/v139/zhu21d/zhu21d.pdf)
[code](https://github.com/CLAW-Lab/ToM)

1. **KEML: A knowledge-enriched meta-learning framework for lexical relation classification,** in AAAI, 2021.
*C. Wang, M. Qiu, J. Huang, and X. He.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17640/17447)

1. **Few-shot learning for multi-label intent detection,** in AAAI, 2021.
*Y. Hou, Y. Lai, Y. Wu, W. Che, and T. Liu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17541/17348)
[code](https://github.com/AtmaHou/FewShotMultiLabel)

1. **SALNet: Semi-supervised few-shot text classification with attention-based lexicon construction,** in AAAI, 2021.
*J.-H. Lee, S.-K. Ko, and Y.-S. Han.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17558/17365)

1. **Learning from my friends: Few-shot personalized conversation systems via social networks,** in AAAI, 2021.
*Z. Tian, W. Bi, Z. Zhang, D. Lee, Y. Song, and N. L. Zhang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17638/17445)
[code](https://github.com/tianzhiliang/FewShotPersonaConvData)

1. **Relative and absolute location embedding for few-shot node classification on graph,** in AAAI, 2021.
*Z. Liu, Y. Fang, C. Liu, and S. C.H. Hoi.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16551/16358)

1. **Few-shot question answering by pretraining span selection,** in ACL-IJCNLP, 2021.
*O. Ram, Y. Kirstain, J. Berant, A. Globerson, and O. Levy.*
[paper](https://aclanthology.org/2021.acl-long.239.pdf)
[code](https://github.com/oriram/splinter)

1. **A closer look at few-shot crosslingual transfer: The choice of shots matters,** in ACL-IJCNLP, 2021.
*M. Zhao, Y. Zhu, E. Shareghi, I. Vulic, R. Reichart, A. Korhonen, and H. Schütze.*
[paper](https://aclanthology.org/2021.acl-long.447.pdf)
[code](https://github.com/fsxlt)

1. **Learning from miscellaneous other-classwords for few-shot named entity recognition,** in ACL-IJCNLP, 2021.
*M. Tong, S. Wang, B. Xu, Y. Cao, M. Liu, L. Hou, and J. Li.*
[paper](https://aclanthology.org/2021.acl-long.487.pdf)
[code](https://github.com/shuaiwa16/OtherClassNER.git)

1. **Distinct label representations for few-shot text classification,** in ACL-IJCNLP, 2021.
*S. Ohashi, J. Takayama, T. Kajiwara, and Y. Arase.*
[paper](https://aclanthology.org/2021.acl-short.105.pdf)
[code](https://github.com/21335732529sky/difference_extractor)

1. **Entity concept-enhanced few-shot relation extraction,** in ACL-IJCNLP, 2021.
*S. Yang, Y. Zhang, G. Niu, Q. Zhao, and S. Pu.*
[paper](https://aclanthology.org/2021.acl-short.124.pdf)
[code](https://github.com/LittleGuoKe/ConceptFERE)

1. **On training instance selection for few-shot neural text generation,** in ACL-IJCNLP, 2021.
*E. Chang, X. Shen, H.-S. Yeh, and V. Demberg.*
[paper](https://aclanthology.org/2021.acl-short.2.pdf)
[code](https://gitlab.com/erniecyc/few-selector)

1. **Unsupervised neural machine translation for low-resource domains via meta-learning,** in ACL-IJCNLP, 2021.
*C. Park, Y. Tae, T. Kim, S. Yang, M. A. Khan, L. Park, and J. Choo.*
[paper](https://aclanthology.org/2021.acl-long.225.pdf)
[code](https://github.com/papago-lab/MetaGUMT)

1. **Meta-learning with variational semantic memory for word sense disambiguation,** in ACL-IJCNLP, 2021.
*Y. Du, N. Holla, X. Zhen, C. Snoek, and E. Shutova.*
[paper](https://aclanthology.org/2021.acl-long.409.pdf)
[code](https://github.com/YDU-uva/VSM_WSD)

1. **Multi-label few-shot learning for aspect category detection,** in ACL-IJCNLP, 2021.
*M. Hu, S. Z. H. Guo, C. Xue, H. Gao, T. Gao, R. Cheng, and Z. Su.*
[paper](https://aclanthology.org/2021.acl-long.495.pdf)

1. **TextSETTR: Few-shot text style extraction and tunable targeted restyling,** in ACL-IJCNLP, 2021.
*P. Rileya, N. Constantb, M. Guob, G. Kumarc, D. Uthusb, and Z. Parekh.*
[paper](https://aclanthology.org/2021.acl-long.293.pdf)

1. **Few-shot text ranking with meta adapted synthetic weak supervision,** in ACL-IJCNLP, 2021.
*S. Sun, Y. Qian, Z. Liu, C. Xiong, K. Zhang, J. Bao, Z. Liu, and P. Bennett.*
[paper](https://aclanthology.org/2021.acl-long.390.pdf)
[code](https://github.com/thunlp/MetaAdaptRank)

1. **PROTAUGMENT: Intent detection meta-learning through unsupervised diverse paraphrasing,** in ACL-IJCNLP, 2021.
*T. Dopierre, C. Gravier, and W. Logerais.*
[paper](https://aclanthology.org/2021.acl-long.191.pdf)
[code](https://github.com/tdopierre/ProtAugment)

1. **AUGNLG: Few-shot natural language generation using self-trained data augmentation,** in ACL-IJCNLP, 2021.
*X. Xu, G. Wang, Y.-B. Kim, and S. Lee.*
[paper](https://aclanthology.org/2021.acl-long.95.pdf)
[code](https://github.com/XinnuoXu/AugNLG)

1. **Meta self-training for few-shot neural sequence labeling,** in KDD, 2021.
*Y. Wang, S. Mukherjee, H. Chu, Y. Tu, M. Wu, J. Gao, and A. H. Awadallah.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467235)
[code](https://github.com/microsoft/MetaST)

1. **Knowledge-enhanced domain adaptation in few-shot relation classification,** in KDD, 2021.
*J. Zhang, J. Zhu, Y. Yang, W. Shi, C. Zhang, and H. Wang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467438)
[code](https://github.com/imJiawen/KEFDA)

1. **Few-shot text classification with triplet networks, data augmentation, and curriculum learning,** in NAACL-HLT, 2021.
*J. Wei, C. Huang, S. Vosoughi, Y. Cheng, and S. Xu.*
[paper](https://aclanthology.org/2021.naacl-main.434.pdf)
[code](https://github.com/jasonwei20/triplet-loss)

1. **Few-shot intent classification and slot filling with retrieved examples,** in NAACL-HLT, 2021.
*D. Yu, L. He, Y. Zhang, X. Du, P. Pasupat, and Q. Li.*
[paper](https://aclanthology.org/2021.naacl-main.59.pdf)

1. **Non-parametric few-shot learning for word sense disambiguation,** in NAACL-HLT, 2021.
*H. Chen, M. Xia, and D. Chen.*
[paper](https://aclanthology.org/2021.naacl-main.142.pdf)
[code](https://github.com/princeton-nlp/metric-wsd)

1. **Towards few-shot fact-checking via perplexity,** in NAACL-HLT, 2021.
*N. Lee, Y. Bang, A. Madotto, and P. Fung.*
[paper](https://aclanthology.org/2021.naacl-main.158.pdf)

1. **ConVEx: Data-efficient and few-shot slot labeling,** in NAACL-HLT, 2021.
*M. Henderson, and I. Vulic.*
[paper](https://aclanthology.org/2021.naacl-main.264.pdf)

1. **Few-shot text generation with natural language instructions,** in EMNLP, 2021.
*T. Schick, and H. Schütze.*
[paper](https://aclanthology.org/2021.emnlp-main.32.pdf)

1. **Towards realistic few-shot relation extraction,** in EMNLP, 2021.
*S. Brody, S. Wu, and A. Benton.*
[paper](https://aclanthology.org/2021.emnlp-main.433.pdf)
[code](https://github.com/bloomberg/emnlp21_fewrel)

1. **Few-shot emotion recognition in conversation with sequential prototypical networks,** in EMNLP, 2021.
*G. Guibon, M. Labeau, H. Flamein, L. Lefeuvre, and C. Clavel.*
[paper](https://aclanthology.org/2021.emnlp-main.549.pdf)
[code](https://github.com/gguibon/protoseq)

1. **Learning prototype representations across few-shot tasks for event detection,** in EMNLP, 2021.
*V. Lai, F. Dernoncourt, and T. H. Nguyen.*
[paper](https://aclanthology.org/2021.emnlp-main.427.pdf)

1. **Exploring task difficulty for few-shot relation extraction,** in EMNLP, 2021.
*J. Han, B. Cheng, and W. Lu.*
[paper](https://aclanthology.org/2021.emnlp-main.204.pdf)
[code](https://github.com/hanjiale/hcrp)

1. **Honey or poison? Solving the trigger curse in few-shot event detection via causal intervention,** in EMNLP, 2021.
*J. Chen, H. Lin, X. Han, and L. Sun.*
[paper](https://aclanthology.org/2021.emnlp-main.637.pdf)
[code](https://github.com/chen700564/causalfsed)

1. **Nearest neighbour few-shot learning for cross-lingual classification,** in EMNLP, 2021.
*M. S. Bari, B. Haider, and S. Mansour.*
[paper](https://aclanthology.org/2021.emnlp-main.131.pdf)

1. **Knowledge-aware meta-learning for low-resource text classification,** in EMNLP, 2021.
*H. Yao, Y. Wu, M. Al-Shedivat, and E. P. Xing.*
[paper](https://aclanthology.org/2021.emnlp-main.136.pdf)
[code](https://github.com/huaxiuyao/KGML)

1. **Few-shot named entity recognition: An empirical baseline study,** in EMNLP, 2021.
*J. Huang, C. Li, K. Subudhi, D. Jose, S. Balakrishnan, W. Chen, B. Peng, J. Gao, and J. Han.*
[paper](https://aclanthology.org/2021.emnlp-main.813.pdf)

1. **MetaTS: Meta teacher-student network for multilingual sequence labeling with minimal supervision,** in EMNLP, 2021.
*Z. Li, D. Zhang, T. Cao, Y. Wei, Y. Song, and B. Yin.*
[paper](https://aclanthology.org/2021.emnlp-main.255.pdf)

1. **Meta-LMTC: Meta-learning for large-scale multi-label text classification,** in EMNLP, 2021.
*R. Wang, X. Su, S. Long, X. Dai, S. Huang, and J. Chen.*
[paper](https://aclanthology.org/2021.emnlp-main.679.pdf)

1. **Ontology-enhanced prompt-tuning for few-shot learning,** in WWW, 2022.
*H. Ye, N. Zhang, S. Deng, X. Chen, H. Chen, F. Xiong, X. Chen, and H. Chen.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511921)

1. **EICO: Improving few-shot text classification via explicit and implicit consistency regularization,** in Findings of ACL, 2022.
*L. Zhao, and C. Yao.*
[paper](https://aclanthology.org/2022.findings-acl.283.pdf)

1. **Dialogue summaries as dialogue states (DS2), template-guided summarization for few-shot dialogue state tracking,** in Findings of ACL, 2022.
*J. Shin, H. Yu, H. Moon, A. Madotto, and J. Park.*
[paper](https://aclanthology.org/2022.findings-acl.302.pdf)
[code](https://github.com/jshin49/ds2)

1. **A few-shot semantic parser for wizard-of-oz dialogues with the precise thingtalk representation,** in Findings of ACL, 2022.
*G. Campagna, S. J. Semnani, R. Kearns, L. J. K. Sato, S. Xu, and M. S. Lam.*
[paper](https://aclanthology.org/2022.findings-acl.317.pdf)

1. **Multi-stage prompting for knowledgeable dialogue generation,** in Findings of ACL, 2022.
*Z. Liu, M. Patwary, R. Prenger, S. Prabhumoye, W. Ping, M. Shoeybi, and B. Catanzaro.*
[paper](https://aclanthology.org/2022.findings-acl.104.pdf)
[code](https://github.com/NVIDIA/Megatron-LM)

1. **Few-shot named entity recognition with self-describing networks,** in ACL, 2022.
*J. Chen, Q. Liu, H. Lin, X. Han, and L. Sun.*
[paper](https://aclanthology.org/2022.acl-long.392.pdf)
[code](https://github.com/chen700564/sdnet)

1. **CLIP models are few-shot learners: Empirical studies on VQA and visual entailment,** in ACL, 2022.
*H. Song, L. Dong, W. Zhang, T. Liu, and F. Wei.*
[paper](https://aclanthology.org/2022.acl-long.421.pdf)

1. **CONTaiNER: Few-shot named entity recognition via contrastive learning,** in ACL, 2022.
*S. S. S. Das, A. Katiyar, R. J. Passonneau, and R. Zhang.*
[paper](https://aclanthology.org/2022.acl-long.439.pdf)
[code](https://github.com/psunlpgroup/container)

1. **Few-shot controllable style transfer for low-resource multilingual settings,** in ACL, 2022.
*K. Krishna, D. Nathani, X. Garcia, B. Samanta, and P. Talukdar.*
[paper](https://aclanthology.org/2022.acl-long.514.pdf)

1. **Label semantic aware pre-training for few-shot text classification,** in ACL, 2022.
*A. Mueller, J. Krone, S. Romeo, S. Mansour, E. Mansimov, Y. Zhang, and D. Roth.*
[paper](https://aclanthology.org/2022.acl-long.570.pdf)

1. **Inverse is better! Fast and accurate prompt for few-shot slot tagging,** in Findings of ACL, 2022.
*Y. Hou, C. Chen, X. Luo, B. Li, and W. Che.*
[paper](https://aclanthology.org/2022.findings-acl.53.pdf)

1. **Label semantics for few shot named entity recognition,** in Findings of ACL, 2022.
*J. Ma, M. Ballesteros, S. Doss, R. Anubhai, S. Mallya, Y. Al-Onaizan, and D. Roth.*
[paper](https://aclanthology.org/2022.findings-acl.155.pdf)

1. **Hierarchical recurrent aggregative generation for few-shot NLG,** in Findings of ACL, 2022.
*G. Zhou, G. Lampouras, and I. Iacobacci.*
[paper](https://aclanthology.org/2022.findings-acl.170.pdf)

1. **Towards few-shot entity recognition in document images: A label-aware sequence-to-sequence framework,** in Findings of ACL, 2022.
*Z. Wang, and J. Shang.*
[paper](https://aclanthology.org/2022.findings-acl.329.pdf)

1. **A good prompt is worth millions of parameters: Low-resource prompt-based learning for vision-language models,** in ACL, 2022.
*W. Jin, Y. Cheng, Y. Shen, W. Chen, and X. Ren.*
[paper](https://aclanthology.org/2022.acl-long.197.pdf)
[code](https://github.com/woojeongjin/fewvlm)

1. **Generated knowledge prompting for commonsense reasoning,** in ACL, 2022.
*J. Liu, A. Liu, X. Lu, S. Welleck, P. West, R. L. Bras, Y. Choi, and H. Hajishirzi.*
[paper](https://aclanthology.org/2022.acl-long.225.pdf)
[code](https://github.com/liujch1998/gkp)

1. **End-to-end modeling via information tree for one-shot natural language spatial video grounding,** in ACL, 2022.
*M. Li, T. Wang, H. Zhang, S. Zhang, Z. Zhao, J. Miao, W. Zhang, W. Tan, J. Wang, P. Wang, S. Pu, and F. Wu.*
[paper](https://aclanthology.org/2022.acl-long.596.pdf)

1. **Leveraging task transferability to meta-learning for clinical section classification with limited data,** in ACL, 2022.
*Z. Chen, J. Kim, R. Bhakta, and M. Y. Sir.*
[paper](https://aclanthology.org/2022.acl-long.461.pdf)

1. **Improving meta-learning for low-resource text classification and generation via memory imitation,** in ACL, 2022.
*Y. Zhao, Z. Tian, H. Yao, Y. Zheng, D. Lee, Y. Song, J. Sun, and N. L. Zhang.*
[paper](https://aclanthology.org/2022.acl-long.44.pdf)

1. **A simple yet effective relation information guided approach for few-shot relation extraction,** in Findings of ACL, 2022.
*Y. Liu, J. Hu, X. Wan, and T. Chang.*
[paper](https://aclanthology.org/2022.findings-acl.62.pdf)
[code](https://github.com/lylylylylyly/simplefsre)

1. **Decomposed meta-learning for few-shot named entity recognition,** in Findings of ACL, 2022.
*T. Ma, H. Jiang, Q. Wu, T. Zhao, and C. Lin.*
[paper](https://aclanthology.org/2022.findings-acl.124.pdf)
[code](https://github.com/microsoft/vert-papers)

1. **Meta-learning for fast cross-lingual adaptation in dependency parsing,** in ACL, 2022.
*A. Langedijk, V. Dankers, P. Lippe, S. Bos, B. C. Guevara, H. Yannakoudakis, and E. Shutova.*
[paper](https://aclanthology.org/2022.acl-long.582.pdf)
[code](https://github.com/annaproxy/udify-metalearning)

1. **Enhancing cross-lingual natural language inference by prompt-learning from cross-lingual templates,** in ACL, 2022.
*K. Qi, H. Wan, J. Du, and H. Chen.*
[paper](https://aclanthology.org/2022.acl-long.134.pdf)
[code](https://github.com/qikunxun/pct)

1. **Few-shot stance detection via target-aware prompt distillation,** in SIGIR, 2022.
*Y. Jiang, J. Gao, H. Shen, and X. Cheng.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531979)
[code](https://github.com/jyjulyseven/TAPD)

1. **Relation-guided few-shot relational triple extraction,** in SIGIR, 2022.
*X. Cong, J. Sheng, S. Cui, B. Yu, T. Liu, and B. Wang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531831)

1. **Curriculum contrastive context denoising for few-shot conversational dense retrieval,** in SIGIR, 2022.
*K. Mao, Z. Dou, and H. Qian.*
[paper](https://dl.acm.org/doi/10.1145/3477495.3531961)
[code](https://github.com/kyriemao/COTED)

1. **Few-shot subgoal planning with language models,** in NAACL, 2022.
*L. Logeswaran, Y. Fu, M. Lee, and H. Lee.*
[paper](https://aclanthology.org/2022.naacl-main.402.pdf)

1. **Template-free prompt tuning for few-shot NER,** in NAACL, 2022.
*R. Ma, X. Zhou, T. Gui, Y. Tan, L. Li, Q. Zhang, and X. Huang.*
[paper](https://aclanthology.org/2022.naacl-main.420.pdf)
[code](https://github.com/rtmaww/EntLM/)

1. **Few-shot document-level relation extraction,** in NAACL, 2022.
*N. Popovic, and M. Färber.*
[paper](https://aclanthology.org/2022.naacl-main.421.pdf)
[code](https://github.com/nicpopovic/FREDo)

1. **An enhanced span-based decomposition method for few-shot sequence labeling,** in NAACL, 2022.
*P. Wang, R. Xu, T. Liu, Q. Zhou, Y. Cao, B. Chang, and Z. Sui.*
[paper](https://aclanthology.org/2022.naacl-main.369.pdf)
[code](https://github.com/Wangpeiyi9979/ESD)

1. **Automatic multi-label prompting: Simple and interpretable few-shot classification,** in NAACL, 2022.
*H. Wang, C. Xu, and J. McAuley.*
[paper](https://aclanthology.org/2022.naacl-main.401.pdf)
[code](https://github.com/HanNight/AMuLaP)

1. **On the effect of pretraining corpora on in-context few-shot learning by a large-scale language model,** in NAACL, 2022.
*S. Shin, S.-W. Lee, H. Ahn, S. Kim, H. Kim, B. Kim, K. Cho, G. Lee, W. Park, J.-W. Ha, and N. Sung.*
[paper](https://aclanthology.org/2022.naacl-main.380.pdf)

1. **MGIMN: Multi-grained interactive matching network for few-shot text classification,** in NAACL, 2022.
*J. Zhang, M. Maimaiti, G. Xing, Y. Zheng, and J. Zhang.*
[paper](https://aclanthology.org/2022.naacl-main.141.pdf)

1. **On the economics of multilingual few-shot learning: Modeling the cost-performance trade-offs of machine translated and manual data,** in NAACL, 2022.
*K. Ahuja, M. Choudhury, and S. Dandapat.*
[paper](https://aclanthology.org/2022.naacl-main.98.pdf)
[code](https://github.com/kabirahuja2431/PerformanceFunctionAnalysis)

1. **OmniTab: Pretraining with natural and synthetic data for few-shot table-based question answering,** in NAACL, 2022.
*Z. Jiang, Y. Mao, P. He, G. Neubig, and W. Chen.*
[paper](https://aclanthology.org/2022.naacl-main.68.pdf)
[code](https://github.com/jzbjyb/OmniTab)

1. **Fine-tuning pre-trained language models for few-shot intent detection: Supervised pre-training and isotropization,** in NAACL, 2022.
*H. Zhang, H. Liang, Y. Zhang, L.-M. Zhan, X.-M. Wu, X. Lu, and A. Y. Lam.*
[paper](https://aclanthology.org/2022.naacl-main.39.pdf)
[code](https://github.com/fanolabs/isoIntentBert-main)

1. **Embedding hallucination for few-shot language fine-tuning,** in NAACL, 2022.
*Y. Jian, C. Gao, and S. Vosoughi.*
[paper](https://aclanthology.org/2022.naacl-main.404.pdf)
[code](https://github.com/yiren-jian/EmbedHalluc)

1. **Few-shot semantic parsing with language models trained on code,** in NAACL, 2022.
*R. Shin, and B. V. Durme.*
[paper](https://aclanthology.org/2022.naacl-main.396.pdf)

1. **LEA: Meta knowledge-driven self-attentive document embedding for few-shot text classification,** in NAACL, 2022.
*S. Hong, and T. Y. Jang.*
[paper](https://aclanthology.org/2022.naacl-main.7.pdf)

1. **Contrastive learning for prompt-based few-shot language learners,** in NAACL, 2022.
*Y. Jian, C. Gao, and S. Vosoughi.*
[paper](https://aclanthology.org/2022.naacl-main.408.pdf)
[code](https://github.com/yiren-jian/LM-SupCon)

1. **Learn from relation information: Towards prototype representation rectification for few-shot relation extraction,** in NAACL, 2022.
*Y. Liu, J. Hu, X. Wan, and T.-H. Chang.*
[paper](https://aclanthology.org/2022.findings-naacl.139.pdf)
[code](https://github.com/lylylylylyly/PRM-FSRE)

1. **Efficient few-shot fine-tuning for opinion summarization,** in NAACL, 2022.
*A. Brazinskas, R. Nallapati, M. Bansal, and M. Dreyer.*
[paper](https://aclanthology.org/2022.findings-naacl.113.pdf)
[code](https://github.com/amazon-research/adasum)

1. **Improving few-shot image classification using machine- and user-generated natural language descriptions,** in NAACL, 2022.
*K. Nishida, K. Nishida, and S. Nishioka.*
[paper](https://aclanthology.org/2022.findings-naacl.106.pdf)

1. **RGL: A simple yet effective relation graph augmented prompt-based tuning approach for few-shot learning,** in NAACL, 2022.
*Y. Wang, X. Tian, H. Xiong, Y. Li, Z. Chen, S. Guo, and D. Dou.*
[paper](https://aclanthology.org/2022.findings-naacl.81.pdf)
[code](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot/RGL)

1. **“Diversity and uncertainty in moderation” are the key to data selection for multilingual few-shot transfer,** in NAACL, 2022.
*S. Kumar, S. Dandapat, and M. Choudhury.*
[paper](https://aclanthology.org/2022.findings-naacl.78.pdf)

1. **A generative language model for few-shot aspect-based sentiment analysis,** in NAACL, 2022.
*E. Hosseini-Asl, W. Liu, and C. Xiong.*
[paper](https://aclanthology.org/2022.findings-naacl.58.pdf)
[code](https://github.com/salesforce/fewshot_absa)

1. **Improving few-shot relation classiﬁcation by prototypical representation learning with deﬁnition text,** in NAACL, 2022.
*L. Zhenzhen, Y. Zhang, J.-Y. Nie, and D. Li.*
[paper](https://aclanthology.org/2022.findings-naacl.34.pdf)

1. **Few-shot self-rationalization with natural language prompts,** in NAACL, 2022.
*A. Marasovic, I. Beltagy, D. Downey, and M. E. Peters.*
[paper](https://aclanthology.org/2022.findings-naacl.31.pdf)
[code](https://github.com/allenai/feb)

1. **How to translate your samples and choose your shots? Analyzing translate-train & few-shot cross-lingual transfer,** in NAACL, 2022.
*I. Jundi, and G. Lapesa.*
[paper](https://aclanthology.org/2022.findings-naacl.11.pdf)
[code](https://github.com/imanjundi/cross-lingual-transfer)

1. **LMTurk: Few-shot learners as crowdsourcing workers in a language-model-as-a-service framework,** in NAACL, 2022.
*M. Zhao, F. Mi, Y. Wang, M. Li, X. Jiang, Q. Liu, and H. Schuetze.*
[paper](https://aclanthology.org/2022.findings-naacl.51.pdf)
[code](https://github.com/lmturk)

1. **LiST: Lite prompted self-training makes efficient few-shot learners,** in NAACL, 2022.
*Y. Wang, S. Mukherjee, X. Liu, J. Gao, A. H. Awadallah, and J. Gao.*
[paper](https://aclanthology.org/2022.findings-naacl.174.pdf)
[code](https://github.com/microsoft/LiST)

1. **Improving in-context few-shot learning via self-supervised training,** in NAACL, 2022.
*M. Chen, J. Du, R. Pasunuru, T. Mihaylov, S. Iyer, V. Stoyanov, and Z. Kozareva.*
[paper](https://aclanthology.org/2022.naacl-main.260.pdf)

1. **Por qué não utiliser alla språk? mixed training with gradient optimization in few-shot cross-lingual transfer,** in NAACL, 2022.
*H. Xu, and K. Murray.*
[paper](https://aclanthology.org/2022.findings-naacl.157.pdf)
[code](https://github.com/fe1ixxu/Mixed-Gradient-Few-Shot)

1. **On the effectiveness of sentence encoding for intent detection meta-learning,** in NAACL, 2022.
*T. Ma, Q. Wu, Z. Yu, T. Zhao, and C.-Y. Lin.*
[paper](https://aclanthology.org/2022.naacl-main.279.pdf)
[code](https://github.com/microsoft/KC/tree/main/papers/IDML)

1. **Few-shot fine-grained entity typing with automatic label interpretation and instance generation,** in KDD, 2022.
*J. Huang, Y. Meng, and J. Han.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539443)
[code](https://github.com/teapot123/Fine-Grained-Entity-Typing)

1. **Label-enhanced prototypical network with contrastive learning for multi-label few-shot aspect category detection,** in KDD, 2022.
*H. Liu, F. Zhang, X. Zhang, S. Zhao, J. Sun, H. Yu, and X. Zhang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539340)

1. **Task-adaptive few-shot node classification,** in KDD, 2022.
*S. Wang, K. Ding, C. Zhang, C. Chen, and J. Li.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539265)
[code](https://github.com/SongW-SW/TENT)

1. **Diversity features enhanced prototypical network for few-shot intent detection,** in IJCAI, 2022.
*F. Yang, X. Zhou, Y. Wang, A. Atawulla, and R. Bi.*
[paper](https://www.ijcai.org/proceedings/2022/0617.pdf)

1. **Function-words adaptively enhanced attention networks for few-shot inverse relation classification,** in IJCAI, 2022.
*C. Dou, S. Wu, X. Zhang, Z. Feng, and K. Wang.*
[paper](https://www.ijcai.org/proceedings/2022/0407.pdf)
[code](https://github.com/DOU123321/FAEA-FSRC)

1. **Curriculum-based self-training makes better few-shot learners for data-to-text generation,** in IJCAI, 2022.
*P. Ke, H. Ji, Z. Yang, Y. Huang, J. Feng, X. Zhu, and M. Huang.*
[paper](https://www.ijcai.org/proceedings/2022/0580.pdf)
[code](https://github.com/kepei1106/CBST)

1. **Graph-based model generation for few-shot relation extraction,** in EMNLP, 2022.
*W. Li, and T. Qian.*
[paper](https://aclanthology.org/2022.emnlp-main.5.pdf)
[code](https://github.com/NLPWM-WHU/GM_GEN)

1. **Prompt-based meta-learning for few-shot text classification,** in EMNLP, 2022.
*H. Zhang, X. Zhang, H. Huang, and L. Yu.*
[paper](https://aclanthology.org/2022.emnlp-main.87.pdf)
[code](https://github.com/MGHZHANG/PBML)

1. **Language models of code are few-shot commonsense learners,** in EMNLP, 2022.
*A. Madaan, S. Zhou, U. Alon, Y. Yang, and G. Neubig.*
[paper](https://aclanthology.org/2022.emnlp-main.90.pdf)
[code](https://github.com/reasoning-machines/CoCoGen)

1. **Large language models are few-shot clinical information extractors,** in EMNLP, 2022.
*M. Agrawal, S. Hegselmann, H. Lang, Y. Kim, and D. Sontag.*
[paper](https://aclanthology.org/2022.emnlp-main.130.pdf)

1. **ToKen: Task decomposition and knowledge infusion for few-shot hate speech detection,** in EMNLP, 2022.
*B. AlKhamissi, F. Ladhak, S. Iyer, V. Stoyanov, Z. Kozareva, X. Li, P. Fung, L. Mathias, A. Celikyilmaz, and M. Diab.*
[paper](https://aclanthology.org/2022.emnlp-main.136.pdf)

1. **Exploiting domain-slot related keywords description for few-shot cross-domain dialogue state tracking,** in EMNLP, 2022.
*Q. Gao, G. Dong, Y. Mou, L. Wang, C. Zeng, D. Guo, M. Sun, and W. Xu.*
[paper](https://aclanthology.org/2022.emnlp-main.157.pdf)

1. **KECP: Knowledge enhanced contrastive prompting for few-shot extractive question answering,** in EMNLP, 2022.
*J. Wang, C. Wang, M. Qiu, Q. Shi, H. Wang, J. huang, and M. Gao.*
[paper](https://aclanthology.org/2022.emnlp-main.206.pdf)
[code](https://github.com/alibaba/EasyNLP)

1. **SpanProto: A two-stage span-based prototypical network for few-shot named entity recognition,** in EMNLP, 2022.
*J. Wang, C. Wang, C. Tan, M. Qiu, S. Huang, J. huang, and M. Gao.*
[paper](https://aclanthology.org/2022.emnlp-main.227.pdf)
[code](https://github.com/alibaba/EasyNLP)

1. **Few-shot query-focused summarization with prefix-merging,** in EMNLP, 2022.
*R. Yuan, Z. Wang, Z. Cao, and W. Li.*
[paper](https://aclanthology.org/2022.emnlp-main.243.pdf)

1. **Incorporating relevance feedback for information-seeking retrieval using few-shot document re-ranking,** in EMNLP, 2022.
*T. Baumgartner, L. F. R. Ribeiro, N. Reimers, and I. Gurevych.*
[paper](https://aclanthology.org/2022.emnlp-main.614.pdf)
[code](https://github.com/UKPLab/incorporating-relevance)

1. **Few-shot learning with multilingual generative language models,** in EMNLP, 2022.
*X. V. Lin, T. Mihaylov, M. Artetxe, T. Wang, S. Chen, D. Simig, M. Ott, N. Goyal, S. Bhosale, J. Du, R. Pasunuru, S. Shleifer, P. S. Koura, V. Chaudhary, B. O'Horo, J. Wang, L. Zettlemoyer, Z. Kozareva, M. Diab, V. Stoyanov, and X. Li.*
[paper](https://aclanthology.org/2022.emnlp-main.616.pdf)
[code](https://github.com/facebookresearch/fairseq/tree/main/examples/xglm)

1. **Don't stop fine-tuning: On training regimes for few-shot cross-lingual transfer with multilingual language models,** in EMNLP, 2022.
*F. D. Schmidt, I. Vulic, and G. Glavas.*
[paper](https://aclanthology.org/2022.emnlp-main.736.pdf)
[code](https://github.com/fdschmidt93/fsxlt)

1. **Better few-shot relation extraction with label prompt dropout,** in EMNLP, 2022.
*P. Zhang, and W. Lu.*
[paper](https://aclanthology.org/2022.emnlp-main.471.pdf)
[code](https://github.com/jzhang38/LPD)

1. **A dual prompt learning framework for few-shot dialogue state tracking,** in WWW, 2023.
*Y. Yang, W. Lei, P. Huang, J. Cao, J. Li, and T.-S. Chua.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583238)
[code](https://github.com/YANG-Yuting/DPL)

1. **MetaTroll: Few-shot detection of state-sponsored trolls with transformer adapters,** in WWW, 2023.
*L. Tian, X. Zhang, and J. H. Lau.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583417)
[code](https://github.com/ltian678/metatroll-code.git)

1. **ContrastNet: A contrastive learning framework for few-shot text classification,** in AAAI, 2022.
*J. Chen, R. Zhang, Y. Mao, and J. Xu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21292)

1. **Few-shot cross-lingual stance detection with sentiment-based pre-training,** in AAAI, 2022.
*M. Hardalov, A. Arora, P. Nakov, and I. Augenstein.*
[paper](https://github.com/checkstep/senti-stance)

1. **ALP: Data augmentation using lexicalized PCFGs for few-shot text classification,** in AAAI, 2022.
*H. H. Kim, D. Woo, S. J. Oh, J.-W. Cha, and Y.-S. Han.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21336/21085)

1. **CINS: Comprehensive instruction for few-shot learning in task-oriented dialog systems,** in AAAI, 2022.
*F. Mi, Y. Wang, and Y. Li.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21356/21105)

1. **An empirical study of GPT-3 for few-shot knowledge-based VQA,** in AAAI, 2022.
*Z. Yang, Z. Gan, J. Wang, X. Hu, Y. Lu, Z. Liu, and L. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20215/19974)

1. **PROMPTAGATOR: Few-shot dense retrieval from 8 examples,** in ICLR, 2023.
*Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu, K. Hall, and M.-W. Chang.*
[paper](https://openreview.net/pdf?id=gmL46YMpu2J)

1. **QAID: Question answering inspired few-shot intent detection,** in ICLR, 2023.
*A. Yehudai, M. Vetzler, Y. Mass, K. Lazar, D. Cohen, and B. Carmeli.*
[paper](https://openreview.net/pdf?id=gNI4_85Cyve)

1. **CLUR: Uncertainty estimation for few-shot text classification with contrastive learning,** in KDD, 2023.
*J. He, X. Zhang, S. Lei, A. Alhamadani, F. Chen, B. Xiao, and C.-T. Lu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599276)
[code](https://github.com/he159ok/CLUR_UncertaintyEst_FewShot_TextCls.)

1. **Learning few-shot sample-set operations for noisy multi-label aspect category detection,** in IJCAI, 2023.
*S. Zhao, W. Chen, and T. Wang.*
[paper](https://www.ijcai.org/proceedings/2023/0589.pd)

1. **Few-shot document-level event argument extraction,** in ACL, 2023.
*X. Yang, Y. Lu, and L. R. Petzold.*
[paper](https://aclanthology.org/2023.acl-long.446.pdf)
[code](https://github.com/Xianjun-Yang/FewDocAE)

1. **FLamE: Few-shot learning from natural language explanations,** in ACL, 2023.
*Y. Zhou, Y. Zhang, and C. Tan.*
[paper](https://aclanthology.org/2023.acl-long.372.pdf)

1. **MetaAdapt: Domain adaptive few-shot misinformation detection via meta learning,** in ACL, 2023.
*Z. Yue, H. Zeng, Y. Zhang, L. Shang, and D. Wang.*
[paper](https://aclanthology.org/2023.acl-long.286.pdf)
[code](https://github.com/Yueeeeeeee/MetaAdapt)

1. **Code4Struct: Code generation for few-shot event structure prediction,** in ACL, 2023.
*X. Wang, S. Li, and H. Ji.*
[paper](https://aclanthology.org/2023.acl-long.202.pdf)
[code](https://github.com/xingyaoww/code4struct.)

1. **MANNER: A variational memory-augmented model for cross domain few-shot named entity recognition,** in ACL, 2023.
*J. Fang, X. Wang, Z. Meng, P. Xie, F. Huang, and Y. Jiang.*
[paper](https://aclanthology.org/2023.acl-long.234.pdf)
[code](https://github.com/Alibaba-NLP/MANNER)

1. **Dual class knowledge propagation network for multi-label few-shot intent detection,** in ACL, 2023.
*F. Zhang, W. Chen, F. Ding, and T. Wang.*
[paper](https://aclanthology.org/2023.acl-long.480.pdf)

1. **Few-shot event detection: An empirical study and a unified view,** in ACL, 2023.
*Y. Ma, Z. Wang, Y. Cao, and A. Sun.*
[paper](https://aclanthology.org/2023.acl-long.628.pdf)
[code](https://github.com/mayubo2333/fewshot_ED)

1. **CodeIE: Large code generation models are better few-shot information extractors,** in ACL, 2023.
*P. Li, T. Sun, Q. Tang, H. Yan, Y. Wu, X. Huang, and X. Qiu.*
[paper](https://aclanthology.org/2023.acl-long.855.pdf)
[code](https://github.com/artpli/CodeIE)

1. **Few-shot data-to-text generation via unified representation and multi-source learning,** in ACL, 2023.
*A. H. Li, M. Shang, E. Spiliopoulou, J. Ma, P. Ng, Z. Wang, B. Min, W. Y. Wang, K. R. McKeown, V. Castelli, D. Roth, and B. Xiang.*
[paper](https://aclanthology.org/2023.acl-long.894.pdf)

1. **Few-shot in-context learning on knowledge base question answering,** in ACL, 2023.
*T. Li, X. Ma, A. Zhuang, Y. Gu, Y. Su, and W. Chen.*
[paper](https://aclanthology.org/2023.acl-long.385.pdf)
[code](https://github.com/ltl3A87/KB-BINDER)

1. **Linguistic representations for fewer-shot relation extraction across domains,** in ACL, 2023.
*S. Gururaja, R. Dutt, T. Liao, and C. P. Rosé.*
[paper](https://aclanthology.org/2023.acl-long.414.pdf)
[code](https://github.com/ShoRit/flow_graphs)

1. **Few-shot reranking for multi-hop QA via language model prompting,** in ACL, 2023.
*M. Khalifa, L. Logeswaran, M. Lee, H. Lee, and L. Wang.*
[paper](https://aclanthology.org/2023.acl-long.885.pdf)
[code](https://github.com/mukhal/PromptRank)



### Knowledge Graph

1. **MetaEXP: Interactive explanation and exploration of large knowledge graphs,** in WWW, 2018.
*F. Behrens, S. Bischoff, P. Ladenburger, J. Rückin, L. Seidel, F. Stolp, M. Vaichenker, A. Ziegler, D. Mottin, F. Aghaei, E. Müller, M. Preusse, N. Müller, and M. Hunger.*
[paper](https://meta-exp.github.io/resources/paper.pdf)
[code](https://hpi.de/en/mueller/metaex)

1. **Meta relational learning for few-shot link prediction in knowledge graphs,** in EMNLP-IJCNLP, 2019.
*M. Chen, W. Zhang, W. Zhang, Q. Chen, and H. Chen.*
[paper](https://www.aclweb.org/anthology/D19-1431.pdf)

1. **Adapting meta knowledge graph information for multi-hop reasoning over few-shot relations,** in EMNLP-IJCNLP, 2019.
*X. Lv, Y. Gu, X. Han, L. Hou, J. Li, and Z. Liu.*
[paper](https://www.aclweb.org/anthology/D19-1334.pdf)

1. **Knowledge graph transfer network for few-shot recognition,** in AAAI, 2020.
*R. Chen, T. Chen, X. Hui, H. Wu, G. Li, and L. Lin.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6630)


1. **Few-shot knowledge graph completion,** in AAAI, 2020.
*C. Zhang, H. Yao, C. Huang, M. Jiang, Z. Li, and N. V. Chawla.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5698)

1. **Adaptive attentional network for few-shot knowledge graph completion,** in EMNLP, 2020.
*J. Sheng, S. Guo, Z. Chen, J. Yue, L. Wang, T. Liu, and H. Xu.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.131.pdf)
[code](https://github.com/JiaweiSheng/FAAN)

1. **Relational learning with gated and attentive neighbor aggregator for few-shot knowledge graph completion,** in SIGIR, 2021.
*G. Niu, Y. Li, C. Tang, R. Geng, J. Dai, Q. Liu, H. Wang, J. Sun, F. Huang, and L. Si.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462925)

1. **Learning inter-entity-interaction for few-shot knowledge graph completion,** in EMNLP, 2022.
*Y. Li, K. Yu, X. Huang, and Y. Zhang.*
[paper](https://aclanthology.org/2022.emnlp-main.524.pdf)
[code](https://github.com/cjlyl/FKGC-CIAN)

1. **Meta-learning based knowledge extrapolation for temporal knowledge graph,** in WWW, 2023.
*Z. Chen, C. Xu, F. Su, Z. Huang, and Y. Dou.*
[paper](hhttps://dl.acm.org/doi/abs/10.1145/3543507.3583279)

1. **Learning to sample and aggregate: Few-shot reasoning over temporal knowledge graphs,** in NeurIPS, 2022.
*R. Wang, z. li, D. Sun, S. Liu, J. Li, B. Yin, and T. Abdelzaher.*
[paper](https://openreview.net/pdf?id=1LmgISIDZJ)

1. **Few-shot relational reasoning via connection subgraph pretraining,** in NeurIPS, 2022.
*Q. Huang, H. Ren, and J. Leskovec.*
[paper](https://openreview.net/pdf?id=LvW71lgly25)
[code](https://github.com/snap-stanford/csr)

1. **Hierarchical relational learning for few-shot knowledge graph completion,** in ICLR, 2023.
*H. Wu, J. Yin, B. Rajaratnam, and J. Guo.*
[paper](https://openreview.net/pdf?id=zlwBI2gQL3K)
[code](https://github.com/alexhw15/HiRe)

1. **The unreasonable effectiveness of few-shot learning for machine translation,** in ICML, 2023.
*X. Garcia, Y. Bansal, C. Cherry, G. F. Foster, M. Krikun, M. Johnson, and O. Firat.*
[paper](https://proceedings.mlr.press/v202/garcia23a/garcia23a.pdf)


### Acoustic Signal Processing

1. **One-shot learning of generative speech concepts,** in CogSci, 2014. 
*B. Lake, C.-Y. Lee, J. Glass, and J. Tenenbaum.*
[paper](https://groups.csail.mit.edu/sls/publications/2014/lake-cogsci14.pdf)

1. **Machine speech chain with one-shot speaker adaptation,** INTERSPEECH, 2018.
*A. Tjandra, S. Sakti, and S. Nakamura.* 
[paper](https://ahcweb01.naist.jp/papers/conference/2018/201809_Interspeech_andros-tj/201809_Interspeech_andros-tj_1.paper.pdf)

1. **Investigation of using disentangled and interpretable representations for one-shot cross-lingual voice conversion,** INTERSPEECH, 2018.
*S. H. Mohammadi and T. Kim.*
[paper](https://isca-speech.org/archive/Interspeech_2018/pdfs/2525.pdf)

1. **Few-shot audio classification with attentional graph neural networks,** INTERSPEECH, 2019.
*S. Zhang, Y. Qin, K. Sun, and Y. Lin.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1532.pdf)

1. **One-shot voice conversion with disentangled representations by leveraging phonetic posteriorgrams,** INTERSPEECH, 2019.
*S. H. Mohammadi, and T. Kim.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1798.pdf)

1. **One-shot voice conversion with global speaker embeddings,** INTERSPEECH, 2019.
*H. Lu, Z. Wu, D. Dai, R. Li, S. Kang, J. Jia, and H. Meng.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2365.pdf)

1. **One-shot voice conversion by separating speaker and content representations with instance normalization,** INTERSPEECH, 2019.
*J.-C. Chou, and H.-Y. Lee.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2663.pdf)

1. **Audio2Head: Audio-driven one-shot talking-head generation with natural head motion,** in IJCAI, 2021.
*S. Wang, L. Li, Y. Ding, C. Fan, and X. Yu.*
[paper](https://www.ijcai.org/proceedings/2021/0152.pdf)

1. **Few-shot low-resource knowledge graph completion with multi-view task representation generation,** in KDD, 2023.
*S. Pei, Z. Kou, Q. Zhang, and X. Zhang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599350)
[code](https://github.com/scpei/FLow-MV)

1. **Normalizing flow-based neural process for few-shot knowledge graph completion,** in SIGIR, 2023.
*L. Luo, Y.-F. Li, G. Haffari, and S. Pan.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591743)
[code](https://github.com/RManLuo/NP-FKGC)


### Recommendation

1. **A meta-learning perspective on cold-start recommendations for items,** in NeurIPS, 2017.
*M. Vartak, A. Thiagarajan, C. Miranda, J. Bratman, and H. Larochelle.*
[paper](https://papers.nips.cc/paper/7266-a-meta-learning-perspective-on-cold-start-recommendations-for-items.pdf)

1. **MeLU: Meta-learned user preference estimator for cold-start recommendation,** in KDD, 2019.
*H. Lee, J. Im, S. Jang, H. Cho, and S. Chung.*
[paper](https://arxiv.org/pdf/1908.00413.pdf)
[code](https://github.com/hoyeoplee/MeLU)

1. **Sequential scenario-specific meta learner for online recommendation,** in KDD, 2019.
*Z. Du, X. Wang, H. Yang, J. Zhou, and J. Tang.*
[paper](https://arxiv.org/pdf/1906.00391.pdf)
[code](https://github.com/THUDM/ScenarioMeta)

1. **Few-shot learning for new user recommendation in location-based social networks,** in WWW, 2020.
*R. Li, X. Wu, X. Chen, and W. Wang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3366423.3379994)

1. **MAMO: Memory-augmented meta-optimization for cold-start recommendation,** in KDD, 2020.
*M. Dong, F. Yuan, L. Yao, X. Xu, and L. Zhu.*
[paper](https://arxiv.org/pdf/2007.03183.pdf)
[code](https://github.com/dongmanqing/Code-for-MAMO)

1. **Meta-learning on heterogeneous information networks for cold-start recommendation,** in KDD, 2020.
*Y. Lu, Y. Fang, and C. Shi.*
[paper](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6158&context=sis_research)
[code](https://github.com/rootlu/MetaHIN)

1. **MetaSelector: Meta-learning for recommendation with user-level adaptive model selection,** in WWW, 2020.
*M. Luo, F. Chen, P. Cheng, Z. Dong, X. He, J. Feng, and Z. Li.*
[paper](https://arxiv.org/pdf/2001.10378v1.pdf)

1. **Fast adaptation for cold-start collaborative filtering with meta-learning,** in ICDM, 2020.
*T. Wei, Z. Wu, R. Li, Z. Hu, F. Feng, X. H. Sun, and W. Wang.*
[paper](https://ieeexplore.ieee.org/document/9338389)

1. **Preference-adaptive meta-learning for cold-start recommendation,** in IJCAI, 2021.
*L. Wang, B. Jin, Z. Huang, H. Zhao, D. Lian, Q. Liu, and E. Chen.*
[paper](https://www.ijcai.org/proceedings/2021/0222.pdf)

1. **Meta-learning helps personalized product search,** in WWW, 2022.
*B. Wu, Z. Meng, Q. Zhang, and S. Liang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512036)

1. **Alleviating cold-start problem in CTR prediction with a variational embedding learning framework,** in WWW, 2022.
*X. Xu, C. Yang, Q. Yu, Z. Fang, J. Wang, C. Fan, Y. He, C. Peng, Z. Lin, and J. Shao.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512048)

1. **PNMTA: A pretrained network modulation and task adaptation approach for user cold-start recommendation,** in WWW, 2022.
*H. Pang, F. Giunchiglia, X. Li, R. Guan, and X. Feng.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511963)

1. **Few-shot news recommendation via cross-lingual transfer,** in WWW, 2023.
*T. Guo, L. Yu, B. Shihada, and X. Zhang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583383)
[code](https://github.com/taichengguo/Few-shot-NewsRec)

1. **ColdNAS: Search to modulate for user cold-start recommendation,** in WWW, 2023.
*S. Wu, Y. Wang, Q. Jing, D. Dong, D. Dou, and Q. Yao.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3543507.3583344)
[code](https://github.com/LARS-research/ColdNAS)

1. **Contrastive collaborative filtering for cold-start item recommendation,** in WWW, 2023.
*Z. Zhou, L. Zhang, and N. Yang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583383)
[code](https://github.com/zzhin/CCFCRec)

1. **Bootstrapping contrastive learning enhanced music cold-start matching,** in WWW, 2023.
*X. Zhao, Y. Zhang, Q. Xiao, Y. Ren, and Y. Yang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3543873.3584626)

1. **A dynamic meta-learning model for time-sensitive cold-start recommendations,** in AAAI, 2022.
*K. P. Neupane, E. Zheng, Y. Kong, and Q. Yu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20756/20515)

1. **SMINet: State-aware multi-aspect interests representation network for cold-start users recommendation,** in AAAI, 2022.
*W. Tao, Y. Li, L. Li, Z. Chen, H. Wen, P. Chen, T. Liang, and Q. Lu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20824)
[code](https://github.com/wanjietao/Fliggy-SMINet-AAAI2022)

1. **Multimodality helps unimodality: Cross-modal few-shot learning with multimodal models,** in CVPR, 2023.
*Z. Lin, S. Yu, Z. Kuang, D. Pathak, and D. Ramanan.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Multimodality_Helps_Unimodality_Cross-Modal_Few-Shot_Learning_With_Multimodal_Models_CVPR_2023_paper.pdf)
[code](https://github.com/linzhiqiu/cross_modal_adaptation)

1. **M2EU: Meta learning for cold-start recommendation via enhancing user preference estimation,** in SIGIR, 2023.
*Z. Wu, and X. Zhou.*
[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591719)
[code](https://github.com/zhenchaowu/M2EU)

1. **TAML: Time-aware meta learning for cold-start problem in news recommendation,** in SIGIR, 2023.
*J. Li, Y. Zhang, X. Lin, X. Yang, G. Zhou, L. Li, H. Chen, and J. Zhou.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3592068)

1. **Uncertainty-aware consistency learning for cold-start item recommendation,** in SIGIR, 2023.
*T. Liu, C. Gao, Z. Wang, D. Li, J. Hao, D. Jin, and Y. Li.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3592078)

1. **DCBT: A simple but effective way for unified warm and cold recommendation,** in SIGIR, 2023.
*J. Yang, L. Zhang, Y. He, K. Ding, Z. Huan, X. Zhang, and L. Mo.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591856)

1. **A preference learning decoupling framework for user cold-start recommendation,** in SIGIR, 2023.
*C. Wang, Y. Zhu, A. Sun, Z. Wang, and K. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591627)

1. **Aligning distillation for cold-start item recommendation,** in SIGIR, 2023.
*F. Huang, Z. Wang, X. Huang, Y. Qian, Z. Li, and H. Chen.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591732)
[code](https://github.com/zfnWong/ALDI)

### Others

1. **Low data drug discovery with one-shot learning,** ACS Central Science, 2017.
*H. Altae-Tran, B. Ramsundar, A. S. Pappu, and V. Pande.* 
[paper](https://arxiv.org/abs/1611.03199)

1. **SMASH: One-shot model architecture search through hypernetworks,** in ICLR, 2018.
*A. Brock, T. Lim, J. Ritchie, and N. Weston.*
[paper](https://openreview.net/forum?id=rydeCEhs-)

1. **SPARC: Self-paced network representation for few-shot rare category characterization,** in KDD, 2018.
*D. Zhou, J. He, H. Yang, and W. Fan.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3219968)

1. **MetaPred: Meta-learning for clinical risk prediction with limited patient electronic health records,** in KDD, 2019.
*X. S. Zhang, F. Tang, H. H. Dodge, J. Zhou, and F. Wang.*
[paper](https://arxiv.org/pdf/1905.03218.pdf)
[code](https://github.com/sheryl-ai/MetaPred)

1. **AffnityNet: Semi-supervised few-shot learning for disease type prediction,** in AAAI, 2019.
*T. Ma, and A. Zhang.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3898/3776)

1. **Data augmentation using learned transformations for one-shot medical image segmentation,** in CVPR, 2019.
*A. Zhao, G. Balakrishnan, F. Durand, J. V. Guttag, and A. V. Dalca.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Data_Augmentation_Using_Learned_Transformations_for_One-Shot_Medical_Image_Segmentation_CVPR_2019_paper.pdf)

1. **Learning from multiple cities: A meta-learning approach for spatial-temporal prediction,** in WWW, 2019.
*H. Yao, Y. Liu, Y. Wei, X. Tang, and Z. Li.*
[paper](https://arxiv.org/pdf/1901.08518.pdf)
[code](https://github.com/huaxiuyao/MetaST)

1. **Federated meta-learning for fraudulent credit card detection,** in IJCAI, 2020.
*W. Zheng, L. Yan, C. Gou, and F. Wang.*
[paper](https://www.ijcai.org/Proceedings/2020/0642.pdf)

1. **Differentially private meta-learning,** in ICLR, 2020.
*J. Li, M. Khodak, S. Caldas, and A. Talwalkar.*
[paper](https://openreview.net/pdf?id=rJgqMRVYvr)

1. **Towards fast adaptation of neural architectures with meta learning,** in ICLR, 2020.
*D. Lian, Y. Zheng, Y. Xu, Y. Lu, L. Lin, P. Zhao, J. Huang, and S. Gao.*
[paper](https://openreview.net/pdf?id=r1eowANFvr)

1. **LT-Net: Label transfer by learning reversible voxel-wise correspondence for one-shot medical image segmentation,** in CVPR, 2020.
*S. Wang, S. Cao, D. Wei, R. Wang, K. Ma, L. Wang, D. Meng, and Y. Zheng.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_LT-Net_Label_Transfer_by_Learning_Reversible_Voxel-Wise_Correspondence_for_One-Shot_CVPR_2020_paper.pdf)

1. **Few-shot pill recognition,** in CVPR, 2020.
*S. Ling, A. Pastor, J. Li, Z. Che, J. Wang, J. Kim, and P. L. Callet.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ling_Few-Shot_Pill_Recognition_CVPR_2020_paper.pdf)

1. **Self-supervision with superpixels: Training few-shot medical image segmentation without annotation,** in ECCV, 2020.
*C. Ouyang, C. Biffi, C. Chen, T. Kart, H. Qiu, and D. Rueckert.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740749.pdf)
[code](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)

1. **Deep complementary joint model for complex scene registration and few-shot segmentation on medical images,** in ECCV, 2020.
*Y. He, T. Li, G. Yang, Y. Kong, Y. Chen, H. Shu, J. Coatrieux, J. Dillenseger, and S. Li.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630749.pdf)

1. **Using optimal embeddings to learn new intents with few examples: An application in the insurance domain,** in KDD, 2020.
*S. Acharya, and G. Fung.*
[paper](http://ceur-ws.org/Vol-2666/KDD_Converse20_paper_10.pdf)

1. **Meta-learning for query conceptualization at web scale,** in KDD, 2020.
*F. X. Han, D. Niu, H. Chen, W. Guo, S. Yan, and B. Long.*
[paper](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/fhan-KDD20.pdf)

1. **Few-sample and adversarial representation learning for continual stream mining,** in WWW, 2020.
*Z. Wang, Y. Wang, Y. Lin, E. Delord, and L. Khan.*
[paper](https://dl.acm.org/doi/10.1145/3366423.3380153)


1. **Few-shot graph learning for molecular property prediction,** in WWW, 2021.
*Z. Guo, C. Zhang, W. Yu, J. Herr, O. Wiest, M. Jiang, and N. V. Chawla.*
[paper](https://doi.org/10.1145/3442381.3450112)
[code](https://github.com/zhichunguo/Meta-MGNN)

1. **Taxonomy-aware learning for few-shot event detection,** in WWW, 2021.
*J. Zheng, F. Cai, W. Chen, W. Lei, and H. Chen.*
[paper](https://doi.org/10.1145/3442381.344994)

1. **Learning from graph propagation via ordinal distillation for one-shot automated essay scoring,** in WWW, 2021.
*Z. Jiang, M. Liu, Y. Yin, H. Yu, Z. Cheng, and Q. Gu.*
[paper](https://doi.org/10.1145/3442381.3450017)

1. **Few-shot network anomaly detection via cross-network meta-learning,** in WWW, 2021.
*K. Ding, Q. Zhou, H. Tong, and H. Liu.*
[paper](https://doi.org/10.1145/3442381.3449922)

1. **Few-shot knowledge validation using rules,** in WWW, 2021.
*M. Loster, D. Mottin, P. Papotti, J. Ehmüller, B. Feldmann, and F. Naumann.*
[paper](https://doi.org/10.1145/3442381.3450040)

1. **Graph learning regularization and transfer learning for few-shot event detection,** in SIGIR, 2021.
*V. D. Lai, M. V. Nguyen, T. H. Nguyen, and F. Dernoncourt.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463054)
[code](https://github.com/laiviet/ed-fsl)

1. **Graph-evolving meta-learning for low-resource medical dialogue generation,** in AAAI, 2021.
*S. Lin, P. Zhou, X. Liang, J. Tang, R. Zhao, Z. Chen, and L. Lin.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17577/17384)

1. **Modeling the probabilistic distribution of unlabeled data for one-shot medical image segmentation,** in AAAI, 2021.
*Y. Ding, X. Yu, and Y. Yang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16212/16019)
[code](https://github.com/dyh127/Modeling-the-Probabilistic-Distribution-of-Unlabeled-Data)

1. **Progressive network grafting for few-shot knowledge distillation,** in AAAI, 2021.
*C. Shen, X. Wang, Y. Yin, J. Song, S. Luo, and M. Song.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16356/16163)
[code](https://github.com/zju-vipa/NetGraft)

1. **Curriculum meta-learning for next POI recommendation,** in KDD, 2021.
*Y. Chen, X. Wang, M. Fan, J. Huang, S. Yang, and W. Zhu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467132)
[code](https://github.com/PaddlePaddle/Research/tree/master/ST_DM/KDD2021-CHAML)

1. **MFNP: A meta-optimized model for few-shot next POI recommendation,** in IJCAI, 2021.
*H. Sun, J. Xu, K. Zheng, P. Zhao, P. Chao, and X. Zhou.*
[paper](https://www.ijcai.org/proceedings/2021/0415.pdf)

1. **Physics-aware spatiotemporal modules with auxiliary tasks for meta-learning,** in IJCAI, 2021.
*S. Seo, C. Meng, S. Rambhatla, and Y. Liu.*
[paper](https://www.ijcai.org/proceedings/2021/0405.pdf)

1. **Recurrent mask refinement for few-shot medical image segmentation,** in ICCV, 2021.
*H. Tang, X. Liu, S. Sun, X. Yan, and X. Xie.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_Recurrent_Mask_Refinement_for_Few-Shot_Medical_Image_Segmentation_ICCV_2021_paper.pdf)
[code](https://github.com/uci-cbcl/RP-Net)

1. **Property-aware relation networks for few-shot molecular property prediction,** in NeurIPS, 2021.
*Y. Wang, A. Abuduweili, Q. Yao, and D. Dou.*
[paper](https://proceedings.neurips.cc/paper/2021/file/91bc333f6967019ac47b49ca0f2fa757-Paper.pdf)
[code](https://github.com/tata1661/PAR-NeurIPS21)

1. **Few-shot data-driven algorithms for low rank approximation,** in NeurIPS, 2021.
*P. Indyk, T. Wagner, and D. Woodruff.*
[paper](https://proceedings.neurips.cc/paper/2021/file/588da7a73a2e919a23cb9a419c4c6d44-Paper.pdf)

1. **Non-Gaussian Gaussian processes for few-shot regression,** in NeurIPS, 2021.
*M. Sendera, J. Tabor, A. Nowak, A. Bedychaj, M. Patacchiola, T. Trzcinski, P. Spurek, and M. Zieba.*
[paper](https://proceedings.neurips.cc/paper/2021/file/54f3bc04830d762a3b56a789b6ff62df-Paper.pdf)

1. **HELP: Hardware-adaptive efficient latency prediction for NAS via meta-learning,** in NeurIPS, 2021.
*H. Lee, S. Lee, S. Chong, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2021/file/e3251075554389fe91d17a794861d47b-Paper.pdf)

1. **Learning to learn dense Gaussian processes for few-shot learning,** in NeurIPS, 2021.
*Z. Wang, Z. Miao, X. Zhen, and Q. Qiu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/6e2713a6efee97bacb63e52c54f0ada0-Paper.pdf)

1. **A meta-learning based stress category detection framework on social media,** in WWW, 2022.
*X. Wang, L. Cao, H. Zhang, L. Feng, Y. Ding, and N. Li.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512013)

1. **Which images to label for few-shot medical landmark detection?,** in CVPR, 2022.
*Q. Quan, Q. Yao, J. Li, and S. K. Zhou.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Quan_Which_Images_To_Label_for_Few-Shot_Medical_Landmark_Detection_CVPR_2022_paper.pdf)

1. **Recognizing medical search query intent by few-shot learning,** in SIGIR, 2022.
*Y. Wang, S. Wang, L. Yanyan, and D. Dou.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531789)
[code](https://github.com/tata1661/MEDIC-SIGIR22)

1. **MetaCare++: Meta-learning with hierarchical subtyping for cold-start diagnosis prediction in healthcare data,** in SIGIR, 2022.
*Y. Tan, C. Yang, X. Wei, C. Chen, W. Liu, L. Li, and J. Z. a. X. Zheng.*
[paper](https://dl.acm.org/doi/10.1145/3477495.3532020)

1. **Spatio-temporal graph few-shot learning with cross-city knowledge transfer,** in KDD, 2022.
*B. Lu, X. Gan, W. Zhang, H. Yao, L. Fu, and X. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539281)
[code](https://github.com/RobinLu1209/ST-GFSL)

1. **Few-shot learning for trajectory-based mobile game cheating detection,** in KDD, 2022.
*Y. Su, D. Yao, X. Chu, W. Li, J. Bi, S. Zhao, R. Wu, S. Zhang, J. Tao, and H. Deng.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539157)
[code](https://github.com/super1225/cheating-detection)

1. **Improving few-shot text-to-SQL with meta self-training via column specificity,** in IJCAI, 2022.
*X. Guo, Y. Chen, G. Qi, T. Wu, and H. Xu.*
[paper](https://www.ijcai.org/proceedings/2022/0576.pdf)
[code](https://github.com/ygxw0909/MST-SQL)

1. **Graph few-shot learning with task-specific structures,** in NeurIPS, 2022.
*S. Wang, C. Chen, and J. Li.*
[paper](https://openreview.net/pdf?id=3yO3MiSOkH4)
[code](https://github.com/SongW-SW/GLITTER)

1. **Meta-learning dynamics forecasting using task inference,** in NeurIPS, 2022.
*R. Wang, R. Walters, and R. Yu.*
[paper](https://openreview.net/pdf?id=BsSP7pZGFQO)
[code](https://github.com/Rose-STL-Lab/Dynamic-Adaptation-Network)

1. **Rapid model architecture adaption for meta-learning,** in NeurIPS, 2022.
*Y. Zhao, X. Gao, I. Shumailov, N. Fusi, and R. D. Mullins.*
[paper](https://openreview.net/pdf?id=Yq6g9xluV)

1. **Cross-domain few-shot graph classification,** in AAAI, 2022.
*and K. Hassani.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25615)

1. **Meta propagation networks for graph few-shot semi-supervised learning,** in AAAI, 2022.
*K. Ding, J. Wang, J. Caverlee, and H. Liu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20605/20364)
[code](https://github.com/kaize0409/Meta-PN)

1. **Pushing the limits of few-shot anomaly detection in industry vision: GraphCore,** in ICLR, 2023.
*G. Xie, J. Wang, J. Liu, F. Zheng, and Y. Jin.*
[paper](https://openreview.net/pdf?id=xzmqxHdZAwO)

1. **Context-enriched molecule representations improve few-shot drug discovery,** in ICLR, 2023.
*J. Schimunek, P. Seidl, L. Friedrich, D. Kuhn, F. Rippmann, S. Hochreiter, and G. Klambauer.*
[paper](https://openreview.net/pdf?id=XrMWUuEevr)
[code](https://github.com/ml-jku/MHNfs)

1. **Sequential latent variable models for few-shot high-dimensional time-series forecasting,** in ICLR, 2023.
*X. Jiang, R. Missel, Z. Li, and L. Wang.*
[paper](https://openreview.net/pdf?id=7C9aRX2nBf2)
[code](https://github.com/john-x-jiang/meta_ss)

1. **Transfer NAS with meta-learned Bayesian surrogates,** in ICLR, 2023.
*G. Shala, T. Elsken, F. Hutter, and J. Grabocka.*
[paper](https://openreview.net/pdf?id=paGvsrl4Ntr)
[code](https://github.com/TNAS-DCS/TNAS-DCS)

1. **Few-shot domain adaptation for end-to-end communication,** in ICLR, 2023.
*J. Raghuram, Y. Zeng, D. Garcia, R. Ruiz, S. Jha, J. Widmer, and S. Banerjee.*
[paper](https://openreview.net/pdf?id=4F1gvduDeL)
[code](https://github.com/jayaram-r/domain-adaptation-autoencoder)

1. **Rethinking few-shot medical segmentation: A vector quantization view,** in CVPR, 2023.
*S. Huang, T. Xu, N. Shen, F. Mu, and J. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Few-Shot_Medical_Segmentation_A_Vector_Quantization_View_CVPR_2023_paper.pdf)

1. **Virtual node tuning for few-shot node classification,** in KDD, 2023.
*Z. Tan, R. Guo, K. Ding, and H. Liu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599541)

1. **Contrastive meta-learning for few-shot node classification,** in KDD, 2023.
*S. Wang, Z. Tan, H. Liu, and J. Li.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599288)
[code](https://github.com/SongW-SW/COSMIC)

1. **Task-equivariant graph few-shot learning,** in KDD, 2023.
*S. Kim, J. Lee, N. Lee, W. Kim, S. Choi, and C. Park.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599515)
[code](https://github.com/sung-won-kim/TEG)

1. **Leveraging transferable knowledge concept graph embedding for cold-start cognitive diagnosis,** in SIGIR, 2023.
*W. Gao, H. Wang, Q. Liu, F. Wang, X. Lin, L. Yue, Z. Zhang, R. Lv, and S. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591774)
[code](https://github.com/bigdata-ustc/TechCD)

38. **Dual meta-learning with longitudinally consistent regularization for one-shot brain tissue segmentation across the human lifespan,** in ICCV, 2023.
*Y. Sun, F. Wang, J. Shu, H. Wang, L. Wang, D. Meng, and C. Lian.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Dual_Meta-Learning_with_Longitudinally_Consistent_Regularization_for_One-Shot_Brain_Tissue_ICCV_2023_paper.pdf)
[code](https://github.com/ladderlab-xjtu/DuMeta)

23. **The rise of AI language pathologists: Exploring two-level prompt learning for few-shot weakly-supervised whole slide image classification,** in NeurIPS, 2023.
*L. Qu, x. Luo, K. Fu, M. Wang, and Z. Song.*
[paper](https://openreview.net/attachment?id=mSDfBXr8Py&name=pdf)
[code](https://github.com/miccaiif/TOP)


## [Theories](#content)

1. **Learning to learn around a common mean,** in NeurIPS, 2018.
*G. Denevi, C. Ciliberto, D. Stamos, and M. Pontil.* 
[paper](https://papers.nips.cc/paper/8220-learning-to-learn-around-a-common-mean.pdf)

1. **Meta-learning and universality: Deep representations and gradient descent can approximate any learning algorithm,** in ICLR, 2018.
*C. Finn and S. Levine.*
[paper](https://openreview.net/forum?id=HyjC5yWCW)

1. **A theoretical analysis of the number of shots in few-shot learning,** in ICLR, 2020.
*T. Cao, M. T. Law, and S. Fidler.*
[paper](https://openreview.net/pdf?id=HkgB2TNYPS)

1. **Rapid learning or feature reuse? Towards understanding the effectiveness of MAML,** in ICLR, 2020.
*A. Raghu, M. Raghu, S. Bengio, and O. Vinyals.*
[paper](https://openreview.net/pdf?id=rkgMkCEtPB)

1. **Robust meta-learning for mixed linear regression with small batches,** in NeurIPS, 2020.
*W. Kong, R. Somani, S. Kakade, and S. Oh.*
[paper](https://proceedings.neurips.cc/paper/2020/file/3214a6d842cc69597f9edf26df552e43-Paper.pdf)

1. **One-shot distributed ridge regression in high dimensions,** in ICML, 2020.
*Y. Sheng, and E. Dobriban.*
[paper](http://proceedings.mlr.press/v119/sheng20a/sheng20a.pdf)

1. **Bridging the gap between practice and PAC-Bayes theory in few-shot meta-learning,** in NeurIPS, 2021.
*N. Ding, X. Chen, T. Levinboim, S. Goodman, and R. Soricut.*
[paper](https://proceedings.neurips.cc/paper/2021/file/f6b6d2a114a9644419dc8d2315f22401-Paper.pdf)

1. **Generalization bounds for meta-learning: An information-theoretic analysis,** in NeurIPS, 2021.
*Q. CHEN, C. Shui, and M. Marchand.*
[paper](https://proceedings.neurips.cc/paper/2021/file/d9d347f57ae11f34235b4555710547d8-Paper.pdf)

1. **Generalization bounds for meta-learning via PAC-Bayes and uniform stability,** in NeurIPS, 2021.
*A. Farid, and A. Majumdar.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf)

1. **Unraveling model-agnostic meta-learning via the adaptation learning rate,** in ICLR, 2022.
*Y. Zou, F. Liu, and Q. Li.*
[paper](https://openreview.net/pdf?id=3rULBvOJ8D2)

1. **On the importance of firth bias reduction in few-shot classification,** in ICLR, 2022.
*S. Ghaffari, E. Saleh, D. Forsyth, and Y. Wang.*
[paper](https://openreview.net/pdf?id=DNRADop4ksB)
[code](https://github.com/ehsansaleh/firth_bias_reduction)

1. **Global convergence of MAML and theory-inspired neural architecture search for few-shot learning,** in CVPR, 2022.
*H. Wang, Y. Wang, R. Sun, and B. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Global_Convergence_of_MAML_and_Theory-Inspired_Neural_Architecture_Search_for_CVPR_2022_paper.pdf)

1. **Smoothed embeddings for certified few-shot learning,** in NeurIPS, 2022.
*M. Pautov, O. Kuznetsova, N. Tursynbek, A. Petiushko, and I. Oseledets.*
[paper](https://openreview.net/pdf?id=m2JJO3iEe_5)
[code](https://github.com/koava36/certrob-fewshot)




## [Few-shot Learning and Zero-shot Learning](#content)

1. **Label-embedding for attribute-based classification,** in CVPR, 2013.
*Z. Akata, F. Perronnin, Z. Harchaoui, and C. Schmid.*
[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Akata_Label-Embedding_for_Attribute-Based_2013_CVPR_paper.pdf)

1. **A unified semantic embedding: Relating taxonomies and attributes,** in NeurIPS, 2014.
*S. J. Hwang and L. Sigal.*
[paper](https://papers.nips.cc/paper/5289-a-unified-semantic-embedding-relating-taxonomies-and-attributes.pdf)

1. **Multi-attention network for one shot learning,** in CVPR, 2017.
*P. Wang, L. Liu, C. Shen, Z. Huang, A. van den Hengel, and H. T. Shen.*
[paper](http://zpascal.net/cvpr2017/Wang_Multi-Attention_Network_for_CVPR_2017_paper.pdf)

1. **Few-shot and zero-shot multi-label learning for structured label spaces,** in EMNLP, 2018.
*A. Rios and R. Kavuluru.*
[paper](https://www.aclweb.org/anthology/D18-1352.pdf)

1. **Learning compositional representations for few-shot recognition,** in ICCV, 2019.
*P. Tokmakov, Y.-X. Wang, and M. Hebert.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tokmakov_Learning_Compositional_Representations_for_Few-Shot_Recognition_ICCV_2019_paper.pdf)
[code](https://sites.google.com/view/comprepr/home)

1. **Large-scale few-shot learning: Knowledge transfer with class hierarchy,** in CVPR, 2019.
*A. Li, T. Luo, Z. Lu, T. Xiang, and L. Wang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Large-Scale_Few-Shot_Learning_Knowledge_Transfer_With_Class_Hierarchy_CVPR_2019_paper.pdf)

1. **Generalized zero- and few-shot learning via aligned variational autoencoders,** in CVPR, 2019.
*E. Schonfeld, S. Ebrahimi, S. Sinha, T. Darrell, and Z. Akata.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schonfeld_Generalized_Zero-_and_Few-Shot_Learning_via_Aligned_Variational_Autoencoders_CVPR_2019_paper.pdf)
[code](https://github.com/edgarschnfld/CADA-VAE-PyTorch)

1. **F-VAEGAN-D2: A feature generating framework for any-shot learning,** in CVPR, 2019.
*Y. Xian, S. Sharma, B. Schiele, and Z. Akata.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xian_F-VAEGAN-D2_A_Feature_Generating_Framework_for_Any-Shot_Learning_CVPR_2019_paper.pdf)

1. **TGG: Transferable graph generation for zero-shot and few-shot learning,** in ACM MM, 2019.
*C. Zhang, X. Lyu, and Z. Tang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3343031.3351000)

1. **Adaptive cross-modal few-shot learning,** in NeurIPS, 2019.
*C. Xing, N. Rostamzadeh, B. N. Oreshkin, and P. O. Pinheiro.*
[paper](https://papers.nips.cc/paper/8731-adaptive-cross-modal-few-shot-learning.pdf)

1. **Learning meta model for zero- and few-shot face anti-spoofing,** in AAAI, 2020.
*Y. Qin, C. Zhao, X. Zhu, Z. Wang, Z. Yu, T. Fu, F. Zhou, J. Shi, and Z. Lei.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6866)

1. **RD-GAN: Few/Zero-shot chinese character style transfer via radical decomposition and rendering,** in ECCV, 2020.
*Y. Huang, M. He, L. Jin, and Y. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510154.pdf)

1. **An empirical study on large-scale multi-label text classification including few and zero-shot labels,** in EMNLP, 2020.
*I. Chalkidis, M. Fergadiotis, S. Kotitsas, P. Malakasiotis, N. Aletras, and I. Androutsopoulos.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.607.pdf)

1. **Multi-label few/zero-shot learning with knowledge aggregated from multiple label graphs,** in EMNLP, 2020.
*J. Lu, L. Du, M. Liu, and J. Dipnall.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.235.pdf)

1. **Emergent complexity and zero-shot transfer via unsupervised environment design,** in NeurIPS, 2020.
*M. Dennis, N. Jaques, E. Vinitsky, A. Bayen, S. Russell, A. Critch, and S. Levine.*
[paper](https://proceedings.neurips.cc/paper/2020/file/985e9a46e10005356bbaf194249f6856-Paper.pdf)

1. **Learning graphs for knowledge transfer with limited labels,** in CVPR, 2021.
*P. Ghosh, N. Saini, L. S. Davis, and A. Shrivastava.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghosh_Learning_Graphs_for_Knowledge_Transfer_With_Limited_Labels_CVPR_2021_paper.pdf)

1. **Improving zero and few-shot abstractive summarization with intermediate fine-tuning and data augmentation,** in NAACL-HLT, 2021.
*A. R. Fabbri, S. Han, H. Li, H. Li, M. Ghazvininejad, S. R. Joty, D. R. Radev, and Y. Mehdad.*
[paper](https://aclanthology.org/2021.naacl-main.57.pdf)

1. **SEQZERO: Few-shot compositional semantic parsing with sequential prompts and zero-shot models,** in NAACL, 2022.
*J. Yang, H. Jiang, Q. Yin, D. Zhang, B. Yin, and D. Yang.*
[paper](https://aclanthology.org/2022.findings-naacl.5.pdf)
[code](https://github.com/amzn/SeqZero)

1. **Label verbalization and entailment for effective zero and few-shot relation extraction,** in EMNLP, 2021.
*O. Sainz, O. L. d. Lacalle, G. Labaka, A. Barrena, and E. Agirre.*
[paper](https://aclanthology.org/2021.emnlp-main.92.pdf)
[code](https://github.com/osainz59/Ask2Transformers)

1. **An empirical investigation of word alignment supervision for zero-shot multilingual neural machine translation,** in EMNLP, 2021.
*A. Raganato, R. Vázquez, M. Creutz, and J. Tiedemann.*
[paper](https://aclanthology.org/2021.emnlp-main.664.pdf)

1. **Bridge to target domain by prototypical contrastive learning and label confusion: Re-explore zero-shot learning for slot filling,** in EMNLP, 2021.
*L. Wang, X. Li, J. Liu, K. He, Y. Yan, and W. Xu.*
[paper](https://aclanthology.org/2021.emnlp-main.746.pdf)
[code](https://github.com/w-lw/pclc)

1. **A label-aware BERT attention network for zero-shot multi-intent detection in spoken language understanding,** in EMNLP, 2021.
*T. Wu, R. Su, and B. Juang.*
[paper](https://aclanthology.org/2021.emnlp-main.399.pdf)

1. **Zero-shot dialogue disentanglement by self-supervised entangled response selection,** in EMNLP, 2021.
*T. Chi, and A. I. Rudnicky.*
[paper](https://aclanthology.org/2021.emnlp-main.400.pdf)
[code](https://github.com/chijames/zero_shot_dialogue_disentanglement)

1. **Robust retrieval augmented generation for zero-shot slot filling,** in EMNLP, 2021.
*M. R. Glass, G. Rossiello, M. F. M. Chowdhury, and A. Gliozzo.*
[paper](https://aclanthology.org/2021.emnlp-main.148.pdf)
[code](https://github.com/IBM/kgi-slot-filling)

1. **Everything is all it takes: A multipronged strategy for zero-shot cross-lingual information extraction,** in EMNLP, 2021.
*M. Yarmohammadi, S. Wu, M. Marone, H. Xu, S. Ebner, G. Qin, Y. Chen, J. Guo, C. Harman, K. Murray, A. S. White, M. Dredze, and B. V. Durme.*
[paper](https://aclanthology.org/2021.emnlp-main.149.pdf)
[code](https://github.com/shijie-wu/crosslingual-nlp)

1. **An empirical study on multiple information sources for zero-shot fine-grained entity typing,** in EMNLP, 2021.
*Y. Chen, H. Jiang, L. Liu, S. Shi, C. Fan, M. Yang, and R. Xu.*
[paper](https://aclanthology.org/2021.emnlp-main.210.pdf)

1. **Zero-shot dialogue state tracking via cross-task transfer,** in EMNLP, 2021.
*Z. Lin, B. Liu, A. Madotto, S. Moon, Z. Zhou, P. Crook, Z. Wang, Z. Yu, E. Cho, R. Subba, and P. Fung.*
[paper](https://aclanthology.org/2021.emnlp-main.622.pdf)
[code](https://github.com/facebookresearch/Zero-Shot-DST)

1. **Finetuned language models are zero-shot learners,** in ICLR, 2022.
*J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le.*
[paper](https://openreview.net/pdf?id=gEZrGCozdqR)
[code](https://github.com/google-research/flan)

1. **Zero-shot stance detection via contrastive learning,** in WWW, 2022.
*B. Liang, Z. Chen, L. Gui, Y. He, M. Yang, and R. Xu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511994)
[code](https://github.com/HITSZ-HLT/PT-HCL)

1. **Reframing instructional prompts to GPTk's language,** in Findings of ACL, 2022.
*D. Khashabi, C. Baral, Y. Choi, and H. Hajishirzi.*
[paper](https://aclanthology.org/2022.findings-acl.50.pdf)

1. **JointCL: A joint contrastive learning framework for zero-shot stance detection,** in ACL, 2022.
*B. Liang, Q. Zhu, X. Li, M. Yang, L. Gui, Y. He, and R. Xu.*
[paper](https://aclanthology.org/2022.acl-long.7.pdf)
[code](https://github.com/hitsz-hlt/jointcl)

1. **Knowledgeable prompt-tuning: Incorporating knowledge into prompt verbalizer for text classification,** in ACL, 2022.
*S. Hu, N. Ding, H. Wang, Z. Liu, J. Wang, J. Li, W. Wu, and M. Sun.*
[paper](https://aclanthology.org/2022.acl-long.158.pdf)
[code](https://github.com/thunlp/knowledgeableprompttuning)

1. **Uni-Perceiver: Pre-training unified architecture for generic perception for zero-shot and few-shot tasks,** in CVPR, 2022.
*X. Zhu, J. Zhu, H. Li, X. Wu, H. Li, X. Wang, and J. Dai.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.pdf)

1. **Enhancing zero-shot stance detection via targeted background knowledge,** in SIGIR, 2022.
*Q. Zhu, B. Liang, J. Sun, J. Du, and L. Z. a. X. Ruifeng.*
[paper](https://dl.acm.org/doi/10.1145/3477495.3531807)

1. **Textual entailment for event argument extraction: Zero- and few-shot with multi-source learning,** in NAACL, 2022.
*O. Sainz, I. Gonzalez-Dios, O. L. d. Lacalle, B. Min, and E. Agirre.*
[paper](https://aclanthology.org/2022.findings-naacl.187.pdf)
[code](https://github.com/osainz59/Ask2Transformers)

1. **Extreme zero-shot learning for extreme text classification,** in NAACL, 2022.
*Y. Xiong, W.-C. Chang, C.-J. Hsieh, H.-F. Yu, and I. S. Dhillon.*
[paper](https://aclanthology.org/2022.naacl-main.399.pdf)
[code](https://github.com/amzn/pecos/tree/mainline/examples/MACLR)

1. **Domain-oriented prefix-tuning: Towards efficient and generalizable fine-tuning for zero-shot dialogue summarization,** in NAACL, 2022.
*L. Zhao, F. Zheng, W. Zeng, K. He, W. Xu, H. Jiang, W. Wu, and Y. Wu.*
[paper](https://aclanthology.org/2022.naacl-main.357.pdf)
[code](https://github.com/Zeng-WH/DOP-Tuning)

1. **Nearest neighbor zero-shot inference,** in EMNLP, 2022.
*W. Shi, J. Michael, S. Gururangan, and L. Zettlemoyer.*
[paper](https://aclanthology.org/2022.emnlp-main.214.pdf)
[code](https://github.com/swj0419/kNN_prompt)

1. **Continued pretraining for better zero- and few-shot promptability,** in EMNLP, 2022.
*Z. Wu, R. L. L. IV, P. Walsh, A. Bhagia, D. Groeneveld, S. Singh, and I. Beltagy.*
[paper](https://aclanthology.org/2022.emnlp-main.300.pdf)
[code](https://github.com/allenai/better-promptability)

1. **InstructDial: Improving zero and few-shot generalization in dialogue through instruction tuning,** in EMNLP, 2022.
*P. Gupta, C. Jiao, Y.-T. Yeh, S. Mehri, M. Eskenazi, and J. P. Bigham.*
[paper](https://aclanthology.org/2022.emnlp-main.33.pdf)
[code](https://github.com/prakharguptaz/Instructdial)

1. **Prompt-and-Rerank: A method for zero-shot and few-shot arbitrary textual style transfer with small language models,** in EMNLP, 2022.
*M. Suzgun, L. Melas-Kyriazi, and D. Jurafsky.*
[paper](https://aclanthology.org/2022.emnlp-main.141.pdf)
[code](https://github.com/suzgunmirac/prompt-and-rerank)

1. **Learning instructions with unlabeled data for zero-shot cross-task generalization,** in EMNLP, 2022.
*Y. Gu, P. Ke, X. Zhu, and M. Huang.*
[paper](https://aclanthology.org/2022.emnlp-main.105.pdf)
[code](https://github.com/thu-coai/UDIT)

1. **Zero-shot cross-lingual transfer of prompt-based tuning with a unified multilingual prompt,** in EMNLP, 2022.
*L. Huang, S. Ma, D. Zhang, F. Wei, and H. Wang.*
[paper](https://aclanthology.org/2022.emnlp-main.790.pdf)
[code](https://github.com/mojave-pku/UniPrompt)

1. **Finetune like you pretrain: Improved finetuning of zero-shot vision models,** in CVPR, 2023.
*S. Goyal, A. Kumar, S. Garg, Z. Kolter, and A. Raghunathan.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Goyal_Finetune_Like_You_Pretrain_Improved_Finetuning_of_Zero-Shot_Vision_Models_CVPR_2023_paper.pdf)
[code](https://github.com/locuslab/FLYP)

1. **WinCLIP: Zero-/few-shot anomaly classification and segmentation,** in CVPR, 2023.
*J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)

1. **SemSup-XC: Semantic supervision for zero and few-shot extreme classification,** in ICML, 2023.
*P. Aggarwal, A. Deshpande, and K. R. Narasimhan.*
[paper](https://proceedings.mlr.press/v202/aggarwal23a/aggarwal23a.pdf)
[code](https://github.com/princeton-nlp/semsup-xc)

1. **Zero- and few-shot event detection via prompt-based meta learning,** in ACL, 2023.
*Z. Yue, H. Zeng, M. Lan, H. Ji, and D. Wang.*
[paper](https://aclanthology.org/2023.acl-long.440.pdf)
[code](https://github.com/Yueeeeeeee/MetaEvent)

1. **HINT: Hypernetwork instruction tuning for efficient zero- and few-shot generalisation,** in ACL, 2023.
*H. Ivison, A. Bhagia, Y. Wang, H. Hajishirzi, and M. E. Peters.*
[paper](https://aclanthology.org/2023.acl-long.631.pdf)
[code](https://github.com/allenai/hyper-task-descriptions)

1. **What does the failure to reason with "respectively" in zero/few-shot settings tell us about language models? acl 2023,** in ACL, 2023.
*R. Cui, S. Lee, D. Hershcovich, and A. Søgaard.*
[paper](https://aclanthology.org/2023.acl-long.489.pdf)
[code](https://github.com/ruixiangcui/WikiResNLI_NatResNLI)

## [Variants of Few-shot Learning](#content)

1. **Continuous adaptation via meta-learning in nonstationary and competitive environments,** in ICLR, 2018.
*M. Al-Shedivat, T. Bansal, Y. Burda, I. Sutskever, I. Mordatch, and P. Abbeel.*
[paper](https://openreview.net/forum?id=Sk2u1g-0-)

1. **Deep online learning via meta-learning: Continual adaptation for model-based RL,** in ICLR, 2018.
*A. Nagabandi, C. Finn, and S. Levine.*
[paper](https://openreview.net/references/pdf?id=ryuIpa6S4)

1. **Incremental few-shot learning with attention attractor networks,** in NeurIPS, 2019.
*M. Ren, R. Liao, E. Fetaya, and R. S. Zemel.*
[paper](https://papers.nips.cc/paper/8769-incremental-few-shot-learning-with-attention-attractor-networks.pdf)
[code](https://github.com/renmengye/inc-few-shot-attractor-public)

1. **Bidirectional one-shot unsupervised domain mapping,** in ICCV, 2019.
*T. Cohen, and L. Wolf.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cohen_Bidirectional_One-Shot_Unsupervised_Domain_Mapping_ICCV_2019_paper.pdf)

1. **XtarNet: Learning to extract task-adaptive representation for incremental few-shot learning,** in ICML, 2020.
*S. W. Yoon, D. Kim, J. Seo, and J. Moon.*
[paper](http://proceedings.mlr.press/v119/yoon20b/yoon20b.pdf)
[code](https://github.com/EdwinKim3069/XtarNet)

1. **Few-shot class-incremental learning,** in CVPR, 2020.
*X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tao_Few-Shot_Class-Incremental_Learning_CVPR_2020_paper.pdf)

1. **Wandering within a world: Online contextualized few-shot learning,** in ICLR, 2021.
*M. Ren, M. L. Iuzzolino, M. C. Mozer, and R. Zemel.*
[paper](https://openreview.net/pdf?id=oZIvHV04XgC)

1. **Repurposing pretrained models for robust out-of-domain few-shot learning,** in ICLR, 2021.
*N. Kwon, H. Na, G. Huang, and S. Lacoste-Julien.*
[paper](https://openreview.net/pdf?id=qkLMTphG5-h)
[code](https://anonymous.4open.science/r/08ef52cf-456a-4e36-a408-04e1ad0bc5a9/)

1. **Prototypical cross-domain self-supervised learning for few-shot unsupervised domain adaptation,** in CVPR, 2021.
*X. Yue, Z. Zheng, S. Zhang, Y. Gao, T. Darrell, K. Keutzer, and A. S. Vincentelli.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Prototypical_Cross-Domain_Self-Supervised_Learning_for_Few-Shot_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)

1. **Self-promoted prototype refinement for few-shot class-incremental learning,** in CVPR, 2021.
*K. Zhu, Y. Cao, W. Zhai, J. Cheng, and Z. Zha.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf)

1. **Semantic-aware knowledge distillation for few-shot class-incremental learning,** in CVPR, 2021.
*A. Cheraghian, S. Rahman, P. Fang, S. K. Roy, L. Petersson, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheraghian_Semantic-Aware_Knowledge_Distillation_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf)

1. **Few-shot incremental learning with continually evolved classifiers,** in CVPR, 2021.
*C. Zhang, N. Song, G. Lin, Y. Zheng, P. Pan, and Y. Xu.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)

1. **Learning a universal template for few-shot dataset generalization,** in ICML, 2021.
*E. Triantafillou, H. Larochelle, R. Zemel, and V. Dumoulin.*
[paper](http://proceedings.mlr.press/v139/triantafillou21a/triantafillou21a.pdf)

1. **GP-Tree: A gaussian process classifier for few-shot incremental learning,** in ICML, 2021.
*I. Achituve, A. Navon, Y. Yemini, G. Chechik, and E. Fetaya.*
[paper](http://proceedings.mlr.press/v139/achituve21a/achituve21a.pdf)
[code](https://github.com/IdanAchituve/GP-Tree)

1. **Addressing catastrophic forgetting in few-shot problems,** in ICML, 2021.
*P. Yap, H. Ritter, and D. Barber.*
[paper](http://proceedings.mlr.press/v139/yap21a/yap21a.pdf)
[code](https://github.com/pauchingyap/boml)

1. **Few-shot conformal prediction with auxiliary tasks,** in ICML, 2021.
*A. Fisch, T. Schuster, T. Jaakkola, and R. Barzilay.*
[paper](http://proceedings.mlr.press/v139/fisch21a/fisch21a.pdf)
[code](https://github.com/ajfisch/few-shot-cp)

1. **Few-shot lifelong learning,** in AAAI, 2021.
*P. Mazumder, P. Singh, and P. Rai.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16334/16141)

1. **Few-shot class-incremental learning via relation knowledge distillation,** in AAAI, 2021.
*S. Dong, X. Hong, X. Tao, X. Chang, X. Wei, and Y. Gong.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16213/16020)

1. **Few-shot one-class classification via meta-learning,** in AAAI, 2021.
*A. Frikha, D. Krompass, H. Koepken, and V. Tresp.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16913/16720)
[code](https://github.com/AhmedFrikha/Few-Shot-One-Class-Classification-via-Meta-Learning)

1. **Practical one-shot federated learning for cross-silo setting,** in IJCAI, 2021.
*Q. Li, B. He, and D. Song.*
[paper](https://www.ijcai.org/proceedings/2021/0205.pdf)
[code](https://github.com/QinbinLi/FedK)

1. **Incremental few-shot text classification with multi-round new classes: Formulation, dataset and system,** in NAACL-HLT, 2021.
*C. Xia, W. Yin, Y. Feng, and P. S. Yu.*
[paper](https://aclanthology.org/2021.naacl-main.106.pdf)

1. **Continual few-shot learning for text classification,** in EMNLP, 2021.
*R. Pasunuru, V. Stoyanov, and M. Bansal.*
[paper](https://aclanthology.org/2021.emnlp-main.460.pdf)
[code](https://github.com/ramakanth-pasunuru/cfl-benchmark)

1. **Self-training with few-shot rationalization,** in EMNLP, 2021.
*M. M. Bhat, A. Sordoni, and S. Mukherjee.*
[paper](https://aclanthology.org/2021.emnlp-main.836.pdf)

1. **Diverse distributions of self-supervised tasks for meta-learning in NLP,** in EMNLP, 2021.
*T. Bansal, K. P. Gunasekaran, T. Wang, T. Munkhdalai, and A. McCallum.*
[paper](https://aclanthology.org/2021.emnlp-main.469.pdf)

1. **Generalized and incremental few-shot learning by explicit learning and calibration without forgetting,** in ICCV, 2021.
*A. Kukleva, H. Kuehne, and B. Schiele.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kukleva_Generalized_and_Incremental_Few-Shot_Learning_by_Explicit_Learning_and_Calibration_ICCV_2021_paper.pdf)

1. **Meta learning on a sequence of imbalanced domains with difficulty awareness,** in ICCV, 2021.
*Z. Wang, T. Duan, L. Fang, Q. Suo, and M. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Meta_Learning_on_a_Sequence_of_Imbalanced_Domains_With_Difficulty_ICCV_2021_paper.pdf)
[code](https://github.com/joey-wang123/imbalancemeta)

1. **Synthesized feature based few-shot class-incremental learning on a mixture of subspaces,** in ICCV, 2021.
*A. Cheraghian, S. Rahman, S. Ramasinghe, P. Fang, C. Simon, L. Petersson, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cheraghian_Synthesized_Feature_Based_Few-Shot_Class-Incremental_Learning_on_a_Mixture_of_ICCV_2021_paper.pdf)

1. **Few-shot and continual learning with attentive independent mechanisms,** in ICCV, 2021.
*E. Lee, C. Huang, and C. Lee.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Few-Shot_and_Continual_Learning_With_Attentive_Independent_Mechanisms_ICCV_2021_paper.pdf)
[code](https://github.com/huang50213/AIM-Fewshot-Continual)

1. **Low-shot validation: Active importance sampling for estimating classifier performance on rare categories,** in ICCV, 2021.
*F. Poms, V. Sarukkai, R. T. Mullapudi, N. S. Sohoni, W. R. Mark, D. Ramanan, and K. Fatahalian.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Poms_Low-Shot_Validation_Active_Importance_Sampling_for_Estimating_Classifier_Performance_on_ICCV_2021_paper.pdf)

1. **Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima,** in NeurIPS, 2021.
*G. SHI, J. CHEN, W. Zhang, L. Zhan, and X. Wu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/357cfba15668cc2e1e73111e09d54383-Paper.pdf)

1. **Variational continual Bayesian meta-learning,** in NeurIPS, 2021.
*Q. Zhang, J. Fang, Z. Meng, S. Liang, and E. Yilmaz.*
[paper](https://proceedings.neurips.cc/paper/2021/file/cdd0500dc0ef6682fa6ec6d2e6b577c4-Paper.pdf)

1. **LFPT5: A unified framework for lifelong few-shot language learning based on prompt tuning of T5,** in ICLR, 2022.
*C. Qin, and S. Joty.*
[paper](https://openreview.net/pdf?id=HCRVf71PMF)
[code](https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning)

1. **Subspace regularizers for few-shot class incremental learning,** in ICLR, 2022.
*A. F. Akyürek, E. Akyürek, D. Wijaya, and J. Andreas.*
[paper](https://openreview.net/pdf?id=boJy41J-tnQ)
[code](https://github.com/feyzaakyurek/subspace-reg)

1. **Meta discovery: Learning to discover novel classes given very limited data,** in ICLR, 2022.
*H. Chi, F. Liu, W. Yang, L. Lan, T. Liu, B. Han, G. Niu, M. Zhou, and M. Sugiyama.*
[paper](https://openreview.net/pdf?id=MEpKGLsY8f)

1. **Topological transduction for hybrid few-shot learning,** in WWW, 2022.
*J. Chen, and A. Zhang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512033)

1. **Continual few-shot relation learning via embedding space regularization and data augmentation,** in ACL, 2022.
*C. Qin, and S. Joty.*
[paper](https://aclanthology.org/2022.acl-long.198.pdf)
[code](https://github.com/qcwthu/continual_fewshot_relation_learning)

1. **Few-shot class-incremental learning for named entity recognition,** in ACL, 2022.
*R. Wang, T. Yu, H. Zhao, S. Kim, S. Mitra, R. Zhang, and R. Henao.*
[paper](https://aclanthology.org/2022.acl-long.43.pdf)

1. **Task-adaptive negative envision for few-shot open-set recognition,** in CVPR, 2022.
*S. Huang, J. Ma, G. Han, and S. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task-Adaptive_Negative_Envision_for_Few-Shot_Open-Set_Recognition_CVPR_2022_paper.pdf)
[code](https://github.com/shiyuanh/TANE)

1. **Forward compatible few-shot class-incremental learning,** in CVPR, 2022.
*D. Zhou, F. Wang, H. Ye, L. Ma, S. Pu, and D. Zhan.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)
[code](https://github.com/zhoudw-zdw/CVPR22-Fact)

1. **Sylph: A hypernetwork framework for incremental few-shot object detection,** in CVPR, 2022.
*L. Yin, J. M. Perez-Rua, and K. J. Liang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_Sylph_A_Hypernetwork_Framework_for_Incremental_Few-Shot_Object_Detection_CVPR_2022_paper.pdf)

1. **Constrained few-shot class-incremental learning,** in CVPR, 2022.
*M. Hersche, G. Karunaratne, G. Cherubini, L. Benini, A. Sebastian, and A. Rahimi.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)

1. **iFS-RCNN: An incremental few-shot instance segmenter,** in CVPR, 2022.
*K. Nguyen, and S. Todorovic.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_iFS-RCNN_An_Incremental_Few-Shot_Instance_Segmenter_CVPR_2022_paper.pdf)

1. **MetaFSCIL: A meta-learning approach for few-shot class incremental learning,** in CVPR, 2022.
*Z. Chi, L. Gu, H. Liu, Y. Wang, Y. Yu, and J. Tang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chi_MetaFSCIL_A_Meta-Learning_Approach_for_Few-Shot_Class_Incremental_Learning_CVPR_2022_paper.pdf)

1. **Few-shot incremental learning for label-to-image translation,** in CVPR, 2022.
*P. Chen, Y. Zhang, Z. Li, and L. Sun.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Few-Shot_Incremental_Learning_for_Label-to-Image_Translation_CVPR_2022_paper.pdf)

1. **Revisiting learnable affines for batch norm in few-shot transfer learning,** in CVPR, 2022.
*M. Yazdanpanah, A. A. Rahman, M. Chaudhary, C. Desrosiers, M. Havaei, E. Belilovsky, and S. E. Kahou.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yazdanpanah_Revisiting_Learnable_Affines_for_Batch_Norm_in_Few-Shot_Transfer_Learning_CVPR_2022_paper.pdf)

1. **Few-shot learning with noisy labels,** in CVPR, 2022.
*K. J. Liang, S. B. Rangrej, V. Petrovic, and T. Hassner.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Few-Shot_Learning_With_Noisy_Labels_CVPR_2022_paper.pdf)

1. **Improving adversarially robust few-shot image classification with generalizable representations,** in CVPR, 2022.
*J. Dong, Y. Wang, J. Lai, and X. Xie.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Improving_Adversarially_Robust_Few-Shot_Image_Classification_With_Generalizable_Representations_CVPR_2022_paper.pdf)

1. **Geometer: Graph few-shot class-incremental learning via prototype representation,** in KDD, 2022.
*B. Lu, X. Gan, L. Yang, W. Zhang, L. Fu, and X. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539280)
[code](https://github.com/RobinLu1209/Geometer)

1. **Few-shot heterogeneous graph learning via cross-domain knowledge transfer,** in KDD, 2022.
*Q. Zhang, X. Wu, Q. Yang, C. Zhang, and X. Zhang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539431)

1. **Few-shot adaptation of pre-trained networks for domain shift,** in IJCAI, 2022.
*W. Zhang, L. Shen, W. Zhang, and C.-S. Foo.*
[paper](https://www.ijcai.org/proceedings/2022/0232.pdf)

1. **MemREIN: Rein the domain shift for cross-domain few-shot learning,** in IJCAI, 2022.
*Y. Xu, L. Wang, Y. Wang, C. Qin, Y. Zhang, and Y. FU.*
[paper](https://www.ijcai.org/proceedings/2022/0505.pdf)

1. **Continual few-shot learning with transformer adaptation and knowledge regularization,** in WWW, 2023.
*X. Wang, Y. Liu, J. Fan, W. Wen, H. Xue, and W. Zhu.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583262)

1. **DENSE: Data-free one-shot federated learning,** in NeurIPS, 2022.
*J. Zhang, C. Chen, B. Li, L. Lyu, S. Wu, S. Ding, C. Shen, and C. Wu.*
[paper](https://openreview.net/pdf?id=QFQoxCFYEkA)

1. **Towards practical few-shot query sets: Transductive minimum description length inference,** in NeurIPS, 2022.
*S. T. Martin, M. Boudiaf, E. Chouzenoux, J.-C. Pesquet, and I. B. Ayed.*
[paper](https://openreview.net/pdf?id=j9JL96S8Vl)
[code](https://github.com/SegoleneMartin/PADDLE)

1. **Task-level differentially private meta learning,** in NeurIPS, 2022.
*X. Zhou, and R. Bassily.*
[paper](https://openreview.net/pdf?id=FhyrZ92DcI9)
[code](https://github.com/xyzhou055/MetaNSGD)

1. **FiT: Parameter efficient few-shot transfer learning for personalized and federated image classification,** in ICLR, 2023.
*A. Shysheya, J. F. Bronskill, M. Patacchiola, S. Nowozin, and R. E. Turner.*
[paper](https://openreview.net/pdf?id=9aokcgBVIj1)
[code](https://github.com/cambridge-mlg/fit)

1. **Towards addressing label skews in one-shot federated learning,** in ICLR, 2023.
*Y. Diao, Q. Li, and B. He.*
[paper](https://openreview.net/pdf?id=rzrqh85f4Sc)
[code](https://github.com/Xtra-Computing/FedOV)

1. **Data-free one-shot federated learning under very high statistical heterogeneity,** in ICLR, 2023.
*C. E. Heinbaugh, E. Luz-Ricca, and H. Shao.*
[paper](https://openreview.net/pdf?id=_hb4vM3jspB)
[code](https://github.com/ceh-2000/fed_cvae)

1. **Contrastive meta-learning for partially observable few-shot learning,** in ICLR, 2023.
*A. Jelley, A. Storkey, A. Antoniou, and S. Devlin.*
[paper](https://openreview.net/pdf?id=6iVJOtr2zL2)
[code](https://github.com/AdamJelley/POEM)

1. **On the soft-subnetwork for few-shot class incremental learning,** in ICLR, 2023.
*H. Kang, J. Yoon, S. R. H. Madjid, S. J. Hwang, and C. D. Yoo.*
[paper](https://openreview.net/pdf?id=z57WK5lGeHd)
[code](https://github.com/ihaeyong/ SoftNet-FSCIL)

1. **Warping the space: Weight space rotation for class-incremental few-shot learning,** in ICLR, 2023.
*D.-Y. Kim, D.-J. Han, J. Seo, and J. Moon.*
[paper](https://openreview.net/pdf?id=kPLzOfPfA2l)
[code](https://github.com/EdwinKim3069/WaRP-CIFSL)

1. **Neural collapse inspired feature-classifier alignment for few-shot class-incremental learning,** in ICLR, 2023.
*Y. Yang, H. Yuan, X. Li, Z. Lin, P. Torr, and D. Tao.*
[paper](https://openreview.net/pdf?id=y5W8tpojhtJ)
[code](https://github.com/NeuralCollapseApplications/FSCIL)

1. **Learning with fantasy: Semantic-aware virtual contrastive constraint for few-shot class-incremental learning,** in CVPR, 2023.
*Z. Song, Y. Zhao, Y. Shi, P. Peng, L. Yuan, and Y. Tian.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Learning_With_Fantasy_Semantic-Aware_Virtual_Contrastive_Constraint_for_Few-Shot_Class-Incremental_CVPR_2023_paper.pdf)
[code](https://github.com/zysong0113/SAVC)

1. **Few-shot class-incremental learning via class-aware bilateral distillation,** in CVPR, 2023.
*L. Zhao, J. Lu, Y. Xu, Z. Cheng, D. Guo, Y. Niu, and X. Fang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Few-Shot_Class-Incremental_Learning_via_Class-Aware_Bilateral_Distillation_CVPR_2023_paper.pdf)
[code](https://github.com/LinglanZhao/BiDistFSCIL)

1. **GKEAL: Gaussian kernel embedded analytic learning for few-shot class incremental task,** in CVPR, 2023.
*H. Zhuang, Z. Weng, R. He, Z. Lin, and Z. Zeng.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.pdf)

1. **Glocal energy-based learning for few-shot open-set recognition,** in CVPR, 2023.
*H. Wang, G. Pang, P. Wang, L. Zhang, W. Wei, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Glocal_Energy-Based_Learning_for_Few-Shot_Open-Set_Recognition_CVPR_2023_paper.pdf)
[code](https://github.com/00why00/Glocal)

1. **Open-set likelihood maximization for few-shot learning,** in CVPR, 2023.
*M. Boudiaf, E. Bennequin, M. Tami, A. Toubhans, P. Piantanida, C. Hudelot, and I. B. Ayed.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Boudiaf_Open-Set_Likelihood_Maximization_for_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/ebennequin/fewshot-open-set.)

1. **Federated few-shot learning,** in KDD, 2023.
*S. Wang, X. Fu, K. Ding, C. Chen, H. Chen, and J. Li.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599347)
[code](https://github.com/SongW-SW/F2L)

4. **LFS-GAN: Lifelong few-shot image generation,** in ICCV, 2023.
*J. Seo, J. Kang, and G. Park.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Seo_LFS-GAN_Lifelong_Few-Shot_Image_Generation_ICCV_2023_paper.pdf)
[code](https://github.com/jjuon/lfs-gan)

5. **Domain adaptive few-shot open-set learning,** in ICCV, 2023.
*D. Pal, D. More, S. Bhargav, D. Tamboli, V. Aggarwal, and B. Banerjee.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Pal_Domain_Adaptive_Few-Shot_Open-Set_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/DebabrataPal7/DAFOSNET)

19. **Few-shot continual infomax learning,** in ICCV, 2023.
*Z. Gu, C. Xu, J. Yang, and Z. Cui.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_Few-shot_Continual_Infomax_Learning_ICCV_2023_paper.pdf)

28. **Prototypical kernel learning and open-set foreground perception for generalized few-shot semantic segmentation,** in ICCV, 2023.
*K. Huang, F. Wang, Y. Xi, and Y. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Prototypical_Kernel_Learning_and_Open-set_Foreground_Perception_for_Generalized_Few-shot_ICCV_2023_paper.pdf)

29. **DETA: Denoised task adaptation for few-shot learning,** in ICCV, 2023.
*J. Zhang, L. Gao, X. Luo, H. Shen, and J. Song.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_DETA_Denoised_Task_Adaptation_for_Few-Shot_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/JimZAI/DETA)

8. **Few-shot class-incremental learning via training-free prototype calibration,** in NeurIPS, 2023.
*Q. Wang, D. Zhou, Y. Zhang, D. Zhan, and H. Ye.*
[paper](https://openreview.net/attachment?id=8NAxGDdf7H&name=pdf)
[code](https://github.com/wangkiw/TEEN)

2. **Alignment with human representations supports robust few-shot learning,** in NeurIPS, 2023.
*I. Sucholutsky, and T. L. Griffiths.*
[paper](https://openreview.net/attachment?id=HYGnmSLBCf&name=pdf)

## [Datasets/Benchmarks](#content)

1. **FewRel: A large-scale supervised few-shot relation classification dataset with state-of-the-art evaluation,** in EMNLP, 2018.
*X. Han, H. Zhu, P. Yu, Z. Wang, Y. Yao, Z. Liu, and M. Sun.*
[paper](https://www.aclweb.org/anthology/D18-1514.pdf)
[code](https://github.com/thunlp/FewRel)

1. **Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning,** arXiv preprint, 2019.
*T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine.*
[paper](https://arxiv.org/abs/1910.10897)
[code](https://meta-world.github.io/)

1. **The Omniglot challenge: A 3-year progress report,** in Current Opinion in Behavioral Sciences, 2019.
*B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum.*
[paper](https://arxiv.org/abs/1902.03477)
[code](https://github.com/brendenlake/omniglot)

1. **FewRel 2.0: Towards more challenging few-shot relation classification,** in EMNLP-IJCNLP, 2019.
*T. Gao, X. Han, H. Zhu, Z. Liu, P. Li, M. Sun, and J. Zhou.*
[paper](https://www.aclweb.org/anthology/D19-1649.pdf)
[code](https://github.com/thunlp/FewRel)

1. **META-DATASET: A dataset of datasets for learning to learn from few examples,** in ICLR, 2020.
*E. Triantafillou, T. Zhu, V. Dumoulin, P. Lamblin, U. Evci, K. Xu, R. Goroshin, C. Gelada, K. Swersky, P. Manzagol, and H. Larochelle.*
[paper](https://openreview.net/pdf?id=rkgAGAVKPr)
[code](https://github.com/google-research/meta-dataset)

1. **Few-shot object detection with attention-rpn and multi-relation detector,** in CVPR, 2020.
*Q. Fan, W. Zhuo, C.-K. Tang, Y.-W. Tai.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Few-Shot_Object_Detection_With_Attention-RPN_and_Multi-Relation_Detector_CVPR_2020_paper.pdf)
[code](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset)

1. **FSS-1000: A 1000-class dataset for few-shot segmentation,** in CVPR, 2020.
*X. Li, T. Wei, Y. P. Chen, Y.-W. Tai, and C.-K. Tang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_FSS-1000_A_1000-Class_Dataset_for_Few-Shot_Segmentation_CVPR_2020_paper.pdf)
[code](https://github.com/HKUSTCV/FSS-1000)

1. **Impact of base dataset design on few-shot image classification,** in ECCV, 2020.
*O. Sbai, C. Couprie, and M. Aubry.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610579.pdf)
[code](https://github.com/facebookresearch/fewshotDatasetDesign)

1. **A unified few-shot classification benchmark to compare transfer and meta learning approaches,** in NeurIPS, 2021.
*V. Dumoulin, N. Houlsby, U. Evci, X. Zhai, R. Goroshin, S. Gelly, and H. Larochelle.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/28dd2c7955ce926456240b2ff0100bde-Paper-round1.pdf)

1. **Few-shot learning evaluation in natural language understanding,** in NeurIPS, 2021.
*S. Mukherjee, X. Liu, G. Zheng, S. Hosseini, H. Cheng, G. Yang, C. Meek, A. H. Awadallah, and J. Gao.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/3644a684f98ea8fe223c713b77189a77-Paper-round2.pdf)
[code](https://github.com/microsoft/CLUES)

1. **FS-Mol: A few-shot learning dataset of molecules,** in NeurIPS, 2021.
*M. Stanley, J. Bronskill, K. Maziarz, H. Misztela, J. Lanini, M. H. S. Segler, N. Schneider, and M. Brockschmidt.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/8d3bba7425e7c98c50f52ca1b52d3735-Paper-round2.pdf)
[code](https://github.com/microsoft/FS-Mol/)

1. **RAFT: A real-world few-shot text classification benchmark,** in NeurIPS, 2021.
*N. Alex, E. Lifland, L. Tunstall, A. Thakur, P. Maham, C. J. Riedel, E. Hine, C. Ashurst, P. Sedille, A. Carlier, M. Noetel, and A. Stuhlmüller.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/ca46c1b9512a7a8315fa3c5a946e8265-Paper-round2.pdf)
[code](https://raft.elicit.org/)

1. **A large-scale benchmark for few-shot program induction and synthesis,** in ICML, 2021.
*F. Alet, J. Lopez-Contreras, J. Koppel, M. Nye, A. Solar-Lezama, T. Lozano-Perez, L. Kaelbling, and J. Tenenbaum.*
[paper](http://proceedings.mlr.press/v139/alet21a/alet21a.pdf)
[code](https://github.com/javierlc2000/progres)

1. **FEW-NERD: A few-shot named entity recognition dataset,** in ACL-IJCNLP, 2021.
*N. Ding, G. Xu, Y. Chen, X. Wang, X. Han, P. Xie, H. Zheng, and Z. Liu.*
[paper](https://aclanthology.org/2021.acl-long.248.pdf)
[code](https://ningding97.github.io/fewnerd/)

1. **CrossFit: A few-shot learning challenge for cross-task generalization in NLP,** in EMNLP, 2021.
*Q. Ye, B. Y. Lin, and X. Ren.*
[paper](https://aclanthology.org/2021.emnlp-main.572.pdf)
[code](https://github.com/INK-USC/CrossFit)

1. **ORBIT: A real-world few-shot dataset for teachable object recognition,** in ICCV, 2021.
*D. Massiceti, L. Zintgraf, J. Bronskill, L. Theodorou, M. T. Harris, E. Cutrell, C. Morrison, K. Hofmann, and S. Stumpf.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Massiceti_ORBIT_A_Real-World_Few-Shot_Dataset_for_Teachable_Object_Recognition_ICCV_2021_paper.pdf)
[code](https://github.com/microsoft/ORBIT-Dataset)

1. **FLEX: Unifying evaluation for few-shot NLP,** in NeurIPS, 2021.
*J. Bragg, A. Cohan, K. Lo, and I. Beltagy.*
[paper](https://proceedings.neurips.cc/paper/2021/file/8493eeaccb772c0878f99d60a0bd2bb3-Paper.pdf)

1. **Two sides of meta-learning evaluation: In vs. out of distribution,** in NeurIPS, 2021.
*A. Setlur, O. Li, and V. Smith.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1e932f24dc0aa4e7a6ac2beec387416d-Paper.pdf)

1. **Realistic evaluation of transductive few-shot learning,** in NeurIPS, 2021.
*O. Veilleux, M. Boudiaf, P. Piantanida, and I. B. Ayed.*
[paper](https://proceedings.neurips.cc/paper/2021/file/4d7a968bb636e25818ff2a3941db08c1-Paper.pdf)

1. **Meta-Album: Multi-domain meta-dataset for few-shot image classification,** in NeurIPS, 2022.
*I. Ullah, D. Carrión-Ojeda, S. Escalera, I. Guyon, M. Huisman, F. Mohr, J. N. v. Rijn, H. Sun, J. Vanschoren, and P. A. Vu.*
[paper](https://papers.nips.cc/paper_files/paper/2022/file/1585da86b5a3c4fb15520a2b3682051f-Paper-Datasets_and_Benchmarks.pdf)
[code](https://meta-album.github.io/)

1. **Geoclidean: Few-shot generalization in euclidean geometry,** in NeurIPS, 2022.
*J. Hsu, J. Wu, and N. D. Goodman.*
[paper](https://papers.nips.cc/paper_files/paper/2022/file/feb34ce77fc8b94c85d12e608b23ce67-Paper-Datasets_and_Benchmarks.pdf)
[code](https://github.com/joyhsu0504/geoclidean_framewor)

1. **FewNLU: Benchmarking state-of-the-art methods for few-shot natural language understanding,** in ACL, 2022.
*Y. Zheng, J. Zhou, Y. Qian, M. Ding, C. Liao, L. Jian, R. Salakhutdinov, J. Tang, S. Ruder, and Z. Yang.*
[paper](https://aclanthology.org/2022.acl-long.38.pdf)
[code](https://github.com/THUDM/FewNLU)

1. **Bongard-HOI: Benchmarking few-shot visual reasoning for human-object interactions,** in CVPR, 2022.
*H. Jiang, X. Ma, W. Nie, Z. Yu, Y. Zhu, and A. Anandkumar.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Bongard-HOI_Benchmarking_Few-Shot_Visual_Reasoning_for_Human-Object_Interactions_CVPR_2022_paper.pdf)
[code](https://github.com/nvlabs/Bongard-HOI)

1. **Hard-Meta-Dataset++: Towards understanding few-shot performance on difficult tasks,** in ICLR, 2023.
*S. Basu, M. Stanley, J. F. Bronskill, S. Feizi, and D. Massiceti.*
[paper](https://openreview.net/pdf?id=wq0luyH3m4)

1. **MEWL: Few-shot multimodal word learning with referential uncertainty,** in ICML, 2023.
*G. Jiang, M. Xu, S. Xin, W. Liang, Y. Peng, C. Zhang, and Y. Zhu.*
[paper](https://proceedings.mlr.press/v202/jiang23i/jiang23i.pdf)
[code](https://github.com/jianggy/MEWL)

1. **UNISUMM and SUMMZOO: Unified model and diverse benchmark for few-shot summarization,** in ACL, 2023.
*Y. Chen, Y. Liu, R. Xu, Z. Yang, C. Zhu, M. Zeng, and Y. Zhang.*
[paper](https://aclanthology.org/2023.acl-long.718.pdf)
[code](https://github.com/microsoft/UniSumm)

1. **EHRSHOT: An EHR benchmark for few-shot evaluation of foundation models,** in NeurIPS, 2023.
*M. Wornow, R. Thapa, E. Steinberg, J. A. Fries, and N. Shah.*
[paper](https://openreview.net/attachment?id=CsXC6IcdwI&name=pdf)
[code](https://github.com/som-shahlab/ehrshot-benchmark)



## [Software Library](#content)

1. **PaddleFSL,** a library for few-shot learning written in *PaddlePaddle*.
[link](https://github.com/tata1661/FSL-Mate/tree/master/PaddleFSL)

1. **Torchmeta,** a library for few-shot learning & meta-learning written in *PyTorch*.
[link](https://github.com/tristandeleu/pytorch-meta#torchmeta)

1. **learn2learn,** a library for meta-learning written in *PyTorch*.
[link](https://github.com/learnables/learn2learn)

1. **keras-fsl,** a library for few-shot learning written in *Tensorflow*.
[link](https://github.com/few-shot-learning/Keras-FewShotLearning)





