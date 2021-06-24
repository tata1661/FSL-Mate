# Few-Shot Papers

This repository contains few-shot learning (FSL) papers mentioned in our FSL survey published in ACM Computing Surveys (JCR Q1, CORE A*). 

For convenience, we also include public implementations of respective authors.

We will update this paper list to include new FSL papers periodically. 
The current version is updated on 2021.02.04.

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
    1. [Embedding Learning](#Embedding-Learning)
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
    1. [Acoustic Signal Processing](#Acoustic-Signal-Processing)
    1. [Recommendation](#Recommendation)
    1. [Others](#others)
6. [Theories](#Theories)
7. [Data Sets](#Data-Sets)
8. [Few-shot Learning and Zero-shot Learning](#Few-shot-Learning-and-Zero-shot-Learning)
9. [Software Library](#Software-Library)


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
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Douze_Low-Shot_Learning_With_CVPR_2018_paper.pdf)
[code](https://github.com/leokarlin/LaSO)

1. **Image deformation meta-networks for one-shot learning,** in CVPR, 2019.
*Z. Chen, Y. Fu, Y.-X. Wang, L. Ma, W. Liu, and M. Hebert.*
[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Image_Deformation_Meta-Networks_for_One-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/tankche1/IDeMe-Net)

1. **Spot and learn: A maximum-entropy patch sampler for few-shot image classification,** in CVPR, 2019.
*W.-H. Chu, Y.-J. Li, J.-C. Chang, and Y.-C. F. Wang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf)

1. **Data augmentation using learned transformations for one-shot medical image segmentation,** in CVPR, 2019.
*A. Zhao, G. Balakrishnan, F. Durand, J. V. Guttag, and A. V. Dalca.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Data_Augmentation_Using_Learned_Transformations_for_One-Shot_Medical_Image_Segmentation_CVPR_2019_paper.pdf)

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

1. **Multi-content GAN for few-shot font style transfer,** in CVPR, 2018. 
*S. Azadi, M. Fisher, V. G. Kim, Z. Wang, E. Shechtman, and T. Darrell.*
[paper](http://www.vovakim.com/papers/18_CVPRSpotlight_FontDropper.pdf)
[code](https://github.com/azadis/MC-GAN)

1. **Feature space transfer for data augmentation,** in CVPR, 2018.
*B. Liu, X. Wang, M. Dixit, R. Kwitt, and N. Vasconcelos.* 
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Feature_Space_Transfer_CVPR_2018_paper.pdf)

1. **One-shot unsupervised cross domain translation,** in NeurIPS, 2018.
*S. Benaim and L. Wolf.* 
[paper](https://papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation.pdf)

1. **Fine-grained visual categorization using meta-learning optimization with sample selection of auxiliary data,** in ECCV, 2018.
*Y. Zhang, H. Tang, and K. Jia.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yabin_Zhang_Fine-Grained_Visual_Categorization_ECCV_2018_paper.pdf)

1. **Few-shot charge prediction with discriminative legal attributes,** in COLING, 2018.
*Z. Hu, X. Li, C. Tu, Z. Liu, and M. Sun.*
[paper](https://www.aclweb.org/anthology/C18-1041.pdf)

1. **Bidirectional one-shot unsupervised domain mapping,** in ICCV, 2019.
*T. Cohen, and L. Wolf*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cohen_Bidirectional_One-Shot_Unsupervised_Domain_Mapping_ICCV_2019_paper.pdf)

1. **Boosting few-shot visual learning with self-supervision,** in ICCV, 2019.
*S. Gidaris, A. Bursuc, N. Komodakis, P. Pérez, and M. Cord*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gidaris_Boosting_Few-Shot_Visual_Learning_With_Self-Supervision_ICCV_2019_paper.pdf)

1. **When does self-supervision improve few-shot learning?,** in ECCV, 2020.
*J. Su, S. Maji, and B. Hariharan.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520630.pdf)

1. **Pareto self-supervised training for few-shot learning,** in CVPR, 2021.
*Z. Chen, J. Ge, H. Zhan, S. Huang, and D. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pareto_Self-Supervised_Training_for_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Prototypical cross-domain self-supervised learning for few-shot unsupervised domain adaptation,** in CVPR, 2021.
*X. Yue, Z. Zheng, S. Zhang, Y. Gao, T. Darrell, K. Keutzer, and A. S. Vincentelli.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Prototypical_Cross-Domain_Self-Supervised_Learning_for_Few-Shot_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)



### Embedding Learning

1. **Object classification from a single example utilizing class relevance metrics,** in NeurIPS, 2005.
*M. Fink.*
[paper](https://papers.nips.cc/paper/2576-object-classification-from-a-single-example-utilizing-class-relevance-metrics.pdf)

1. **Optimizing one-shot recognition with micro-set learning,** in CVPR, 2010.
*K. D. Tang, M. F. Tappen, R. Sukthankar, and C. H. Lampert.*
[paper](http://www.cs.ucf.edu/~mtappen/pubs/cvpr10_oneshot.pdf)

1. **Siamese neural networks for one-shot image recognition,** ICML deep learning workshop, 2015.
*G. Koch, R. Zemel, and R. Salakhutdinov*
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

1. **Induction networks for few-shot text classification,** in EMNLP, 2019.
*R. Geng, B. Li, Y. Li, X. Zhu, P. Jian, and J. Sun.*
[paper](https://www.aclweb.org/anthology/D19-1403.pdf)

1. **Hierarchical attention prototypical networks for few-shot text classification,** in EMNLP, 2019.
*S. Sun, Q. Sun, K. Zhou, and T. Lv.*
[paper](https://www.aclweb.org/anthology/D19-1045.pdf)

1. **Cross attention network for few-shot classification,** in NeurIPS, 2019.
*R. Hou, H. Chang, B. Ma, S. Shan, and X. Chen.*
[paper](https://papers.nips.cc/paper/8655-cross-attention-network-for-few-shot-classification.pdf)

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

1. **Cross-domain few-shot classification via learned feature-wise transformation,** in ICLR, 2020.
*H. Tseng, H. Lee, J. Huang, and M. Yang.*
[paper](https://openreview.net/pdf?id=SJl5Np4tPr)
[code](https://github.com/hytseng0509/CrossDomainFewShot)

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

1. **XtarNet: Learning to extract task-adaptive representation for incremental few-shot learning,** in ICML, 2020:.
*S. W. Yoon, D. Kim, J. Seo, and J. Moon.*
[paper](http://proceedings.mlr.press/v119/yoon20b/yoon20b.pdf)
[code](https://github.com/EdwinKim3069/XtarNet)

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

1. **Self-promoted prototype refinement for few-shot class-incremental learning,** in CVPR, 2021.
*K. Zhu, Y. Cao, W. Zhai, J. Cheng, and Z. Zha.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf)

1. **Few-shot classification with feature map reconstruction networks,** in CVPR, 2021.
*D. Wertheimer, L. Tang, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.pdf)
[code](https://github.com/Tsingularity/FRN)

1. **ECKPN: Explicit class knowledge propagation network for transductive few-shot learning,** in CVPR, 2021.
*C. Chen, X. Yang, C. Xu, X. Huang, and Z. Ma.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_ECKPN_Explicit_Class_Knowledge_Propagation_Network_for_Transductive_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Semantic-aware knowledge distillation for few-shot class-incremental learning,** in CVPR, 2021.
*A. Cheraghian, S. Rahman, P. Fang, S. K. Roy, L. Petersson, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheraghian_Semantic-Aware_Knowledge_Distillation_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf)

1. **Exploring complementary strengths of invariant and equivariant representations for few-shot learning,** in CVPR, 2021.
*M. N. Rizve, S. Khan, F. S. Khan, and M. Shah.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Rizve_Exploring_Complementary_Strengths_of_Invariant_and_Equivariant_Representations_for_Few-Shot_CVPR_2021_paper.pdf)

1. **Rethinking class relations: Absolute-relative supervised and unsupervised few-shot learning,** in CVPR, 2021.
*H. Zhang, P. Koniusz, S. Jian, H. Li, and P. H. S. Torr.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Rethinking_Class_Relations_Absolute-Relative_Supervised_and_Unsupervised_Few-Shot_Learning_CVPR_2021_paper.pdf)

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
*J. Kim, T.-H. Oh, S. Lee, F. Pan, and I. S. Kweon*
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

1. **Incremental few-shot learning with attention attractor networks,** in NeurIPS, 2019.
*M. Ren, R. Liao, E. Fetaya, and R. S. Zemel.*
[paper](https://papers.nips.cc/paper/8769-incremental-few-shot-learning-with-attention-attractor-networks.pdf)
[code](https://github.com/renmengye/inc-few-shot-attractor-public)

1. **Low shot box correction for weakly supervised object detection,** in IJCAI, 2019.
*T. Pan, B. Wang, G. Ding, J. Han, and J. Yong*
[paper](https://www.ijcai.org/Proceedings/2019/0125.pdf)

1. **Diversity with cooperation: Ensemble methods for few-shot classification,** in ICCV, 2019.
*N. Dvornik, C. Schmid, and J. Mairal*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dvornik_Diversity_With_Cooperation_Ensemble_Methods_for_Few-Shot_Classification_ICCV_2019_paper.pdf)

1. **Few-shot image recognition with knowledge transfer,** in ICCV, 2019.
*Z. Peng, Z. Li, J. Zhang, Y. Li, G.-J. Qi, and J. Tang*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Few-Shot_Image_Recognition_With_Knowledge_Transfer_ICCV_2019_paper.pdf)

1. **Generating classification weights with gnn denoising autoencoders for few-shot learning,** in CVPR, 2019.
*S. Gidaris, and N. Komodakis.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gidaris_Generating_Classification_Weights_With_GNN_Denoising_Autoencoders_for_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/gidariss/wDAE_GNN_FewShot)

1. **Dense classification and implanting for few-shot learning,** in CVPR, 2019.
*Y. Lifchitz, Y. Avrithis, S. Picard, and A. Bursuc*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lifchitz_Dense_Classification_and_Implanting_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

1. **Few-shot adaptive faster R-CNN,** in CVPR, 2019.
*T. Wang, X. Zhang, L. Yuan, and J. Feng*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Few-Shot_Adaptive_Faster_R-CNN_CVPR_2019_paper.pdf)

1. **Few-shot class-incremental learning,** in CVPR, 2020.
*X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tao_Few-Shot_Class-Incremental_Learning_CVPR_2020_paper.pdf)

1. **TransMatch: A transfer-learning scheme for semi-supervised few-shot learning,** in CVPR, 2020.
*Z. Yu, L. Chen, Z. Cheng, and J. Luo*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_TransMatch_A_Transfer-Learning_Scheme_for_Semi-Supervised_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Learning to select base classes for few-shot classification,** in CVPR, 2020.
*L. Zhou, P. Cui, X. Jia, S. Yang, and Q. Tian*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Learning_to_Select_Base_Classes_for_Few-Shot_Classification_CVPR_2020_paper.pdf)

1. **Few-shot NLG with pre-trained language model,** in ACL, 2020.
*Z. Chen, H. Eavani, W. Chen, Y. Liu, and W. Y. Wang.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.18.pdf)
[code](https://github.com/czyssrs/Few-Shot-NLG)

1. **Span-ConveRT: Few-shot span extraction for dialog with pretrained conversational representations,** in ACL, 2020.
*S. Coope, T. Farghly, D. Gerz, I. Vulic, and M. Henderson.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.11.pdf)

1. **A baseline for few-shot image classification,** in ICLR, 2020.
*G. S. Dhillon, P. Chaudhari, A. Ravichandran, and S. Soatto.*
[paper](https://openreview.net/pdf?id=rylXBkrYDS)

1. **Graph few-shot learning via knowledge transfer,** in AAAI, 2020.
*H. Yao, C. Zhang, Y. Wei, M. Jiang, S. Wang, J. Huang, N. V. Chawla, and Z. Li.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6142)

1. **Knowledge graph transfer network for few-shot recognition,** in AAAI, 2020.
*R. Chen, T. Chen, X. Hui, H. Wu, G. Li, and L. Lin.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6630)

1. **Context-Transformer: Tackling object confusion for few-shot detection,** in AAAI, 2020.
*Z. Yang, Y. Wang, X. Chen, J. Liu, and Y. Qiao.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6957)

1. **Selecting relevant features from a multi-domain representation for few-shot classification,** in ECCV, 2020.
*N. Dvornik, C. Schmid, and J. Mairal.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550766.pdf)
[code](https://github.com/dvornikita/SUR)

1. **A universal representation transformer layer for few-shot image classification,** in ICLR, 2021.
*L. Liu, W. L. Hamilton, G. Long, J. Jiang, and H. Larochelle.*
[paper](https://openreview.net/pdf?id=04cII6MumYV)

1. **Prototype completion with primitive knowledge for few-shot learning,** in CVPR, 2021.
*B. Zhang, X. Li, Y. Ye, Z. Huang, and L. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Prototype_Completion_With_Primitive_Knowledge_for_Few-Shot_Learning_CVPR_2021_paper.pdf)
[code](https://github.com/zhangbq-research/Prototype_Completion_for_FSL)

1. **Few-shot incremental learning with continually evolved classifiers,** in CVPR, 2021.
*C. Zhang, N. Song, G. Lin, Y. Zheng, P. Pan, and Y. Xu.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)

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

1. **Meta relational learning for few-shot link prediction in knowledge graphs,** in EMNLP, 2019.
*M. Chen, W. Zhang, W. Zhang, Q. Chen, and H. Chen.*
[paper](https://www.aclweb.org/anthology/D19-1431.pdf)

1. **Adapting meta knowledge graph information for multi-hop reasoning over few-shot relations,** in EMNLP, 2019.
*X. Lv, Y. Gu, X. Han, L. Hou, J. Li, and Z. Liu.*
[paper](https://www.aclweb.org/anthology/D19-1334.pdf)

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


### Learning Search Steps

1. **Optimization as a model for few-shot learning,** in ICLR, 2017.
*S. Ravi and H. Larochelle.*
[paper](https://openreview.net/forum?id=rJY0-Kcll)
[code](https://github.com/twitter/meta-learning-lstm)


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

1. **Self-supervision with superpixels: Training few-shot medical image segmentation without annotation,** in ECCV, 2020.
*C. Ouyang, C. Biffi, C. Chen, T. Kart, H. Qiu, and D. Rueckert.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740749.pdf)
[code](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)

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

1. **Deep complementary joint model for complex scene registration and few-shot segmentation on medical images,** in ECCV, 2020.
*Y. He, T. Li, G. Yang, Y. Kong, Y. Chen, H. Shu, J. Coatrieux, J. Dillenseger, and S. Li.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630749.pdf)

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
*Y. Li, R. Zhang, J. (. Lu, and E. Shechtman.*
[paper](https://proceedings.neurips.cc/paper/2020/file/b6d767d2f8ed5d21a44b0e5886680cb9-Paper.pdf)

1. **Few-shot visual reasoning with meta-analogical contrastive learning,** in NeurIPS, 2020.
*Y. Kim, J. Shin, E. Yang, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/c39e1a03859f9ee215bc49131d0caf33-Paper.pdf)

1. **Crosstransformers: spatially-aware few-shot transfer,** in NeurIPS, 2020.
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

1. **Temporal-relational crosstransformers for few-shot action recognition,** in CVPR, 2021.
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

1. **What if we only use real datasets for scene text recognition? toward scene text recognition with fewer labels,** in CVPR, 2021.
*J. Baek, Y. Matsui, and K. Aizawa.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf)
[code](https://github.com/ku21fan/STR-Fewer-Labels)

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

1. **Continuous adaptation via meta-learning in nonstationary and competitive environments,** in ICLR, 2018.
*M. Al-Shedivat, T. Bansal, Y. Burda, I. Sutskever, I. Mordatch, and P. Abbeel.*
[paper](https://openreview.net/forum?id=Sk2u1g-0-)

1. **Deep online learning via meta-learning: Continual adaptation for model-based RL,** in ICLR, 2018.
*A. Nagabandi, C. Finn, and S. Levine.*
[paper](https://openreview.net/references/pdf?id=ryuIpa6S4)

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

1. **Wandering within a world: Online contextualized few-shot learning,** in ICLR, 2021.
*M. Ren, M. L. Iuzzolino, M. C. Mozer, and R. Zemel.*
[paper](https://openreview.net/pdf?id=oZIvHV04XgC)


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

1. **Few-shot knowledge graph completion,** in AAAI, 2020.
*C. Zhang, H. Yao, C. Huang, M. Jiang, Z. Li, and N. V. Chawla.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5698)

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

1. **Adaptive attentional network for few-shot knowledge graph completion,** in EMNLP, 2020.
*J. Sheng, S. Guo, Z. Chen, J. Yue, L. Wang, T. Liu, and H. Xu.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.131.pdf)
[code](https://github.com/JiaweiSheng/FAAN)

1. **Few-shot complex knowledge base question answering via meta reinforcement learning,** in EMNLP, 2020.
*Y. Hua, Y. Li, G. Haffari, G. Qi, and T. Wu.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.469.pdf)
[code](https://github.com/DevinJake/MRL-CQA)

1. **Self-supervised meta-learning for few-shot natural language classification tasks,** in EMNLP, 2020.
*T. Bansal, R. Jha, T. Munkhdalai, and A. McCallum.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.38.pdf)
[code](https://github.com/iesl/metanlp)

1. **Structural supervision improves few-shot learning and syntactic generalization in neural language models,** in EMNLP, 2020.
*E. Wilcox, P. Qian, R. Futrell, R. Kohita, R. Levy, and M. Ballesteros.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.375.pdf)
[code](https://github.com/wilcoxeg/fsl_invar)

1. **Language models are few-shot learners,** in NeurIPS, 2020.
*T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei.*
[paper](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

1. **Uncertainty-aware self-training for few-shot text classification,** in NeurIPS, 2020.
*S. Mukherjee, and A. Awadallah.*
[paper](https://proceedings.neurips.cc/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf)
[code](https://github.com/microsoft/UST)

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


### Others

1. **Low data drug discovery with one-shot learning,** ACS Central Science, 2017.
*H. Altae-Tran, B. Ramsundar, A. S. Pappu, and V. Pande.* 
[paper](https://arxiv.org/abs/1611.03199)

1. **SMASH: One-shot model architecture search through hypernetworks,** in ICLR, 2018.
*A. Brock, T. Lim, J. Ritchie, and N. Weston.*
[paper](https://openreview.net/forum?id=rydeCEhs-)

1. **MetaEXP: Interactive explanation and exploration of large knowledge graphs,** in WWW, 2018.
*F. Behrens, S. Bischoff, P. Ladenburger, J. Rückin, L. Seidel, F. Stolp, M. Vaichenker, A. Ziegler, D. Mottin, F. Aghaei, E. Müller, M. Preusse, N. Müller, and M. Hunger.*
[paper](https://meta-exp.github.io/resources/paper.pdf)
[code](https://hpi.de/en/mueller/metaex)

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

1. **Learning from multiple cities: A meta-learning approach for spatial-temporal prediction,** in WWW, 2019.
*H. Yao, Y. Liu, Y. Wei, X. Tang, and Z. Li.*
[paper](https://arxiv.org/pdf/1901.08518.pdf)
[code](https://github.com/huaxiuyao/MetaST)

1. **Few-shot pill recognition,** in CVPR, 2020.
*S. Ling, A. Pastor, J. Li, Z. Che, J. Wang, J. Kim, and P. L. Callet.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ling_Few-Shot_Pill_Recognition_CVPR_2020_paper.pdf)

1. **LT-Net: Label transfer by learning reversible voxel-wise correspondence for one-shot medical image segmentation,** in CVPR, 2020.
*S. Wang, S. Cao, D. Wei, R. Wang, K. Ma, L. Wang, D. Meng, and Y. Zheng.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_LT-Net_Label_Transfer_by_Learning_Reversible_Voxel-Wise_Correspondence_for_One-Shot_CVPR_2020_paper.pdf)

1. **Federated meta-learning for fraudulent credit card detection,** in IJCAI, 2020.
*W. Zheng, L. Yan, C. Gou, and F. Wang.*
[paper](https://www.ijcai.org/Proceedings/2020/0642.pdf)

1. **Differentially private meta-learning,** in ICLR, 2020.
*J. Li, M. Khodak, S. Caldas, and A. Talwalkar.*
[paper](https://openreview.net/pdf?id=rJgqMRVYvr)

1. **Towards fast adaptation of neural architectures with meta learning,** in ICLR, 2020.
*D. Lian, Y. Zheng, Y. Xu, Y. Lu, L. Lin, P. Zhao, J. Huang, and S. Gao.*
[paper](https://openreview.net/pdf?id=r1eowANFvr)

1. **Learning to extrapolate knowledge: Transductive few-shot out-of-graph link prediction,** in NeurIPS, 2020:.
*J. Baek, D. B. Lee, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/0663a4ddceacb40b095eda264a85f15c-Paper.pdf)
[code](https://github.com/JinheonBaek/GEN)

1. **Repurposing pretrained models for robust out-of-domain few-shot learning,** in ICLR, 2021.
*N. Kwon, H. Na, G. Huang, and S. Lacoste-Julien.*
[paper](https://openreview.net/pdf?id=qkLMTphG5-h)
[code](https://anonymous.4open.science/r/08ef52cf-456a-4e36-a408-04e1ad0bc5a9/)

1. **Using optimal embeddings to learn new intents with few examples: An application in the insurance domain,** in KDD, 2020:.
*S. Acharya, and G. Fung.*
[paper](http://ceur-ws.org/Vol-2666/KDD_Converse20_paper_10.pdf)

1. **Meta-learning for query conceptualization at web scale,** in KDD, 2020.
*F. X. Han, D. Niu, H. Chen, W. Guo, S. Yan, and B. Long.*
[paper](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/fhan-KDD20.pdf)

1. **Few-sample and adversarial representation learning for continual stream mining,** in WWW, 2020.
*Z. Wang, Y. Wang, Y. Lin, E. Delord, and L. Khan.*
[paper](https://dl.acm.org/doi/10.1145/3366423.3380153)


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


## [Data Sets](#content)

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

1. **FewRel 2.0: Towards more challenging few-shot relation classification,** in EMNLP, 2019.
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

1. **A broader study of cross-domain few-shot learning,** in ECCV, 2020.
*Y. Guo, N. C. Codella, L. Karlinsky, J. V. Codella, J. R. Smith, K. Saenko, T. Rosing, and R. Feris.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf)
[code](https://github.com/IBM/cdfsl-benchmark)

1. **Impact of base dataset design on few-shot image classification,** in ECCV, 2020.
*O. Sbai, C. Couprie, and M. Aubry.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610579.pdf)
[code](https://github.com/facebookresearch/fewshotDatasetDesign)


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


## [Software Library](#content)

1. **Torchmeta,** a library for few-shot learning & meta-learning baselines written in *PyTorch*.
[link](https://github.com/tristandeleu/pytorch-meta#torchmeta)

1. **learn2learn,** a library for meta-learning baselines written in *PyTorch*.
[link](https://github.com/learnables/learn2learn)

1. **keras-fsl,** a library for few-shot learning baselines written in *Tensorflow*.
[link](https://github.com/few-shot-learning/Keras-FewShotLearning)

1. **PaddleFSL,** a library for few-shot learning baselines written in *PaddlePaddle*.
[link](https://github.com/PaddlePaddle/Research/tree/master/CV/PaddleFSL)




