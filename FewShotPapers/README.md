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
1. [Survey](#survey)
2. [Data](#data)
3. [Model](#model)  
    1. [Multitask Learning](#multitask-learning)  
    1. [Embedding/Metric Learning](#embeddingmetric-learning)  
    1. [Learning with External Memory](#learning-with-external-memory)  
    1. [Generative Modeling](#generative-modeling)
4. [Algorithm](#algorithm)  
    1. [Refining Existing Parameters](#refining-existing-parameters)  
    1. [Refining Meta-learned Parameters](#refining-meta-learned-parameters)  
    1. [Learning Search Steps](#learning-search-steps)
5. [Applications](#applications)  
    1. [Computer Vision](#computer-vision)  
    1. [Robotics](#robotics)  
    1. [Natural Language Processing](#natural-language-processing)  
    1. [Acoustic Signal Processing](#acoustic-signal-processing)  
    1. [Graph Learning](#graph-learning)  
    1. [Recommendation](#recommendation)  
    1. [Anomaly Detection](#anomaly-detection)  
    1. [AI for Science](#ai-for-science)  
    1. [AI for Healthcare](#ai-for-healthcare)  
    1. [Others](#others)
6. [Theories](#theories)
7. [Few-shot Learning and Zero-shot Learning](#few-shot-learning-and-zero-shot-learning)
8. [Variants of Few-shot Learning](#variants-of-few-shot-learning)
9. [Datasets/Benchmarks](#datasetsbenchmarks)
10. [Software Library](#software-library)



## [Survey](#content)

1. **Generalizing From a Few Examples: A Survey on Few-Shot Learning,** in CSUR, 2020.
*Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni.*
[paper](https://dl.acm.org/doi/10.1145/3386252?cid=99659542534)
[arXiv](https://arxiv.org/abs/1904.05046)

## [Data](#content)

1. **Learning From One Example Through Shared Densities on Transforms,** in CVPR, 2000.
*E. G. Miller, N. E. Matsakis, and P. A. Viola.*
[paper](https://people.cs.umass.edu/~elm/papers/Miller_congealing.pdf)

1. **Domain-Adaptive Discriminative One-Shot Learning of Gestures,** in ECCV, 2014.
*T. Pfister, J. Charles, and A. Zisserman.*
[paper](https://www.robots.ox.ac.uk/~vgg/publications/2014/Pfister14/pfister14.pdf)

1. **One-Shot Learning of Scene Locations via Feature Trajectory Transfer,** in CVPR, 2016.
*R. Kwitt, S. Hegenbart, and M. Niethammer.* 
[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Kwitt_One-Shot_Learning_of_CVPR_2016_paper.pdf)

1. **Low-Shot Visual Recognition by Shrinking and Hallucinating Features,** in ICCV, 2017.
*B. Hariharan and R. Girshick.*
[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hariharan_Low-Shot_Visual_Recognition_ICCV_2017_paper.pdf)
[code](https://github.com/facebookresearch/low-shot-shrink-hallucinate)

1. **Fast Parameter Adaptation for Few-Shot Image Captioning and Visual Question Answering,** in ACM MM, 2018.
*X. Dong, L. Zhu, D. Zhang, Y. Yang, and F. Wu.* 
[paper](https://xuanyidong.com/resources/papers/ACM-MM-18-FPAIT.pdf)

1. **Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning,** in CVPR, 2018.
*Y. Wu, Y. Lin, X. Dong, Y. Yan, W. Ouyang, and Y. Yang.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf)

1. **Low-Shot Learning With Large-Scale Diffusion,** in CVPR, 2018.
*M. Douze, A. Szlam, B. Hariharan, and H. Jégou.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Douze_Low-Shot_Learning_With_CVPR_2018_paper.pdf)

1. **Diverse Few-Shot Text Classification With Multiple Metrics,** in NAACL-HLT, 2018.
*M. Yu, X. Guo, J. Yi, S. Chang, S. Potdar, Y. Cheng, G. Tesauro, H. Wang, and B. Zhou.* 
[paper](https://www.aclweb.org/anthology/N18-1109.pdf)
[code](https://github.com/Gorov/DiverseFewShot_Amazon)

1. **Delta-Encoder: An Effective Sample Synthesis Method for Few-Shot Object Recognition,** in NeurIPS, 2018.
*E. Schwartz, L. Karlinsky, J. Shtok, S. Harary, M. Marder, A. Kumar, R. Feris, R. Giryes, and A. Bronstein.*
[paper](https://papers.nips.cc/paper/7549-delta-encoder-an-effective-sample-synthesis-method-for-few-shot-object-recognition.pdf)

1. **Low-Shot Learning via Covariance-Preserving Adversarial Augmentation Networks,** in NeurIPS, 2018.
*H. Gao, Z. Shou, A. Zareian, H. Zhang, and S. Chang.*
[paper](https://papers.nips.cc/paper/7376-low-shot-learning-via-covariance-preserving-adversarial-augmentation-networks.pdf)

1. **Learning to Self-Train for Semi-Supervised Few-Shot Classification,** in NeurIPS, 2019.
*X. Li, Q. Sun, Y. Liu, S. Zheng, Q. Zhou, T.-S. Chua, and B. Schiele.*
[paper](https://papers.nips.cc/paper/9216-learning-to-self-train-for-semi-supervised-few-shot-classification.pdf)

1. **Few-Shot Learning With Global Class Representations,** in ICCV, 2019.
*A. Li, T. Luo, T. Xiang, W. Huang, and L. Wang.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Few-Shot_Learning_With_Global_Class_Representations_ICCV_2019_paper.pdf)

1. **AutoAugment: Learning Augmentation Policies From Data,** in CVPR, 2019.
*E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le.*
[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf)

1. **EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks,** in EMNLP and IJCNLP, 2019.
*J. Wei and K. Zou.*
[paper](https://www.aclweb.org/anthology/D19-1670.pdf)

1. **LaSO: Label-Set Operations Networks for Multi-Label Few-Shot Learning,** in CVPR, 2019.
*A. Alfassy, L. Karlinsky, A. Aides, J. Shtok, S. Harary, R. Feris, R. Giryes, and A. M. Bronstein.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Alfassy_LaSO_Label-Set_Operations_Networks_for_Multi-Label_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/leokarlin/LaSO)

1. **Image Deformation Meta-Networks for One-Shot Learning,** in CVPR, 2019.
*Z. Chen, Y. Fu, Y.-X. Wang, L. Ma, W. Liu, and M. Hebert.*
[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Image_Deformation_Meta-Networks_for_One-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/tankche1/IDeMe-Net)

1. **Spot and Learn: A Maximum-Entropy Patch Sampler for Few-Shot Image Classification,** in CVPR, 2019.
*W.-H. Chu, Y.-J. Li, J.-C. Chang, and Y.-C. F. Wang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf)

1. **Adversarial Feature Hallucination Networks for Few-Shot Learning,** in CVPR, 2020.
*K. Li, Y. Zhang, K. Li, and Y. Fu.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Adversarial_Feature_Hallucination_Networks_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Instance Credibility Inference for Few-Shot Learning,** in CVPR, 2020.
*Y. Wang, C. Xu, C. Liu, L. Zhang, and Y. Fu.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Instance_Credibility_Inference_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Diversity Transfer Network for Few-Shot Learning,** in AAAI, 2020.
*M. Chen, Y. Fang, X. Wang, H. Luo, Y. Geng, X. Zhang, C. Huang, W. Liu, and B. Wang.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6628)
[code](https://github.com/Yuxin-CV/DTN)

1. **Neural Snowball for Few-Shot Relation Learning,** in AAAI, 2020.
*T. Gao, X. Han, R. Xie, Z. Liu, F. Lin, L. Lin, and M. Sun.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6281)
[code](https://github.com/thunlp/Neural-Snowball)

1. **Associative Alignment for Few-Shot Image Classification,** in ECCV, 2020.
*A. Afrasiyabi, J. Lalonde, and C. Gagné.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500018.pdf)
[code](https://github.com/ArmanAfrasiyabi/associative-alignment-fs)

1. **Information Maximization for Few-Shot Learning,** in NeurIPS, 2020.
*M. Boudiaf, I. Ziko, J. Rony, J. Dolz, P. Piantanida, and I. B. Ayed.*
[paper](https://proceedings.neurips.cc/paper/2020/file/196f5641aa9dc87067da4ff90fd81e7b-Paper.pdf)
[code](https://github.com/mboudiaf/TIM)

1. **Self-Training for Few-Shot Transfer Across Extreme Task Differences,** in ICLR, 2021.
*C. P. Phoo, and B. Hariharan.*
[paper](https://openreview.net/pdf?id=O3Y56aqpChA)

1. **Free Lunch for Few-Shot Learning: Distribution Calibration,** in ICLR, 2021.
*S. Yang, L. Liu, and M. Xu.*
[paper](https://openreview.net/pdf?id=JWOiYxMG92s)
[code](https://github.com/ShuoYang-1998/ICLR2021-Oral_Distribution_Calibration)

1. **Parameterless Transductive Feature Re-Representation for Few-Shot Learning,** in ICML, 2021.
*W. Cui, and Y. Guo;.*
[paper](http://proceedings.mlr.press/v139/cui21a/cui21a.pdf)

1. **Learning Intact Features by Erasing-Inpainting for Few-Shot Classification,** in AAAI, 2021.
*J. Li, Z. Wang, and X. Hu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17021/16828)

1. **Variational Feature Disentangling for Fine-Grained Few-Shot Classification,** in ICCV, 2021.
*J. Xu, H. Le, M. Huang, S. Athar, and D. Samaras.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Variational_Feature_Disentangling_for_Fine-Grained_Few-Shot_Classification_ICCV_2021_paper.pdf)

1. **Coarsely-Labeled Data for Better Few-Shot Transfer,** in ICCV, 2021.
*C. P. Phoo, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Phoo_Coarsely-Labeled_Data_for_Better_Few-Shot_Transfer_ICCV_2021_paper.pdf)

1. **Pseudo-Loss Confidence Metric for Semi-Supervised Few-Shot Learning,** in ICCV, 2021.
*K. Huang, J. Geng, W. Jiang, X. Deng, and Z. Xu.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Pseudo-Loss_Confidence_Metric_for_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Iterative Label Cleaning for Transductive and Semi-Supervised Few-Shot Learning,** in ICCV, 2021.
*M. Lazarou, T. Stathaki, and Y. Avrithis.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lazarou_Iterative_Label_Cleaning_for_Transductive_and_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Meta Two-Sample Testing: Learning Kernels for Testing With Limited Data,** in NeurIPS, 2021.
*F. Liu, W. Xu, J. Lu, and D. J. Sutherland.*
[paper](https://proceedings.neurips.cc/paper/2021/file/2e6d9c6052e99fcdfa61d9b9da273ca2-Paper.pdf)

1. **Dynamic Distillation Network for Cross-Domain Few-Shot Recognition With Unlabeled Data,** in NeurIPS, 2021.
*A. Islam, C.-F. Chen, R. Panda, L. Karlinsky, R. Feris, and R. Radke.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1d6408264d31d453d556c60fe7d0459e-Paper.pdf)

1. **Towards Better Understanding and Better Generalization of Low-Shot Classification in Histology Images With Contrastive Learning,** in ICLR, 2022.
*J. Yang, H. Chen, J. Yan, X. Chen, and J. Yao.*
[paper](https://openreview.net/pdf?id=kQ2SOflIOVC)
[code](https://github.com/TencentAILabHealthcare/Few-shot-WSI)

1. **FlipDA: Effective and Robust Data Augmentation for Few-Shot Learning,** in ACL, 2022.
*J. Zhou, Y. Zheng, J. Tang, L. Jian, and Z. Yang.*
[paper](https://aclanthology.org/2022.acl-long.592.pdf)
[code](https://github.com/zhouj8553/flipda)

1. **PromDA: Prompt-Based Data Augmentation for Low-Resource NLU Tasks,** in ACL, 2022.
*Y. Wang, C. Xu, Q. Sun, H. Hu, C. Tao, X. Geng, and D. Jiang.*
[paper](https://aclanthology.org/2022.acl-long.292.pdf)
[code](https://github.com/garyyufei/promda)

1. **N-Shot Learning for Augmenting Task-Oriented Dialogue State Tracking,** in Findings of ACL, 2022.
*I. T. Aksu, Z. Liu, M. Kan, and N. F. Chen.*
[paper](https://aclanthology.org/2022.findings-acl.131.pdf)

1. **Generating Representative Samples for Few-Shot Classification,** in CVPR, 2022.
*J. Xu, and H. Le.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Generating_Representative_Samples_for_Few-Shot_Classification_CVPR_2022_paper.pdf)
[code](https://github.com/cvlab-stonybrook/)

1. **Semi-Supervised Few-Shot Learning via Multi-Factor Clustering,** in CVPR, 2022.
*J. Ling, L. Liao, M. Yang, and J. Shuai.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ling_Semi-Supervised_Few-Shot_Learning_via_Multi-Factor_Clustering_CVPR_2022_paper.pdf)

1. **Information Augmentation for Few-Shot Node Classification,** in IJCAI, 2022.
*Z. Wu, P. Zhou, G. Wen, Y. Wan, J. Ma, D. Cheng, and X. Zhu.*
[paper](https://www.ijcai.org/proceedings/2022/0500.pdf)

1. **Improving Task-Specific Generalization in Few-Shot Learning via Adaptive Vicinal Risk Minimization,** in NeurIPS, 2022.
*L.-K. Huang, and Y. Wei.*
[paper](https://openreview.net/pdf?id=fHUBa3gQno)

1. **An Embarrassingly Simple Approach to Semi-Supervised Few-Shot Learning,** in NeurIPS, 2022.
*X.-S. Wei, H.-Y. Xu, F. Zhang, Y. Peng, and W. Zhou.*
[paper](https://openreview.net/pdf?id=-3Pg7QNIF1S)

1. **FeLMi : Few Shot Learning With Hard Mixup,** in NeurIPS, 2022.
*A. Roy, A. Shah, K. Shah, P. Dhar, A. Cherian, and R. Chellappa.*
[paper](https://openreview.net/pdf?id=xpdaDM_B4D)
[code](https://github.com/aniket004/Felmi)

1. **Understanding Cross-Domain Few-Shot Learning Based on Domain Similarity and Few-Shot Difficulty,** in NeurIPS, 2022.
*J. Oh, S. Kim, N. Ho, J.-H. Kim, H. Song, and S.-Y. Yun.*
[paper](https://openreview.net/pdf?id=rH-X09cB50f)
[code](https://github.com/sungnyun/understanding-cdfsl)

1. **Label Hallucination for Few-Shot Classification,** in AAAI, 2022.
*Y. Jian, and L. Torresani.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20659/20418)
[code](https://github.com/yiren-jian/LabelHalluc)

1. **STUNT: Few-Shot Tabular Learning With Self-Generated Tasks From Unlabeled Tables,** in ICLR, 2023.
*J. Nam, J. Tack, K. Lee, H. Lee, and J. Shin.*
[paper](https://openreview.net/pdf?id=_xlsjehDvlY)
[code](https://github.com/jaehyun513/STUNT)

1. **Unsupervised Meta-Learning via Few-Shot Pseudo-Supervised Contrastive Learning,** in ICLR, 2023.
*H. Jang, H. Lee, and J. Shin.*
[paper](https://openreview.net/pdf?id=TdTGGj7fYYJ)
[code](https://github.com/alinlab/PsCo)

1. **Progressive Mix-Up for Few-Shot Supervised Multi-Source Domain Transfer,** in ICLR, 2023.
*R. Zhu, R. Zhu, X. Yu, and S. Li.*
[paper](https://openreview.net/pdf?id=H7M_5K5qKJV)
[code](https://github.com/ronghangzhu/P-Mixup)

1. **Cross-Level Distillation and Feature Denoising for Cross-Domain Few-Shot Classification,** in ICLR, 2023.
*H. ZHENG, R. Wang, J. Liu, and A. Kanezaki.*
[paper](https://openreview.net/pdf?id=Kn-HA8DFik)
[code](https://gitee.com/mindspore/models/tree/master/research/cv/CLDFD)

1. **Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning,** in ICML, 2023.
*Y. Meng, M. Michalski, J. Huang, Y. Zhang, T. F. Abdelzaher, and J. Han.*
[paper](http://proceedings.mlr.press/v202/meng23b/meng23b.pdf)
[code](https://github.com/yumeng5/FewGen)

1. **Self-Evolution Learning for Mixup: Enhance Data Augmentation on Few-Shot Text Classification Tasks,** in EMNLP, 2023.
*H. Zheng, Q. Zhong, L. Ding, Z. Tian, X. Niu, C. Wang, D. Li, and D. Tao.*
[paper](https://aclanthology.org/2023.emnlp-main.555.pdf)

1. **Effective Data Augmentation With Diffusion Models,** in ICLR, 2024.
*B. Trabucco, K. Doherty, M. A. Gurinas, and R. Salakhutdinov.*
[paper](https://openreview.net/attachment?id=ZWzUA9zeAg&name=pdf)
[code](https://github.com/anonymous-da-fusion/da-fusion)

1. **Frozen Feature Augmentation for Few-Shot Image Classification,** in CVPR, 2024.
*A. Bär, N. Houlsby, M. Dehghani, and M. Kumar.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01519)

1. **Exploring Cross-Domain Few-Shot Classification via Frequency-Aware Prompting,** in IJCAI, 2024.
*T. Zhang, Q. Cai, F. Gao, L. Qi, and J. Dong.*
[paper](https://www.ijcai.org/proceedings/2024/607)

1. **FSL-Rectifier: Rectify Outliers in Few-Shot Learning via Test-Time Augmentation,** in AAAI, 2025.
*Y. Bai, Y. K. Tan, S. Chen, Y. Shu, and T. Chen.*
[paper](https://doi.org/10.1609/aaai.v39i15.33697)

## [Model](#content)

### Multitask Learning

1. **Multi-Task Transfer Methods to Improve One-Shot Learning for Multimedia Event Detection,** in BMVC, 2015.
*W. Yan, J. Yap, and G. Mori.*
[paper](http://www.bmva.org/bmvc/2015/papers/paper037/index.html)

1. **Label Efficient Learning of Transferable Representations Across Domains and Tasks,** in NeurIPS, 2017.
*Z. Luo, Y. Zou, J. Hoffman, and L. Fei-Fei.*
[paper](https://papers.nips.cc/paper/6621-label-efficient-learning-of-transferable-representations-acrosss-domains-and-tasks.pdf)

1. **Few-Shot Adversarial Domain Adaptation,** in NeurIPS, 2017.
*S. Motiian, Q. Jones, S. Iranmanesh, and G. Doretto.*
[paper](https://papers.nips.cc/paper/7244-few-shot-adversarial-domain-adaptation)

1. **One-Shot Unsupervised Cross Domain Translation,** in NeurIPS, 2018.
*S. Benaim and L. Wolf.* 
[paper](https://papers.nips.cc/paper/7480-one-shot-unsupervised-cross-domain-translation.pdf)

1. **Multi-Content GAN for Few-Shot Font Style Transfer,** in CVPR, 2018. 
*S. Azadi, M. Fisher, V. G. Kim, Z. Wang, E. Shechtman, and T. Darrell.*
[paper](http://www.vovakim.com/papers/18_CVPRSpotlight_FontDropper.pdf)
[code](https://github.com/azadis/MC-GAN)

1. **Feature Space Transfer for Data Augmentation,** in CVPR, 2018.
*B. Liu, X. Wang, M. Dixit, R. Kwitt, and N. Vasconcelos.* 
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Feature_Space_Transfer_CVPR_2018_paper.pdf)

1. **Fine-Grained Visual Categorization Using Meta-Learning Optimization With Sample Selection of Auxiliary Data,** in ECCV, 2018.
*Y. Zhang, H. Tang, and K. Jia.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yabin_Zhang_Fine-Grained_Visual_Categorization_ECCV_2018_paper.pdf)

1. **Few-Shot Charge Prediction With Discriminative Legal Attributes,** in COLING, 2018.
*Z. Hu, X. Li, C. Tu, Z. Liu, and M. Sun.*
[paper](https://www.aclweb.org/anthology/C18-1041.pdf)

1. **Boosting Few-Shot Visual Learning With Self-Supervision,** in ICCV, 2019.
*S. Gidaris, A. Bursuc, N. Komodakis, P. Pérez, and M. Cord.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gidaris_Boosting_Few-Shot_Visual_Learning_With_Self-Supervision_ICCV_2019_paper.pdf)

1. **When Does Self-Supervision Improve Few-Shot Learning?,** in ECCV, 2020.
*J. Su, S. Maji, and B. Hariharan.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520630.pdf)

1. **Pareto Self-Supervised Training for Few-Shot Learning,** in CVPR, 2021.
*Z. Chen, J. Ge, H. Zhan, S. Huang, and D. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pareto_Self-Supervised_Training_for_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Bridging Multi-Task Learning and Meta-Learning: Towards Efficient Training and Effective Adaptation,** in ICML, 2021.
*H. Wang, H. Zhao, and B. Li;.*
[paper](http://proceedings.mlr.press/v139/wang21ad/wang21ad.pdf)
[code](https://github.com/AI-secure/multi-task-learning)

1. **Task-Level Self-Supervision for Cross-Domain Few-Shot Learning,** in AAAI, 2022.
*W. Yuan, Z. Zhang, C. Wang, H. Song, Y. Xie, and L. Ma.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20230/19989)

1. **Improving Few-Shot Generalization by Exploring and Exploiting Auxiliary Data,** in NeurIPS, 2023.
*A. Albalak, C. Raffel, and W. Y. Wang.*
[paper](https://openreview.net/attachment?id=JDnLXc4NOn&name=pdf)
[code](https://github.com/alon-albalak/FLAD)


### Embedding/Metric Learning

1. **Object Classification From a Single Example Utilizing Class Relevance Metrics,** in NeurIPS, 2005.
*M. Fink.*
[paper](https://papers.nips.cc/paper/2576-object-classification-from-a-single-example-utilizing-class-relevance-metrics.pdf)

1. **Optimizing One-Shot Recognition With Micro-Set Learning,** in CVPR, 2010.
*K. D. Tang, M. F. Tappen, R. Sukthankar, and C. H. Lampert.*
[paper](http://www.cs.ucf.edu/~mtappen/pubs/cvpr10_oneshot.pdf)

1. **Siamese neural networks for one-shot image recognition,** ICML deep learning workshop, 2015.
*G. Koch, R. Zemel, and R. Salakhutdinov.*
[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

1. **Matching Networks for One Shot Learning,** in NeurIPS, 2016.
*O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al.* 
[paper](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)

1. **Learning Feed-Forward One-Shot Learners,** in NeurIPS, 2016.
*L. Bertinetto, J. F. Henriques, J. Valmadre, P. Torr, and A. Vedaldi.*
[paper](https://papers.nips.cc/paper/6068-learning-feed-forward-one-shot-learners.pdf)

1. **Few-Shot Learning Through an Information Retrieval Lens,** in NeurIPS, 2017.
*E. Triantafillou, R. Zemel, and R. Urtasun.*
[paper](https://papers.nips.cc/paper/6820-few-shot-learning-through-an-information-retrieval-lens.pdf)

1. **Prototypical Networks for Few-Shot Learning,** in NeurIPS, 2017.
*J. Snell, K. Swersky, and R. S. Zemel.*
[paper](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)
[code](https://github.com/jakesnell/prototypical-networks)

1. **Attentive Recurrent Comparators,** in ICML, 2017.
*P. Shyam, S. Gupta, and A. Dukkipati.*
[paper](http://proceedings.mlr.press/v70/shyam17a/shyam17a.pdf)

1. **Learning Algorithms for Active Learning,** in ICML, 2017.
*P. Bachman, A. Sordoni, and A. Trischler.*
[paper](http://proceedings.mlr.press/v70/bachman17a.pdf)

1. **Structured Set Matching Networks for One-Shot Part Labeling,** in CVPR, 2018.
*J. Choi, J. Krishnamurthy, A. Kembhavi, and A. Farhadi.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Structured_Set_Matching_CVPR_2018_paper.pdf)

1. **Low-Shot Learning From Imaginary Data,** in CVPR, 2018.
*Y.-X. Wang, R. Girshick, M. Hebert, and B. Hariharan.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Low-Shot_Learning_From_CVPR_2018_paper.pdf)

1. **Learning to Compare: Relation Network for Few-Shot Learning,** in CVPR, 2018.
*F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. Torr, and T. M. Hospedales.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf)
[code](https://github.com/floodsung/LearningToCompare_FSL)

1. **Dynamic Conditional Networks for Few-Shot Learning,** in ECCV, 2018.
*F. Zhao, J. Zhao, S. Yan, and J. Feng.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fang_Zhao_Dynamic_Conditional_Networks_ECCV_2018_paper.pdf)
[code](https://github.com/ZhaoJ9014/Dynamic-Conditional-Networks.PyTorch)

1. **TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning,** in NeurIPS, 2018.
*B. Oreshkin, P. R. López, and A. Lacoste.*
[paper](https://papers.nips.cc/paper/7352-tadam-task-dependent-adaptive-metric-for-improved-few-shot-learning.pdf)

1. **Meta-Learning for Semi-Supervised Few-Shot Classification,** in ICLR, 2018.
*M. Ren, S. Ravi, E. Triantafillou, J. Snell, K. Swersky, J. B. Tenen- baum, H. Larochelle, and R. S. Zemel.* 
[paper](https://openreview.net/forum?id=r1n5Osurf)
[code](https://github.com/renmengye/few-shot-ssl-public)

1. **Few-Shot Learning With Graph Neural Networks,** in ICLR, 2018.
*V. G. Satorras and J. B. Estrach.*
[paper](https://openreview.net/pdf?id=HJcSzz-CZ)
[code](https://github.com/vgsatorras/few-shot-gnn)

1. **A Simple Neural Attentive Meta-Learner,** in ICLR, 2018.
*N. Mishra, M. Rohaninejad, X. Chen, and P. Abbeel.*
[paper](https://openreview.net/forum?id=B1DmUzWAW)

1. **Meta-Learning With Differentiable Closed-Form Solvers,** in ICLR, 2019.
*L. Bertinetto, J. F. Henriques, P. Torr, and A. Vedaldi.* 
[paper](https://openreview.net/forum?id=HyxnZh0ct7)

1. **Learning to Propagate Labels: Transductive Propagation Network for Few-Shot Learning,** in ICLR, 2019.
*Y. Liu, J. Lee, M. Park, S. Kim, E. Yang, S. Hwang, and Y. Yang.*
[paper](https://openreview.net/forum?id=SyVuRiC5K7)
[code](https://github.com/csyanbin/TPN-pytorch)

1. **Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification,** in ACL, 2019.
*Z.-X. Ye, and Z.-H. Ling.*
[paper](https://www.aclweb.org/anthology/P19-1277.pdf)

1. **Induction Networks for Few-Shot Text Classification,** in EMNLP-IJCNLP, 2019.
*R. Geng, B. Li, Y. Li, X. Zhu, P. Jian, and J. Sun.*
[paper](https://www.aclweb.org/anthology/D19-1403.pdf)

1. **Hierarchical Attention Prototypical Networks for Few-Shot Text Classification,** in EMNLP-IJCNLP, 2019.
*S. Sun, Q. Sun, K. Zhou, and T. Lv.*
[paper](https://www.aclweb.org/anthology/D19-1045.pdf)

1. **Cross Attention Network for Few-Shot Classification,** in NeurIPS, 2019.
*R. Hou, H. Chang, B. Ma, S. Shan, and X. Chen.*
[paper](https://papers.nips.cc/paper/8655-cross-attention-network-for-few-shot-classification.pdf)

1. **Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes,** in NeurIPS, 2019.
*J. Requeima, J. Gordon, J. Bronskill, S. Nowozin, and R. E. Turner.*
[paper](https://proceedings.neurips.cc/paper/2019/file/1138d90ef0a0848a542e57d1595f58ea-Paper.pdf)
[code](https://github.com/cambridge-mlg/cnaps)

1. **Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification,** in AAAI, 2019.
*T. Gao, X. Han, Z. Liu, and M. Sun.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4604/4482)
[code](https://github.com/thunlp/HATT-Proto)

1. **Attention-Based Multi-Context Guiding for Few-Shot Semantic Segmentation,** in AAAI, 2019.
*T. Hu, P. Yang, C. Zhang, G. Yu, Y. Mu and C. G. M. Snoek.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4604/4482)

1. **Distribution Consistency Based Covariance Metric Networks for Few-Shot Learning,** in AAAI, 2019.
*W. Li, L. Wang, J. Xu, J. Huo, Y. Gao and J. Luo.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4885/4758)

1. **A Dual Attention Network With Semantic Embedding for Few-Shot Learning,** in AAAI, 2019.
*S. Yan, S. Zhang, and X. He.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4940/4813)

1. **TapNet: Neural Network Augmented With Task-Adaptive Projection for Few-Shot Learning,** in ICML, 2019.
*S. W. Yoon, J. Seo, and J. Moon.*
[paper](http://proceedings.mlr.press/v97/yoon19a/yoon19a.pdf)

1. **Prototype Propagation Networks (PPN) for Weakly-Supervised Few-Shot Learning on Category Graph,** in IJCAI, 2019.
*L. Liu, T. Zhou, G. Long, J. Jiang, L. Yao, C. Zhang.*
[paper](https://www.ijcai.org/Proceedings/2019/0418.pdf)
[code](https://github.com/liulu112601/Prototype-Propagation-Net)

1. **Collect and Select: Semantic Alignment Metric Learning for Few-Shot Learning,** in ICCV, 2019.
*F. Hao, F. He, J. Cheng, L. Wang, J. Cao, and D. Tao.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hao_Collect_and_Select_Semantic_Alignment_Metric_Learning_for_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **Transductive Episodic-Wise Adaptive Metric for Few-Shot Learning,** in ICCV, 2019.
*L. Qiao, Y. Shi, J. Li, Y. Wang, T. Huang, and Y. Tian.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qiao_Transductive_Episodic-Wise_Adaptive_Metric_for_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **Few-Shot Learning With Embedded Class Models and Shot-Free Meta Training,** in ICCV, 2019.
*A. Ravichandran, R. Bhotika, and S. Soatto.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ravichandran_Few-Shot_Learning_With_Embedded_Class_Models_and_Shot-Free_Meta_Training_ICCV_2019_paper.pdf)

1. **PARN: Position-Aware Relation Networks for Few-Shot Learning,** in ICCV, 2019.
*Z. Wu, Y. Li, L. Guo, and K. Jia.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_PARN_Position-Aware_Relation_Networks_for_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **PANet: Few-Shot Image Semantic Segmentation With Prototype Alignment,** in ICCV, 2019.
*K. Wang, J. H. Liew, Y. Zou, D. Zhou, and J. Feng.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf)
[code](https://github.com/kaixin96/PANet)

1. **RepMet: Representative-Based Metric Learning for Classification and Few-Shot Object Detection,** in CVPR, 2019.
*L. Karlinsky, J. Shtok, S. Harary, E. Schwartz, A. Aides, R. Feris, R. Giryes, and A. M. Bronstein.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karlinsky_RepMet_Representative-Based_Metric_Learning_for_Classification_and_Few-Shot_Object_Detection_CVPR_2019_paper.pdf)
[code](https://github.com/jshtok/RepMet)

1. **Edge-Labeling Graph Neural Network for Few-Shot Learning,** in CVPR, 2019.
*J. Kim, T. Kim, S. Kim, and C. D. Yoo.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Edge-Labeling_Graph_Neural_Network_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

1. **Finding Task-Relevant Features for Few-Shot Learning by Category Traversal,** in CVPR, 2019.
*H. Li, D. Eigen, S. Dodge, M. Zeiler, and X. Wang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Finding_Task-Relevant_Features_for_Few-Shot_Learning_by_Category_Traversal_CVPR_2019_paper.pdf)
[code](https://github.com/Clarifai/few-shot-ctm)

1. **Revisiting Local Descriptor Based Image-to-Class Measure for Few-Shot Learning,** in CVPR, 2019.
*W. Li, L. Wang, J. Xu, J. Huo, Y. Gao, and J. Luo.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/WenbinLee/DN4)

1. **TAFE-Net: Task-Aware Feature Embeddings for Low Shot Learning,** in CVPR, 2019.
*X. Wang, F. Yu, R. Wang, T. Darrell, and J. E. Gonzalez.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_TAFE-Net_Task-Aware_Feature_Embeddings_for_Low_Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/ucbdrive/tafe-net)

1. **Improved Few-Shot Visual Classification,** in CVPR, 2020.
*P. Bateni, R. Goyal, V. Masrani, F. Wood, and L. Sigal.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.pdf)

1. **Boosting Few-Shot Learning With Adaptive Margin Loss,** in CVPR, 2020.
*A. Li, W. Huang, X. Lan, J. Feng, Z. Li, and L. Wang.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Boosting_Few-Shot_Learning_With_Adaptive_Margin_Loss_CVPR_2020_paper.pdf)

1. **Adaptive Subspaces for Few-Shot Learning,** in CVPR, 2020.
*C. Simon, P. Koniusz, R. Nock, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **DPGN: Distribution Propagation Graph Network for Few-Shot Learning,** in CVPR, 2020.
*L. Yang, L. Li, Z. Zhang, X. Zhou, E. Zhou, and Y. Liu.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_DPGN_Distribution_Propagation_Graph_Network_for_Few-Shot_Learning_CVPR_2020_paper_check.pdf)

1. **Few-Shot Learning via Embedding Adaptation With Set-to-Set Functions,** in CVPR, 2020.
*H.-J. Ye, H. Hu, D.-C. Zhan, and F. Sha.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf)
[code](https://github.com/Sha-Lab/FEAT)

1. **DeepEMD: Few-Shot Image Classification With Differentiable Earth Mover's Distance and Structured Classifiers,** in CVPR, 2020.
*C. Zhang, Y. Cai, G. Lin, and C. Shen.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.pdf)
[code](https://github.com/icoz69/DeepEMD)

1. **Few-Shot Text Classification With Distributional Signatures,** in ICLR, 2020.
*Y. Bao, M. Wu, S. Chang, and R. Barzilay.*
[paper](https://openreview.net/pdf?id=H1emfT4twB)
[code](https://github.com/YujiaBao/Distributional-Signatures)

1. **Learning Task-Aware Local Representations for Few-Shot Learning,** in IJCAI, 2020.
*C. Dong, W. Li, J. Huo, Z. Gu, and Y. Gao.*
[paper](https://www.ijcai.org/Proceedings/2020/0100.pdf)

1. **SimPropNet: Improved Similarity Propagation for Few-Shot Image Segmentation,** in IJCAI, 2020.
*S. Gairola, M. Hemani, A. Chopra, and B. Krishnamurthy.*
[paper](https://www.ijcai.org/Proceedings/2020/0080.pdf)

1. **Asymmetric Distribution Measure for Few-Shot Learning,** in IJCAI, 2020.
*W. Li, L. Wang, J. Huo, Y. Shi, Y. Gao, and J. Luo.*
[paper](https://www.ijcai.org/Proceedings/2020/0409.pdf)

1. **Transductive Relation-Propagation Network for Few-Shot Learning,** in IJCAI, 2020.
*Y. Ma, S. Bai, S. An, W. Liu, A. Liu, X. Zhen, and X. Liu.*
[paper](https://www.ijcai.org/Proceedings/2020/0112.pdf)

1. **Weakly Supervised Few-Shot Object Segmentation Using Co-Attention With Visual and Semantic Embeddings,** in IJCAI, 2020.
*M. Siam, N. Doraiswamy, B. N. Oreshkin, H. Yao, and M. Jägersand.*
[paper](https://www.ijcai.org/Proceedings/2020/0120.pdf)

1. **Few-Shot Learning on Graphs via Super-Classes Based on Graph Spectral Measures,** in ICLR, 2020.
*J. Chauhan, D. Nathani, and M. Kaul.*
[paper](https://openreview.net/pdf?id=Bkeeca4Kvr)

1. **SGAP-Net: Semantic-Guided Attentive Prototypes Network for Few-Shot Human-Object Interaction Recognition,** in AAAI, 2020.
*Z. Ji, X. Liu, Y. Pang, and X. Li.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6764)

1. **One-Shot Image Classification by Learning to Restore Prototypes,** in AAAI, 2020.
*W. Xue, and W. Wang.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6130)

1. **Negative Margin Matters: Understanding Margin in Few-Shot Classification,** in ECCV, 2020.
*B. Liu, Y. Cao, Y. Lin, Q. Li, Z. Zhang, M. Long, and H. Hu.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490426.pdf)
[code](https://github.com/bl0/negative-margin.few-shot)

1. **Prototype Rectification for Few-Shot Learning,** in ECCV, 2020.
*J. Liu, L. Song, and Y. Qin.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460715.pdf)

1. **Rethinking Few-Shot Image Classification: A Good Embedding Is All You Need?,** in ECCV, 2020.
*Y. Tian, Y. Wang, D. Krishnan, J. B. Tenenbaum, and P. Isola.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590256.pdf)
[code](https://github.com/WangYueFt/rfs/)

1. **SEN: A Novel Feature Normalization Dissimilarity Measure for Prototypical Few-Shot Learning Networks,** in ECCV, 2020.
*V. N. Nguyen, S. Løkse, K. Wickstrøm, M. Kampffmeyer, D. Roverso, and R. Jenssen.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680120.pdf)

1. **TAFSSL: Task-Adaptive Feature Sub-Space Learning for Few-Shot Classification,** in ECCV, 2020.
*M. Lichtenstein, P. Sattigeri, R. Feris, R. Giryes, and L. Karlinsky.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520511.pdf)

1. **Attentive Prototype Few-Shot Learning With Capsule Network-Based Embedding,** in ECCV, 2020.
*F. Wu, J. S.Smith, W. Lu, C. Pang, and B. Zhang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730239.pdf)

1. **Embedding Propagation: Smoother Manifold for Few-Shot Classification,** in ECCV, 2020.
*P. Rodríguez, I. Laradji, A. Drouin, and A. Lacoste.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710120.pdf)
[code](https://github.com/ElementAI/embedding-propagation)

1. **Laplacian Regularized Few-Shot Learning,** in ICML, 2020.
*I. M. Ziko, J. Dolz, E. Granger, and I. B. Ayed.*
[paper](http://proceedings.mlr.press/v119/ziko20a/ziko20a.pdf)
[code](https://github.com/imtiazziko/LaplacianShot)

1. **TAdaNet: Task-Adaptive Network for Graph-Enriched Meta-Learning,** in KDD, 2020.
*Q. Suo, i. Chou, W. Zhong, and A. Zhang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403230)

1. **Concept Learners for Few-Shot Learning,** in ICLR, 2021.
*K. Cao, M. Brbic, and J. Leskovec.*
[paper](https://openreview.net/pdf?id=eJIJF3-LoZO)

1. **Reinforced Attention for Few-Shot Learning and Beyond,** in CVPR, 2021.
*J. Hong, P. Fang, W. Li, T. Zhang, C. Simon, M. Harandi, and L. Petersson.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Reinforced_Attention_for_Few-Shot_Learning_and_Beyond_CVPR_2021_paper.pdf)

1. **Mutual CRF-GNN for Few-Shot Learning,** in CVPR, 2021.
*S. Tang, D. Chen, L. Bai, K. Liu, Y. Ge, and W. Ouyang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Mutual_CRF-GNN_for_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Few-Shot Classification With Feature Map Reconstruction Networks,** in CVPR, 2021.
*D. Wertheimer, L. Tang, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.pdf)
[code](https://github.com/Tsingularity/FRN)

1. **ECKPN: Explicit Class Knowledge Propagation Network for Transductive Few-Shot Learning,** in CVPR, 2021.
*C. Chen, X. Yang, C. Xu, X. Huang, and Z. Ma.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_ECKPN_Explicit_Class_Knowledge_Propagation_Network_for_Transductive_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning,** in CVPR, 2021.
*M. N. Rizve, S. Khan, F. S. Khan, and M. Shah.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Rizve_Exploring_Complementary_Strengths_of_Invariant_and_Equivariant_Representations_for_Few-Shot_CVPR_2021_paper.pdf)

1. **Rethinking Class Relations: Absolute-Relative Supervised and Unsupervised Few-Shot Learning,** in CVPR, 2021.
*H. Zhang, P. Koniusz, S. Jian, H. Li, and P. H. S. Torr.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Rethinking_Class_Relations_Absolute-Relative_Supervised_and_Unsupervised_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Unsupervised Embedding Adaptation via Early-Stage Feature Reconstruction for Few-Shot Classification,** in ICML, 2021.
*D. H. Lee, and S. Chung.*
[paper](http://proceedings.mlr.press/v139/lee21d/lee21d.pdf)
[code](https://github.com/movinghoon/ESFR)

1. **Learning a Few-Shot Embedding Model With Contrastive Learning,** in AAAI, 2021.
*C. Liu, Y. Fu, C. Xu, S. Yang,  J. Li, C. Wang, and L. Zhang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17047/16854)

1. **Looking Wider for Better Adaptive Representation in Few-Shot Learning,** in AAAI, 2021.
*J. Zhao, Y. Yang, X. Lin, J. Yang, and L. He.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17311/17118)

1. **Tailoring Embedding Function to Heterogeneous Few-Shot Tasks by Global and Local Feature Adaptors,** in AAAI, 2021.
*S. Lu, H. Ye, and D.-C. Zhan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17063/16870)

1. **Knowledge Guided Metric Learning for Few-Shot Text Classification,** in NAACL-HLT, 2021.
*D. Sui, Y. Chen, B. Mao, D. Qiu, K. Liu, and J. Zhao.*
[paper](https://aclanthology.org/2021.naacl-main.261.pdf)

1. **Mixture-Based Feature Space Learning for Few-Shot Image Classification,** in ICCV, 2021.
*A. Afrasiyabi, J. Lalonde, and C. Gagné.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Afrasiyabi_Mixture-Based_Feature_Space_Learning_for_Few-Shot_Image_Classification_ICCV_2021_paper.pdf)

1. **Z-Score Normalization, Hubness, and Few-Shot Learning,** in ICCV, 2021.
*N. Fei, Y. Gao, Z. Lu, and T. Xiang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Fei_Z-Score_Normalization_Hubness_and_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Relational Embedding for Few-Shot Classification,** in ICCV, 2021.
*D. Kang, H. Kwon, J. Min, and M. Cho.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kang_Relational_Embedding_for_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/dahyun-kang/renet)

1. **Transductive Few-Shot Classification on the Oblique Manifold,** in ICCV, 2021.
*G. Qi, H. Yu, Z. Lu, and S. Li.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Qi_Transductive_Few-Shot_Classification_on_the_Oblique_Manifold_ICCV_2021_paper.pdf)
[code](https://github.com/GuodongQi/FSL-OM)

1. **Curvature Generation in Curved Spaces for Few-Shot Learning,** in ICCV, 2021.
*Z. Gao, Y. Wu, Y. Jia, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Curvature_Generation_in_Curved_Spaces_for_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **On Episodes, Prototypical Networks, and Few-Shot Learning,** in NeurIPS, 2021.
*S. Laenen, and L. Bertinetto.*
[paper](https://proceedings.neurips.cc/paper/2021/file/cdfa4c42f465a5a66871587c69fcfa34-Paper.pdf)

1. **Few-Shot Learning as Cluster-Induced Voronoi Diagrams: A Geometric Approach,** in ICLR, 2022.
*C. Ma, Z. Huang, M. Gao, and J. Xu.*
[paper](https://openreview.net/pdf?id=6kCiVaoQdx9)
[code](https://github.com/horsepurve/DeepVoro)

1. **Few-Shot Learning With Siamese Networks and Label Tuning,** in ACL, 2022.
*T. Müller, G. Pérez-Torró, and M. Franco-Salvador.*
[paper](https://aclanthology.org/2022.acl-long.584.pdf)
[code](https://github.com/symanto-research/few-shot-learning-label-tuning)

1. **Learning to Affiliate: Mutual Centralized Learning for Few-Shot Classification,** in CVPR, 2022.
*Y. Liu, W. Zhang, C. Xiang, T. Zheng, D. Cai, and X. He.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Learning_To_Affiliate_Mutual_Centralized_Learning_for_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **Matching Feature Sets for Few-Shot Image Classification,** in CVPR, 2022.
*A. Afrasiyabi, H. Larochelle, J. Lalonde, and C. Gagné.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Afrasiyabi_Matching_Feature_Sets_for_Few-Shot_Image_Classification_CVPR_2022_paper.pdf)
[code](https://lvsn.github.io/SetFeat/)

1. **Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification,** in CVPR, 2022.
*J. Xie, F. Long, J. Lv, Q. Wang, and P. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Joint_Distribution_Matters_Deep_Brownian_Distance_Covariance_for_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **CAD: Co-Adapting Discriminative Features for Improved Few-Shot Classification,** in CVPR, 2022.
*P. Chikontwe, S. Kim, and S. H. Park.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chikontwe_CAD_Co-Adapting_Discriminative_Features_for_Improved_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **Ranking Distance Calibration for Cross-Domain Few-Shot Learning,** in CVPR, 2022.
*P. Li, S. Gong, C. Wang, and Y. Fu.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Ranking_Distance_Calibration_for_Cross-Domain_Few-Shot_Learning_CVPR_2022_paper.pdf)

1. **EASE: Unsupervised Discriminant Subspace Learning for Transductive Few-Shot Learning,** in CVPR, 2022.
*H. Zhu, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_EASE_Unsupervised_Discriminant_Subspace_Learning_for_Transductive_Few-Shot_Learning_CVPR_2022_paper.pdf)
[code](https://github.com/allenhaozhu/EASE)

1. **Cross-Domain Few-Shot Learning With Task-Specific Adapters,** in CVPR, 2022.
*W. Li, X. Liu, and H. Bilen.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Few-Shot_Learning_With_Task-Specific_Adapters_CVPR_2022_paper.pdf)
[code](https://github.com/VICO-UoE/URL)

1. **Hyperbolic Knowledge Transfer With Class Hierarchy for Few-Shot Learning,** in IJCAI, 2022.
*B. Zhang, H. Jiang, S. Feng, X. Li, Y. Ye, and R. Ye.*
[paper](https://www.ijcai.org/proceedings/2022/0517.pdf)

1. **Better Embedding and More Shots for Few-Shot Learning,** in IJCAI, 2022.
*Z. Chi, Z. Wang, M. Yang, W. Guo, and X. Xu.*
[paper](https://www.ijcai.org/proceedings/2022/0398.pdf)

1. **A Closer Look at Prototype Classifier for Few-Shot Image Classification,** in NeurIPS, 2022.
*M. Hou, and I. Sato.*
[paper](https://openreview.net/pdf?id=U_hOegGGglw)

1. **Rethinking Generalization in Few-Shot Classification,** in NeurIPS, 2022.
*M. Hiller, R. Ma, M. Harandi, and T. Drummond.*
[paper](https://openreview.net/pdf?id=p_g2nHlMus)
[code](https://github.com/mrkshllr/FewTURE)

1. **DMN4: Few-Shot Learning via Discriminative Mutual Nearest Neighbor Neural Network,** in AAAI, 2022.
*Y. Liu, T. Zheng, J. Song, D. Cai, and X. He.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20076/19835)

1. **Hybrid Graph Neural Networks for Few-Shot Learning,** in AAAI, 2022.
*T. Yu, S. He, Y.-Z. Song, and T. Xiang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20226/19985)
[code](https://github.com/TianyuanYu/HGNN)

1. **Adaptive Poincaré Point to Set Distance for Few-Shot Classification,** in AAAI, 2022.
*R. Ma, P. Fang, T. Drummond, and M. Harandi.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20087/19846)

1. **Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-Shot Learning With Hyperspherical Embeddings,** in CVPR, 2023.
*D. J. Trosten, R. Chakraborty, S. Løkse, K. K. Wickstrøm, R. Jenssen, and M. C. Kampffmeyer.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Trosten_Hubs_and_Hyperspheres_Reducing_Hubness_and_Improving_Transductive_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/uitml/noHub)

1. **Revisiting Prototypical Network for Cross Domain Few-Shot Learning,** in CVPR, 2023.
*F. Zhou, P. Wang, L. Zhang, W. Wei, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Revisiting_Prototypical_Network_for_Cross_Domain_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/NWPUZhoufei/LDP-Net)

1. **Transductive Few-Shot Learning With Prototype-Based Label Propagation by Iterative Graph Refinement,** in CVPR, 2023.
*H. Zhu, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Transductive_Few-Shot_Learning_With_Prototype-Based_Label_Propagation_by_Iterative_Graph_CVPR_2023_paper.pdf)
[code](https://github.com/allenhaozhu/protoLP)

1. **Few-Shot Classification via Ensemble Learning With Multi-Order Statistics,** in IJCAI, 2023.
*S. Yang, F. Liu, D. Chen, and J. Zhou.*
[paper](https://www.ijcai.org/proceedings/2023/0181.pdf)

1. **Few-Sample Feature Selection via Feature Manifold Learning,** in ICML, 2023.
*D. Cohen, T. Shnitzer, Y. Kluger, and R. Talmon.*
[paper](https://proceedings.mlr.press/v202/cohen23b/cohen23b.pdf)
[code](https://github.com/DavidCohen2/ManiFeSt)

1. **Interval Bound Interpolation for Few-Shot Learning With Few Tasks,** in ICML, 2023.
*S. Datta, S. S. Mullick, A. Chakrabarty, and S. Das.*
[paper](https://proceedings.mlr.press/v202/datta23a/datta23a.pdf)
[code](https://github.com/SankhaSubhra/maml-ibp-ibi)

1. **A Closer Look at Few-Shot Classification Again,** in ICML, 2023.
*X. Luo, H. Wu, J. Zhang, L. Gao, J. Xu, and J. Song.*
[paper](https://proceedings.mlr.press/v202/luo23e/luo23e.pdf)
[code](https://github.com/Frankluox/CloserLookAgainFewShot)

1. **TART: Improved Few-Shot Text Classification Using Task-Adaptive Reference Transformation,** in ACL, 2023.
*S. Lei, X. Zhang, J. He, F. Chen, and C.-T. Lu.*
[paper](https://aclanthology.org/2023.acl-long.617.pdf)
[code](https://github.com/slei109/TART)

1. **Class-Aware Patch Embedding Adaptation for Few-Shot Image Classification,** in ICCV, 2023.
*F. Hao, F. He, L. Liu, F. Wu, D. Tao, and J. Cheng.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Hao_Class-Aware_Patch_Embedding_Adaptation_for_Few-Shot_Image_Classification_ICCV_2023_paper.pdf)
[code](https://github.com/FushengHao/CPEA)

1. **Frequency Guidance Matters in Few-Shot Learning,** in ICCV, 2023.
*H. Cheng, S. Yang, J. T. Zhou, L. Guo, and B. Wen.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_Frequency_Guidance_Matters_in_Few-Shot_Learning_ICCV_2023_paper.pdf)

1. **Prototypes-Oriented Transductive Few-Shot Learning With Conditional Transport,** in ICCV, 2023.
*L. Tian, J. Feng, X. Chai, W. Chen, L. Wang, X. Liu, and B. Chen.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Tian_Prototypes-oriented_Transductive_Few-shot_Learning_with_Conditional_Transport_ICCV_2023_paper.pdf)
[code](https://github.com/RashLog/PUTM)

1. **Understanding Few-Shot Learning: Measuring Task Relatedness and Adaptation Difficulty via Attributes,** in NeurIPS, 2023.
*M. Hu, H. Chang, Z. Guo, B. Ma, S. Shan, and X. Chen.*
[paper](https://openreview.net/attachment?id=Pvgxecj5aS&name=pdf)
[code](https://github.com/hu-my/TaskAttributeDistance)

1. **DiffKendall: A Novel Approach for Few-Shot Learning With Differentiable Kendall's Rank Correlation,** in NeurIPS, 2023.
*K. Zheng, H. Zhang, and W. Huang.*
[paper](https://openreview.net/attachment?id=SVUQX1W7RL&name=pdf)

1. **Compositional Prototypical Networks for Few-Shot Classification,** in AAAI, 2023.
*Q. Lyu, and W. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26082/25854)
[code](https://github.com/fikry102/CPN)

1. **FoPro: Few-Shot Guided Robust Webly-Supervised Prototypical Learning,** in AAAI, 2023.
*Y. Qin, X. Chen, C. Chen, Y. Shen, B. Ren, Y. Gu, J. Yang, and C. Shen.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25303/25075)
[code](https://github.com/yuleiqin/fopro)

1. **SpatialFormer: Semantic and Target Aware Attentions for Few-Shot Learning,** in AAAI, 2023.
*J. Lai, S. Yang, W. Wu, T. Wu, G. Jiang, X. Wang, J. Liu, B.-B. Gao, W. Zhang, Y. Xie, and C. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26016/25788)

1. **RankDNN: Learning to Rank for Few-Shot Learning,** in AAAI, 2023.
*Q. Guo, H. Gong, X. Wei, Y. Fu, Y. Yu, W. Zhang, and W. Ge.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25150/24922)
[code](https://github.com/guoqianyu-alberta/RankDNN)

1. **Boosting Few-Shot Text Classification via Distribution Estimation,** in AAAI, 2023.
*H. Liu, F. Zhang, X. Zhang, S. Zhao, F. Ma, X.-M. Wu, H. Chen, H. Yu, and X. Zhang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26552/26324)

1. **Feature Distribution Fitting With Direction-Driven Weighting for Few-Shot Images Classification,** in AAAI, 2023.
*X. Wei, W. Du, H. Wan, and W. Min.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26228/26000)

1. **Exploring Tuning Characteristics of Ventral Stream's Neurons for Few-Shot Image Classification,** in AAAI, 2023.
*L. Dong, W. Zhai, and Z.-J. Zha.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25128/24900)

1. **RAPL: A Relation-Aware Prototype Learning Approach for Few-Shot Document-Level Relation Extraction,** in EMNLP, 2023.
*S. Meng, X. Hu, A. Liu, S. Li, F. Ma, Y. Yang, and L. Wen.*
[paper](https://aclanthology.org/2023.emnlp-main.316.pdf)
[code](https://github.com/THU-BPM/RAPL)

1. **BECLR: Batch Enhanced Contrastive Few-Shot Learning,** in ICLR, 2024.
*S. Poulakakis-Daktylidis, and H. Jamali-Rad.*
[paper](https://openreview.net/attachment?id=k9SVcrmXL8&name=pdf)
[code](https://github.com/stypoumic/BECLR)

1. **Adversarially Robust Few-shot Learning via Parameter Co-distillation of Similarity and Class Concept Learners,** in CVPR, 2024.
*J. Dong, P. Koniusz, J. Chen, X. Xie, and Y.-S. Ong.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02696)

1. **With a Little Help from Language: Semantic Enhanced Visual Prototype Framework for Few-Shot Learning,** in IJCAI, 2024.
*H. Cai, Y. Liu, S. Huang, and J. Lv.*
[paper](https://www.ijcai.org/proceedings/2024/415)

1. **A Density-driven Iterative Prototype Optimization for Transductive Few-shot Learning,** in IJCAI, 2024.
*J. Li, C. Ye, F. Wang, and J. Pan.*
[paper](https://www.ijcai.org/proceedings/2024/488)



### Learning with External Memory

1. **Meta-Learning With Memory-Augmented Neural Networks,** in ICML, 2016.
*A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap.*
[paper](http://proceedings.mlr.press/v48/santoro16.pdf)

1. **Few-Shot Object Recognition From Machine-Labeled Web Images,** in CVPR, 2017.
*Z. Xu, L. Zhu, and Y. Yang.*
[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Few-Shot_Object_Recognition_CVPR_2017_paper.pdf)

1. **Learning to Remember Rare Events,** in ICLR, 2017.
*Ł. Kaiser, O. Nachum, A. Roy, and S. Bengio.*
[paper](https://openreview.net/forum?id=SJTQLdqlg)

1. **Meta Networks,** in ICML, 2017.
*T. Munkhdalai and H. Yu.* 
[paper](http://proceedings.mlr.press/v70/munkhdalai17a/munkhdalai17a.pdf)

1. **Memory Matching Networks for One-Shot Image Recognition,** in CVPR, 2018.
*Q. Cai, Y. Pan, T. Yao, C. Yan, and T. Mei.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Memory_Matching_Networks_CVPR_2018_paper.pdf)

1. **Compound Memory Networks for Few-Shot Video Classification,** in ECCV, 2018.
*L. Zhu and Y. Yang.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.pdf)

1. **Memory, Show the Way: Memory Based Few Shot Word Representation Learning,** in EMNLP, 2018.
*J. Sun, S. Wang, and C. Zong.*
[paper](https://www.aclweb.org/anthology/D18-1173.pdf)

1. **Rapid Adaptation With Conditionally Shifted Neurons,** in ICML, 2018.
*T. Munkhdalai, X. Yuan, S. Mehri, and A. Trischler.*
[paper](http://proceedings.mlr.press/v80/munkhdalai18a/munkhdalai18a.pdf)

1. **Adaptive Posterior Learning: Few-Shot Learning With a Surprise-Based Memory Module,** in ICLR, 2019. 
*T. Ramalho and M. Garnelo.*
[paper](https://openreview.net/forum?id=ByeSdsC9Km)
[code](https://github.com/cogentlabs/apl)

1. **Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks,** in CVPR, 2019. 
*S. Yoo, H. Bahng, S. Chung, J. Lee, J. Chang, and J. Choo.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Coloring_With_Limited_Data_Few-Shot_Colorization_via_Memory_Augmented_Networks_CVPR_2019_paper.pdf)

1. **ACMM: Aligned Cross-Modal Memory for Few-Shot Image and Sentence Matching,** in ICCV, 2019. 
*Y. Huang, and L. Wang.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_ACMM_Aligned_Cross-Modal_Memory_for_Few-Shot_Image_and_Sentence_Matching_ICCV_2019_paper.pdf)

1. **Dynamic Memory Induction Networks for Few-Shot Text Classification,** in ACL, 2020.
*R. Geng, B. Li, Y. Li, J. Sun, and X. Zhu.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.102.pdf)

1. **Few-Shot Visual Learning With Contextual Memory and Fine-Grained Calibration,** in IJCAI, 2020.
*Y. Ma, W. Liu, S. Bai, Q. Zhang, A. Liu, W. Chen, and X. Liu.*
[paper](https://www.ijcai.org/Proceedings/2020/0113.pdf)

1. **Learn From Concepts: Towards the Purified Memory for Few-Shot Learning,** in IJCAI, 2021.
*X. Liu, X. Tian, S. Lin, Y. Qu, L. Ma, W. Yuan, Z. Zhang, and Y. Xie.*
[paper](https://www.ijcai.org/proceedings/2021/0123.pdf)

1. **Prototype Memory and Attention Mechanisms for Few Shot Image Generation,** in ICLR, 2022.
*T. Li, Z. Li, A. Luo, H. Rockwell, A. B. Farimani, and T. S. Lee.*
[paper](https://openreview.net/pdf?id=lY0-7bj0Vfz)
[code](https://github.com/Crazy-Jack/MoCA_release)

1. **Hierarchical Variational Memory for Few-Shot Learning Across Domains,** in ICLR, 2022.
*Y. Du, X. Zhen, L. Shao, and C. G. M. Snoek.*
[paper](https://openreview.net/pdf?id=i3RI65sR7N)
[code](https://github.com/YDU-uva/HierMemory)

1. **Remember the Difference: Cross-Domain Few-Shot Semantic Segmentation via Meta-Memory Transfer,** in CVPR, 2022.
*W. Wang, L. Duan, Y. Wang, Q. En, J. Fan, and Z. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Remember_the_Difference_Cross-Domain_Few-Shot_Semantic_Segmentation_via_Meta-Memory_Transfer_CVPR_2022_paper.pdf)

1. **Consistent Prototype Learning for Few-Shot Continual Relation Extraction,** in ACL, 2023.
*X. Chen, H. Wu, and X. Shi.*
[paper](https://aclanthology.org/2023.acl-long.409.pdf)
[code](https://github.com/XiudiChen/ConPL)

1. **Few-Shot Generation via Recalling Brain-Inspired Episodic-Semantic Memory,** in NeurIPS, 2023.
*Z. Duan, L. Zhiyi, C. Wang, B. Chen, B. An, and M. Zhou.*
[paper](https://openreview.net/attachment?id=dxPcdEeQk9&name=pdf)



### Generative Modeling

1. **One-shot learning of object categories,** TPAMI, 2006.
*L. Fei-Fei, R. Fergus, and P. Perona.*
[paper](http://vision.stanford.edu/documents/Fei-FeiFergusPerona2006.pdf)

1. **Learning to Learn With Compound HD Models,** in NeurIPS, 2011.
*A. Torralba, J. B. Tenenbaum, and R. R. Salakhutdinov.*
[paper](https://papers.nips.cc/paper/4474-learning-to-learn-with-compound-hd-models.pdf)

1. **One-Shot Learning With a Hierarchical Nonparametric Bayesian Model,** in ICML Workshop on Unsupervised and Transfer Learning, 2012.
*R. Salakhutdinov, J. Tenenbaum, and A. Torralba.*
[paper](http://proceedings.mlr.press/v27/salakhutdinov12a/salakhutdinov12a.pdf)

1. **Human-level concept learning through probabilistic program induction,** Science, 2015.
*B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum.*
[paper](https://web.mit.edu/cocosci/Papers/Science-2015-Lake-1332-8.pdf)

1. **One-Shot Generalization in Deep Generative Models,** in ICML, 2016.
*D. Rezende, I. Danihelka, K. Gregor, and D. Wierstra.*
[paper](https://arxiv.org/pdf/1603.05106)

1. **One-Shot Video Object Segmentation,** in CVPR, 2017.
*S. Caelles, K.-K. Maninis, J. Pont-Tuset, L. Leal-Taixé, D. Cremers,
and L. Van Gool.*
[paper](http://zpascal.net/cvpr2017/Caelles_One-Shot_Video_Object_CVPR_2017_paper.pdf)

1. **Towards a Neural Statistician,** in ICLR, 2017.
*H. Edwards and A. Storkey.*
[paper](https://openreview.net/forum?id=HJDBUF5le)

1. **Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples,** in ACL, 2018.
*V. Joshi, M. Peters, and M. Hopkins.*
[paper](https://www.aclweb.org/anthology/P18-1110.pdf)

1. **MetaGAN: An Adversarial Approach to Few-Shot Learning,** in NeurIPS, 2018.
*R. Zhang, T. Che, Z. Ghahramani, Y. Bengio, and Y. Song.*
[paper](https://papers.nips.cc/paper/7504-metagan-an-adversarial-approach-to-few-shot-learning.pdf)

1. **Few-Shot Autoregressive Density Estimation: Towards Learning to Learn Distributions,** in ICLR, 2018.
*S. Reed, Y. Chen, T. Paine, A. van den Oord, S. M. A. Eslami, D. Rezende, O. Vinyals, and N. de Freitas.* 
[paper](https://openreview.net/forum?id=r1wEFyWCW)

1. **The Variational Homoencoder: Learning to Learn High Capacity Generative Models From Few Examples,** in UAI, 2018.
*L. B. Hewitt, M. I. Nye, A. Gane, T. Jaakkola, and J. B. Tenenbaum.*
[paper](http://auai.org/uai2018/proceedings/papers/351.pdf)

1. **Meta-Learning Probabilistic Inference for Prediction,** in ICLR, 2019.
*J. Gordon, J. Bronskill, M. Bauer, S. Nowozin, and R. Turner.*
[paper](https://openreview.net/forum?id=HkxStoC5F7)

1. **Variational Prototyping-Encoder: One-Shot Learning With Prototypical Images,** in CVPR, 2019.
*J. Kim, T.-H. Oh, S. Lee, F. Pan, and I. S. Kweon.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Variational_Prototyping-Encoder_One-Shot_Learning_With_Prototypical_Images_CVPR_2019_paper.pdf)
[code](https://github.com/mibastro/VPE)

1. **Variational Few-Shot Learning,** in ICCV, 2019.
*J. Zhang, C. Zhao, B. Ni, M. Xu, and X. Yang.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Variational_Few-Shot_Learning_ICCV_2019_paper.pdf)

1. **Infinite Mixture Prototypes for Few-Shot Learning,** in ICML, 2019.
*K. Allen, E. Shelhamer, H. Shin, and J. Tenenbaum.*
[paper](http://proceedings.mlr.press/v97/allen19b/allen19b.pdf)

1. **Dual Variational Generation for Low Shot Heterogeneous Face Recognition,** in NeurIPS, 2019.
*C. Fu, X. Wu, Y. Hu, H. Huang, and R. He.*
[paper](https://papers.nips.cc/paper/8535-dual-variational-generation-for-low-shot-heterogeneous-face-recognition.pdf)

1. **Bayesian Meta Sampling for Fast Uncertainty Adaptation,** in ICLR, 2020.
*Z. Wang, Y. Zhao, P. Yu, R. Zhang, and C. Chen.*
[paper](https://openreview.net/pdf?id=Bkxv90EKPB)

1. **Empirical Bayes Transductive Meta-Learning With Synthetic Gradients,** in ICLR, 2020.
*S. X. Hu, P. G. Moreno, Y. Xiao, X. Shen, G. Obozinski, N. D. Lawrence, and A. C. Damianou.*
[paper](https://openreview.net/pdf?id=Hkg-xgrYvH)

1. **Few-Shot Relation Extraction via Bayesian Meta-Learning on Relation Graphs,** in ICML, 2020.
*M. Qu, T. Gao, L. A. C. Xhonneux, and J. Tang.*
[paper](http://proceedings.mlr.press/v119/qu20a/qu20a.pdf)
[code](https://github.com/DeepGraphLearning/FewShotRE)

1. **Interventional Few-Shot Learning,** in NeurIPS, 2020.
*Z. Yue, H. Zhang, Q. Sun, and X. Hua.*
[paper](https://proceedings.neurips.cc/paper/2020/file/1cc8a8ea51cd0adddf5dab504a285915-Paper.pdf)
[code](https://github.com/yue-zhongqi/ifsl)

1. **Bayesian Few-Shot Classification With One-vs-Each Pólya-Gamma Augmented Gaussian Processes,** in ICLR, 2021.
*J. Snell, and R. Zemel.*
[paper](https://openreview.net/pdf?id=lgNx56yZh8a)

1. **Few-Shot Bayesian Optimization With Deep Kernel Surrogates,** in ICLR, 2021.
*M. Wistuba, and J. Grabocka.*
[paper](https://openreview.net/pdf?id=bJxgv5C3sYc)

1. **Reinforced Few-Shot Acquisition Function Learning for Bayesian Optimization,** in NeurIPS, 2021.
*B. Hsieh, P. Hsieh, and X. Liu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/3fab5890d8113d0b5a4178201dc842ad-Paper.pdf)

1. **GanOrCon: Are Generative Models Useful for Few-Shot Segmentation?,** in CVPR, 2022.
*O. Saha, Z. Cheng, and S. Maji.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Saha_GanOrCon_Are_Generative_Models_Useful_for_Few-Shot_Segmentation_CVPR_2022_paper.pdf)

1. **Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment,** in CVPR, 2022.
*J. Xiao, L. Li, C. Wang, Z. Zha, and Q. Huang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xiao_Few_Shot_Generative_Model_Adaption_via_Relaxed_Spatial_Structural_Alignment_CVPR_2022_paper.pdf)

1. **SCHA-VAE: Hierarchical Context Aggregation for Few-Shot Generation,** in ICML, 2022.
*G. Giannone, and O. Winther.*
[paper](https://proceedings.mlr.press/v162/giannone22a/giannone22a.pdf)
[code](https://github.com/georgosgeorgos/hierarchical-few-shot-generative-models)

1. **Diversity vs. Recognizability: Human-Like Generalization in One-Shot Generative Models,** in NeurIPS, 2022.
*V. Boutin, L. Singhal, X. Thomas, and T. Serre.*
[paper](https://openreview.net/pdf?id=DVfZKXSFW5m)
[code](https://github.com/serre-lab/diversity_vs_recognizability)

1. **Generalized One-Shot Domain Adaptation of Generative Adversarial Networks,** in NeurIPS, 2022.
*Z. Zhang, Y. Liu, C. Han, T. Guo, T. Yao, and T. Mei.*
[paper](https://openreview.net/pdf?id=mfxq7BrMfga)
[code](https://github.com/zhangzc21/Generalized-One-shot-GAN-Adaptation)

1. **Towards Diverse and Faithful One-Shot Adaption of Generative Adversarial Networks,** in NeurIPS, 2022.
*Y. Zhang, m. Yao, Y. Wei, Z. Ji, J. Bai, and W. Zuo.*
[paper](https://openreview.net/pdf?id=IXoHxXIGpyV)
[code](https://github.com/YBYBZhang/DiFa)

1. **Few-Shot Cross-Domain Image Generation via Inference-Time Latent-Code Learning,** in ICLR, 2023.
*A. K. Mondal, P. Tiwary, P. Singla, and P. AP.*
[paper](https://openreview.net/pdf?id=sCYXJr3QJM8)
[code](https://github.com/arnabkmondal/GenDA)

1. **Adaptive IMLE for Few-Shot Pretraining-Free Generative Modelling,** in ICML, 2023.
*M. Aghabozorgi, S. Peng, and K. Li.*
[paper](https://proceedings.mlr.press/v202/aghabozorgi23a/aghabozorgi23a.pdf)
[code](https://github.com/mehranagh20/AdaIMLE)

1. **Diversity-Enhancing Generative Network for Few-Shot Hypothesis Adaptation,** in ICML, 2023.
*R. Dong, F. Liu, H. Chi, T. Liu, M. Gong, G. Niu, M. Sugiyama, and B. Han.*
[paper](https://proceedings.mlr.press/v202/dong23d/dong23d.pdf)

1. **MetaModulation: Learning Variational Feature Hierarchies for Few-Shot Learning With Fewer Tasks,** in ICML, 2023.
*W. Sun, Y. Du, X. Zhen, F. Wang, L. Wang, and C. G. M. Snoek.*
[paper](https://proceedings.mlr.press/v202/sun23b/sun23b.pdf)
[code](https://github.com/lmsdss/MetaModulation)

1. **Revisiting Logistic-Softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification,** in NeurIPS, 2023.
*T. Ke, H. Cao, Z. Ling, and F. Zhou.*
[paper](https://openreview.net/attachment?id=Z1W0u3Cr74&name=pdf)
[code](https://github.com/keanson/revisit-logistic-softmax)

1. **Human-Like Few-Shot Learning via Bayesian Reasoning Over Natural Language,** in NeurIPS, 2023.
*and K. Ellis.*
[paper](https://openreview.net/attachment?id=dVnhdm9MIg&name=pdf)
[code](https://github.com/ellisk42/humanlike_fewshot_learning)

1. **CMVAE: Causal Meta VAE for Unsupervised Meta-Learning,** in AAAI, 2023.
*G. Qi, and H. Yu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26135/25907)
[code](https://github.com/GuodongQi/CMVAE)

1. **Progressive Few-Shot Adaptation of Generative Model With Align-Free Spatial Correlation,** in AAAI, 2023.
*J. Moon, H. Kim, and J.-P. Heo.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25283/25055)

1. **Few-Shot Learner Parameterization by Diffusion Time-Steps,** in CVPR, 2024.
*Z. Yue, P. Zhou, R. Hong, H. Zhang, and Q. Sun.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02195)

1. **Generate Universal Adversarial Perturbations for Few-Shot Learning.,** in NeurIPS, 2024.
*Y. Hu, Y. Zou, R. Li, and Y. Li.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/ba1c5356d9164bb64c446a4b690226b0-Paper-Conference.pdf)

1. **Few-Shot Task Learning through Inverse Generative Modeling.,** in NeurIPS, 2024.
*A. Netanyahu, Y. Du, A. Bronars, J. Pari, J. Tenenbaum, T. Shu, and P. Agrawal.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/b25222d2d405e0768d218e7fc90070b2-Paper-Conference.pdf)

1. **Few-Shot Diffusion Models Escape the Curse of Dimensionality.,** in NeurIPS, 2024.
*R. Yang, B. Jiang, C. Chen, R. Jin, B. Wang, and S. Li.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/7eb6233e02f7d9efbb84acd839a996fb-Paper-Conference.pdf)

1. **Accelerating Convergence in Bayesian Few-Shot Classification.,** in ICML, 2024.
*T. Ke, H. Cao, and F. Zhou.*
[paper](https://openreview.net/forum?id=9PQnc6EWdL)

## [Algorithm](#content)

### Refining Existing Parameters

1. **Cross-Generalization: Learning Novel Classes From a Single Example by Feature Replacement,** in CVPR, 2005. 
*E. Bart and S. Ullman.*
[paper](http://www.inf.tu-dresden.de/content/institutes/ki/is/HS_SS08_Papers/BartUllmanCVPR05.pdf)

1. **One-Shot Adaptation of Supervised Deep Convolutional Models,** in ICLR, 2013.
*J. Hoffman, E. Tzeng, J. Donahue, Y. Jia, K. Saenko, and T. Darrell.*
[paper](https://openreview.net/forum?id=tPCrkaLa9Y5ld)

1. **Learning to Learn: Model Regression Networks for Easy Small Sample Learning,** in ECCV, 2016.
*Y.-X. Wang and M. Hebert.*
[paper](https://ri.cmu.edu/pub_files/2016/10/yuxiongw_eccv16_learntolearn.pdf)

1. **Learning From Small Sample Sets by Combining Unsupervised Meta-Training With CNNs,** in NeurIPS, 2016.
*Y.-X. Wang and M. Hebert.*
[paper](https://papers.nips.cc/paper/6408-learning-from-small-sample-sets-by-combining-unsupervised-meta-training-with-cnns)

1. **Efficient K-Shot Learning With Regularized Deep Networks,** in AAAI, 2018.
*D. Yoo, H. Fan, V. N. Boddeti, and K. M. Kitani.*
[paper](https://arxiv.org/abs/1710.02277)

1. **CLEAR: Cumulative Learning for One-Shot One-Class Image Recognition,** in CVPR, 2018.
*J. Kozerawski and M. Turk.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kozerawski_CLEAR_Cumulative_LEARning_CVPR_2018_paper.pdf)

1. **Learning Structure and Strength of CNN Filters for Small Sample Size Training,** in CVPR, 2018. 
*R. Keshari, M. Vatsa, R. Singh, and A. Noore.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Keshari_Learning_Structure_and_CVPR_2018_paper.pdf)

1. **Dynamic Few-Shot Visual Learning Without Forgetting,** in CVPR, 2018.
*S. Gidaris and N. Komodakis.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gidaris_Dynamic_Few-Shot_Visual_CVPR_2018_paper.pdf)
[code](https://github.com/gidariss/FewShotWithoutForgetting)

1. **Low-Shot Learning With Imprinted Weights,** in CVPR, 2018.
*H. Qi, M. Brown, and D. G. Lowe.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf)

1. **Neural Voice Cloning With a Few Samples,** in NeurIPS, 2018.
*S. Arik, J. Chen, K. Peng, W. Ping, and Y. Zhou.*
[paper](https://papers.nips.cc/paper/8206-neural-voice-cloning-with-a-few-samples.pdf)

1. **Text Classification With Few Examples Using Controlled Generalization,** in NAACL-HLT, 2019.
*A. Mahabal, J. Baldridge, B. K. Ayan, V. Perot, and D. Roth.* 
[paper](https://www.aclweb.org/anthology/N19-1319.pdf)

1. **Low Shot Box Correction for Weakly Supervised Object Detection,** in IJCAI, 2019.
*T. Pan, B. Wang, G. Ding, J. Han, and J. Yong.*
[paper](https://www.ijcai.org/Proceedings/2019/0125.pdf)

1. **Diversity With Cooperation: Ensemble Methods for Few-Shot Classification,** in ICCV, 2019.
*N. Dvornik, C. Schmid, and J. Mairal.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dvornik_Diversity_With_Cooperation_Ensemble_Methods_for_Few-Shot_Classification_ICCV_2019_paper.pdf)

1. **Few-Shot Image Recognition With Knowledge Transfer,** in ICCV, 2019.
*Z. Peng, Z. Li, J. Zhang, Y. Li, G.-J. Qi, and J. Tang.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Few-Shot_Image_Recognition_With_Knowledge_Transfer_ICCV_2019_paper.pdf)

1. **Generating Classification Weights With GNN Denoising Autoencoders for Few-Shot Learning,** in CVPR, 2019.
*S. Gidaris, and N. Komodakis.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gidaris_Generating_Classification_Weights_With_GNN_Denoising_Autoencoders_for_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/gidariss/wDAE_GNN_FewShot)

1. **Dense Classification and Implanting for Few-Shot Learning,** in CVPR, 2019.
*Y. Lifchitz, Y. Avrithis, S. Picard, and A. Bursuc.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lifchitz_Dense_Classification_and_Implanting_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

1. **Few-Shot Adaptive Faster R-CNN,** in CVPR, 2019.
*T. Wang, X. Zhang, L. Yuan, and J. Feng.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Few-Shot_Adaptive_Faster_R-CNN_CVPR_2019_paper.pdf)

1. **TransMatch: A Transfer-Learning Scheme for Semi-Supervised Few-Shot Learning,** in CVPR, 2020.
*Z. Yu, L. Chen, Z. Cheng, and J. Luo.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_TransMatch_A_Transfer-Learning_Scheme_for_Semi-Supervised_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Learning to Select Base Classes for Few-Shot Classification,** in CVPR, 2020.
*L. Zhou, P. Cui, X. Jia, S. Yang, and Q. Tian.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Learning_to_Select_Base_Classes_for_Few-Shot_Classification_CVPR_2020_paper.pdf)

1. **Few-Shot NLG With Pre-Trained Language Model,** in ACL, 2020.
*Z. Chen, H. Eavani, W. Chen, Y. Liu, and W. Y. Wang.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.18.pdf)
[code](https://github.com/czyssrs/Few-Shot-NLG)

1. **Span-ConveRT: Few-Shot Span Extraction for Dialog With Pretrained Conversational Representations,** in ACL, 2020.
*S. Coope, T. Farghly, D. Gerz, I. Vulic, and M. Henderson.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.11.pdf)

1. **Structural Supervision Improves Few-Shot Learning and Syntactic Generalization in Neural Language Models,** in EMNLP, 2020.
*E. Wilcox, P. Qian, R. Futrell, R. Kohita, R. Levy, and M. Ballesteros.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.375.pdf)
[code](https://github.com/wilcoxeg/fsl_invar)

1. **A Baseline for Few-Shot Image Classification,** in ICLR, 2020.
*G. S. Dhillon, P. Chaudhari, A. Ravichandran, and S. Soatto.*
[paper](https://openreview.net/pdf?id=rylXBkrYDS)

1. **Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation,** in ICLR, 2020.
*H. Tseng, H. Lee, J. Huang, and M. Yang.*
[paper](https://openreview.net/pdf?id=SJl5Np4tPr)
[code](https://github.com/hytseng0509/CrossDomainFewShot)

1. **Graph Few-Shot Learning via Knowledge Transfer,** in AAAI, 2020.
*H. Yao, C. Zhang, Y. Wei, M. Jiang, S. Wang, J. Huang, N. V. Chawla, and Z. Li.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6142)

1. **Context-Transformer: Tackling Object Confusion for Few-Shot Detection,** in AAAI, 2020.
*Z. Yang, Y. Wang, X. Chen, J. Liu, and Y. Qiao.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6957)

1. **A Broader Study of Cross-Domain Few-Shot Learning,** in ECCV, 2020.
*Y. Guo, N. C. Codella, L. Karlinsky, J. V. Codella, J. R. Smith, K. Saenko, T. Rosing, and R. Feris.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf)
[code](https://github.com/IBM/cdfsl-benchmark)

1. **Selecting Relevant Features From a Multi-Domain Representation for Few-Shot Classification,** in ECCV, 2020.
*N. Dvornik, C. Schmid, and J. Mairal.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550766.pdf)
[code](https://github.com/dvornikita/SUR)

1. **Prototype Completion With Primitive Knowledge for Few-Shot Learning,** in CVPR, 2021.
*B. Zhang, X. Li, Y. Ye, Z. Huang, and L. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Prototype_Completion_With_Primitive_Knowledge_for_Few-Shot_Learning_CVPR_2021_paper.pdf)
[code](https://github.com/zhangbq-research/Prototype_Completion_for_FSL)

1. **Partial Is Better Than All: Revisiting Fine-Tuning Strategy for Few-Shot Learning,** in AAAI, 2021.
*Z. Shen, Z. Liu, J. Qin, M. Savvides, and K.-T. Cheng.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17155/16962)

1. **PTN: A Poisson Transfer Network for Semi-Supervised Few-Shot Learning,** in AAAI, 2021.
*H. Huang, J. Zhang, J. Zhang, Q. Wu, and C. Xu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16252/16059)

1. **A Universal Representation Transformer Layer for Few-Shot Image Classification,** in ICLR, 2021.
*L. Liu, W. L. Hamilton, G. Long, J. Jiang, and H. Larochelle.*
[paper](https://openreview.net/pdf?id=04cII6MumYV)

1. **Making Pre-Trained Language Models Better Few-Shot Learners,** in ACL-IJCNLP, 2021.
*T. Gao, A. Fisch, and D. Chen.*
[paper](https://aclanthology.org/2021.acl-long.295.pdf)
[code](https://github.com/princeton-nlp/LM-BFF)

1. **Self-Supervised Network Evolution for Few-Shot Classification,** in IJCAI, 2021.
*X. Tang, Z. Teng, B. Zhang, and J. Fan.*
[paper](https://www.ijcai.org/proceedings/2021/0419.pdf)

1. **Calibrate Before Use: Improving Few-Shot Performance of Language Models,** in ICML, 2021.
*Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh.*
[paper](http://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)
[code](https://www.github.com/tonyzhaozh/few-shot-learning)

1. **Language Models Are Few-Shot Learners,** in NeurIPS, 2020.
*T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei.*
[paper](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

1. **It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners,** in NAACL-HLT, 2021.
*T. Schick, and H. Schütze.*
[paper](https://aclanthology.org/2021.naacl-main.185.pdf)
[code](https://github.com/timoschick/pet)

1. **Self-Training Improves Pre-Training for Few-Shot Learning in Task-Oriented Dialog Systems,** in EMNLP, 2021.
*F. Mi, W. Zhou, L. Kong, F. Cai, M. Huang, and B. Faltings.*
[paper](https://aclanthology.org/2021.emnlp-main.142.pdf)

1. **Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning,** in EMNLP, 2021.
*J. Zhang, T. Bui, S. Yoon, X. Chen, Z. Liu, C. Xia, Q. H. Tran, W. Chang, and P. S. Yu.*
[paper](https://aclanthology.org/2021.emnlp-main.144.pdf)
[code](https://github.com/jianguoz/Few-Shot-Intent-Detection)

1. **Avoiding Inference Heuristics in Few-Shot Prompt-Based Finetuning,** in EMNLP, 2021.
*P. A. Utama, N. S. Moosavi, V. Sanh, and I. Gurevych.*
[paper](https://aclanthology.org/2021.emnlp-main.713.pdf)
[code](https://github.com/ukplab/emnlp2021-prompt-ft-heuristics)

1. **Constrained Language Models Yield Few-Shot Semantic Parsers,** in EMNLP, 2021.
*R. Shin, C. H. Lin, S. Thomson, C. Chen, S. Roy, E. A. Platanios, A. Pauls, D. Klein, J. Eisner, and B. V. Durme.*
[paper](https://aclanthology.org/2021.emnlp-main.608.pdf)
[code](https://github.com/microsoft/semantic_parsing_with_constrained_lm)

1. **Revisiting Self-Training for Few-Shot Learning of Language Model,** in EMNLP, 2021.
*Y. Chen, Y. Zhang, C. Zhang, G. Lee, R. Cheng, and H. Li.*
[paper](https://aclanthology.org/2021.emnlp-main.718.pdf)
[code](https://github.com/matthewcym/sflm)

1. **Language Models Are Few-Shot Butlers,** in EMNLP, 2021.
*V. Micheli, and F. Fleuret.*
[paper](https://aclanthology.org/2021.emnlp-main.734.pdf)
[code](https://github.com/vmicheli/lm-butlers)

1. **FewshotQA: A Simple Framework for Few-Shot Learning of Question Answering Tasks Using Pre-Trained Text-to-Text Models,** in EMNLP, 2021.
*R. Chada, and P. Natarajan.*
[paper](https://aclanthology.org/2021.emnlp-main.491.pdf)

1. **TransPrompt: Towards an Automatic Transferable Prompting Framework for Few-Shot Text Classification,** in EMNLP, 2021.
*C. Wang, J. Wang, M. Qiu, J. Huang, and M. Gao.*
[paper](https://aclanthology.org/2021.emnlp-main.221.pdf)

1. **Meta Distant Transfer Learning for Pre-Trained Language Models,** in EMNLP, 2021.
*C. Wang, H. Pan, M. Qiu, J. Huang, F. Yang, and Y. Zhang.*
[paper](https://aclanthology.org/2021.emnlp-main.768.pdf)

1. **STraTA: Self-Training With Task Augmentation for Better Few-Shot Learning,** in EMNLP, 2021.
*T. Vu, M. Luong, Q. V. Le, G. Simon, and M. Iyyer.*
[paper](https://aclanthology.org/2021.emnlp-main.462.pdf)
[code](https://github.com/google-research/google-research)

1. **Few-Shot Image Classification: Just Use a Library of Pre-Trained Feature Extractors and a Simple Classifier,** in ICCV, 2021.
*A. Chowdhury, M. Jiang, S. Chaudhuri, and C. Jermaine.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chowdhury_Few-Shot_Image_Classification_Just_Use_a_Library_of_Pre-Trained_Feature_ICCV_2021_paper.pdf)
[code](https://github.com/arjish/PreTrainedFullLibrary_FewShot)

1. **On the Importance of Distractors for Few-Shot Classification,** in ICCV, 2021.
*R. Das, Y. Wang, and J. M. F. Moura.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Das_On_the_Importance_of_Distractors_for_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/quantacode/contrastive-finetuning)

1. **A Multi-Mode Modulator for Multi-Domain Few-Shot Classification,** in ICCV, 2021.
*Y. Liu, J. Lee, L. Zhu, L. Chen, H. Shi, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_A_Multi-Mode_Modulator_for_Multi-Domain_Few-Shot_Classification_ICCV_2021_paper.pdf)

1. **Universal Representation Learning From Multiple Domains for Few-Shot Classification,** in ICCV, 2021.
*W. Li, X. Liu, and H. Bilen.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Universal_Representation_Learning_From_Multiple_Domains_for_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/VICO-UoE/URL)

1. **Boosting the Generalization Capability in Cross-Domain Few-Shot Learning via Noise-Enhanced Supervised Autoencoder,** in ICCV, 2021.
*H. Liang, Q. Zhang, P. Dai, and J. Lu.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Boosting_the_Generalization_Capability_in_Cross-Domain_Few-Shot_Learning_via_Noise-Enhanced_ICCV_2021_paper.pdf)

1. **How Fine-Tuning Allows for Effective Meta-Learning,** in NeurIPS, 2021.
*K. Chua, Q. Lei, and J. D. Lee.*
[paper](https://proceedings.neurips.cc/paper/2021/file/4a533591763dfa743a13affab1a85793-Paper.pdf)

1. **Multimodal Few-Shot Learning With Frozen Language Models,** in NeurIPS, 2021.
*M. Tsimpoukelli, J. Menick, S. Cabi, S. M. A. Eslami, O. Vinyals, and F. Hill.*
[paper](https://proceedings.neurips.cc/paper/2021/file/01b7575c38dac42f3cfb7d500438b875-Paper.pdf)

1. **Grad2Task: Improved Few-Shot Text Classification Using Gradients for Task Representation,** in NeurIPS, 2021.
*J. Wang, K. Wang, F. Rudzicz, and M. Brudno.*
[paper](https://proceedings.neurips.cc/paper/2021/file/33a854e247155d590883b93bca53848a-Paper.pdf)

1. **True Few-Shot Learning With Language Models,** in NeurIPS, 2021.
*E. Perez, D. Kiela, and K. Cho.*
[paper](https://proceedings.neurips.cc/paper/2021/file/5c04925674920eb58467fb52ce4ef728-Paper.pdf)

1. **POODLE: Improving Few-Shot Learning via Penalizing Out-of-Distribution Samples,** in NeurIPS, 2021.
*D. Le, K. Nguyen, Q. Tran, R. Nguyen, and B. Hua.*
[paper](https://proceedings.neurips.cc/paper/2021/file/c91591a8d461c2869b9f535ded3e213e-Paper.pdf)

1. **TOHAN: A One-Step Approach Towards Few-Shot Hypothesis Adaptation,** in NeurIPS, 2021.
*H. Chi, F. Liu, W. Yang, L. Lan, T. Liu, B. Han, W. Cheung, and J. Kwok.*
[paper](https://proceedings.neurips.cc/paper/2021/file/af5d5ef24881f3c3049a7b9bfe74d58b-Paper.pdf)

1. **Task Affinity With Maximum Bipartite Matching in Few-Shot Learning,** in ICLR, 2022.
*C. P. Le, J. Dong, M. Soltani, and V. Tarokh.*
[paper](https://openreview.net/pdf?id=u2GZOiUTbt)

1. **Differentiable Prompt Makes Pre-Trained Language Models Better Few-Shot Learners,** in ICLR, 2022.
*N. Zhang, L. Li, X. Chen, S. Deng, Z. Bi, C. Tan, F. Huang, and H. Chen.*
[paper](https://openreview.net/pdf?id=ek9a0qIafW)
[code](https://github.com/zjunlp/DART)

1. **ConFeSS: A Framework for Single Source Cross-Domain Few-Shot Learning,** in ICLR, 2022.
*D. Das, S. Yun, and F. Porikli.*
[paper](https://openreview.net/pdf?id=zRJu6mU2BaE)

1. **Switch to Generalize: Domain-Switch Learning for Cross-Domain Few-Shot Classification,** in ICLR, 2022.
*Z. Hu, Y. Sun, and Y. Yang.*
[paper](https://openreview.net/pdf?id=H-iABMvzIc)

1. **LM-BFF-MS: Improving Few-Shot Fine-Tuning of Language Models Based on Multiple Soft Demonstration Memory,** in ACL, 2022.
*E. Park, D. H. Jeon, S. Kim, I. Kang, and S. Na.*
[paper](https://aclanthology.org/2022.acl-short.34.pdf)
[code](https://github.com/judepark96/lm-bff-ms)

1. **Meta-Learning via Language Model in-Context Tuning,** in ACL, 2022.
*Y. Chen, R. Zhong, S. Zha, G. Karypis, and H. He.*
[paper](https://aclanthology.org/2022.acl-long.53.pdf)
[code](https://github.com/yandachen/in-context-tuning)

1. **Few-Shot Tabular Data Enrichment Using Fine-Tuned Transformer Architectures,** in ACL, 2022.
*A. Harari, and G. Katz.*
[paper](https://aclanthology.org/2022.acl-long.111.pdf)

1. **Noisy Channel Language Model Prompting for Few-Shot Text Classification,** in ACL, 2022.
*S. Min, M. Lewis, H. Hajishirzi, and L. Zettlemoyer.*
[paper](https://aclanthology.org/2022.acl-long.365.pdf)
[code](https://github.com/shmsw25/Channel-LM-Prompting)

1. **Prompt for Extraction? PAIE: Prompting Argument Interaction for Event Argument Extraction,** in ACL, 2022.
*Y. Ma, Z. Wang, Y. Cao, M. Li, M. Chen, K. Wang, and J. Shao.*
[paper](https://aclanthology.org/2022.acl-long.466.pdf)
[code](https://github.com/mayubo2333/paie)

1. **Are Prompt-Based Models Clueless?,** in ACL, 2022.
*P. Kavumba, R. Takahashi, and Y. Oda.*
[paper](https://aclanthology.org/2022.acl-long.166.pdf)

1. **Prototypical Verbalizer for Prompt-Based Few-Shot Tuning,** in ACL, 2022.
*G. Cui, S. Hu, N. Ding, L. Huang, and Z. Liu.*
[paper](https://aclanthology.org/2022.acl-long.483.pdf)
[code](https://github.com/thunlp/OpenPrompt)

1. **Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity,** in ACL, 2022.
*Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp.*
[paper](https://aclanthology.org/2022.acl-long.556.pdf)

1. **PPT: Pre-Trained Prompt Tuning for Few-Shot Learning,** in ACL, 2022.
*Y. Gu, X. Han, Z. Liu, and M. Huang.*
[paper](https://aclanthology.org/2022.acl-long.576.pdf)
[code](https://github.com/thu-coai/ppt)

1. **ASCM: An Answer Space Clustered Prompting Method Without Answer Engineering,** in Findings of ACL, 2022.
*Z. Wang, Y. Yang, Z. Xi, B. Ma, L. Wang, R. Dong, and A. Anwar.*
[paper](https://aclanthology.org/2022.findings-acl.193.pdf)
[code](https://github.com/miaomiao1215/ascm)

1. **Exploiting Language Model Prompts Using Similarity Measures: A Case Study on the Word-in-Context Task,** in ACL, 2022.
*M. Tabasi, K. Rezaee, and M. T. Pilehvar.*
[paper](https://aclanthology.org/2022.acl-short.36.pdf)

1. **P-Tuning: Prompt Tuning Can Be Comparable to Fine-Tuning Across Scales and Tasks,** in ACL, 2022.
*X. Liu, K. Ji, Y. Fu, W. Tam, Z. Du, Z. Yang, and J. Tang.*
[paper](https://aclanthology.org/2022.acl-short.8.pdf)

1. **Cutting Down on Prompts and Parameters: Simple Few-Shot Learning With Language Models,** in Findings of ACL, 2022.
*R. L. L. IV, I. Balazevic, E. Wallace, F. Petroni, S. Singh, and S. Riedel.*
[paper](https://aclanthology.org/2022.findings-acl.222.pdf)
[code](https://github.com/ucinlp/null-prompts)

1. **Prompt-Free and Efficient Few-Shot Learning With Language Models,** in ACL, 2022.
*R. K. Mahabadi, L. Zettlemoyer, J. Henderson, L. Mathias, M. Saeidi, V. Stoyanov, and M. Yazdani.*
[paper](https://aclanthology.org/2022.acl-long.254.pdf)
[code](https://github.com/facebookresearch/perfect)

1. **Pre-Training to Match for Unified Low-Shot Relation Extraction,** in ACL, 2022.
*F. Liu, H. Lin, X. Han, B. Cao, and L. Sun.*
[paper](https://aclanthology.org/2022.acl-long.397.pdf)
[code](https://github.com/fc-liu/mcmn)

1. **Dual Context-Guided Continuous Prompt Tuning for Few-Shot Learning,** in Findings of ACL, 2022.
*J. Zhou, L. Tian, H. Yu, Z. Xiao, H. Su, and J. Zhou.*
[paper](https://aclanthology.org/2022.findings-acl.8.pdf)

1. **Cluster & Tune: Boost Cold Start Performance in Text Classification,** in ACL, 2022.
*E. Shnarch, A. Gera, A. Halfon, L. Dankin, L. Choshen, R. Aharonov, and N. Slonim.*
[paper](https://aclanthology.org/2022.acl-long.526.pdf)
[code](https://github.com/ibm/intermediate-training-using-clustering)

1. **Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference,** in CVPR, 2022.
*S. X. Hu, D. Li, J. Stühmer, M. Kim, and T. M. Hospedales.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Pushing_the_Limits_of_Simple_Pipelines_for_Few-Shot_Learning_External_CVPR_2022_paper.pdf)
[code](https://hushell.github.io/pmf/)

1. **HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning,** in ICML, 2022.
*A. Zhmoginov, M. Sandler, and M. Vladymyrov.*
[paper](https://proceedings.mlr.press/v162/zhmoginov22a/zhmoginov22a.pdf)
[code](https://github.com/google-research/google-research/tree/master/hypertransformer)

1. **Prompting ELECTRA: Few-Shot Learning With Discriminative Pre-Trained Models,** in EMNLP, 2022.
*M. Xia, M. Artetxe, J. Du, D. Chen, and V. Stoyanov.*
[paper](https://aclanthology.org/2022.emnlp-main.780.pdf)
[code](https://github.com/facebookresearch/ELECTRA-Fewshot-Learning)

1. **Continual Training of Language Models for Few-Shot Learning,** in EMNLP, 2022.
*Z. Ke, H. Lin, Y. Shao, H. Xu, L. Shu, and B. Liu.*
[paper](https://aclanthology.org/2022.emnlp-main.695.pdf)
[code](https://github.com/UIC-Liu-Lab/CPT)

1. **GPS: Genetic Prompt Search for Efficient Few-Shot Learning,** in EMNLP, 2022.
*H. Xu, Y. Chen, Y. Du, N. Shao, Y. Wang, H. Li, and Z. Yang.*
[paper](https://aclanthology.org/2022.emnlp-main.559.pdf)
[code](https://github.com/hwxu20/GPS)

1. **On Measuring the Intrinsic Few-Shot Hardness of Datasets,** in EMNLP, 2022.
*X. Zhao, S. Murty, and C. D. Manning.*
[paper](https://aclanthology.org/2022.emnlp-main.262.pdf)
[code](https://github.com/colinzhaoust/intrinsic_fewshot_hardness)

1. **AMAL: Meta Knowledge-Driven Few-Shot Adapter Learning,** in EMNLP, 2022.
*S. K. Hong, and T. Y. Jang.*
[paper](https://aclanthology.org/2022.emnlp-main.709.pdf)

1. **Flamingo: A Visual Language Model for Few-Shot Learning,** in NeurIPS, 2022.
*J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan.*
[paper](https://openreview.net/pdf?id=EbMuimAbPbs)

1. **Language Models With Image Descriptors Are Strong Few-Shot Video-Language Learners,** in NeurIPS, 2022.
*Z. Wang, M. Li, R. Xu, L. Zhou, J. Lei, X. Lin, S. Wang, Z. Yang, C. Zhu, D. Hoiem, S.-F. Chang, M. Bansal, and H. Ji.*
[paper](https://openreview.net/pdf?id=_LceCyuVcH)
[code](https://github.com/MikeWangWZHL/VidIL)

1. **Singular Value Fine-Tuning: Few-Shot Segmentation Requires Few-Parameters Fine-Tuning,** in NeurIPS, 2022.
*Y. Sun, Q. Chen, X. He, J. Wang, H. Feng, J. Han, E. Ding, J. Cheng, Z. Li, and J. Wang.*
[paper](https://openreview.net/pdf?id=LEqYZz7cZOI)
[code](https://github.com/syp2ysy/SVF)

1. **Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper Than in-Context Learning,** in NeurIPS, 2022.
*H. Liu, D. Tam, M. Mohammed, J. Mohta, T. Huang, M. Bansal, and C. Raffel.*
[paper](https://openreview.net/pdf?id=rBCvMG-JsPd)
[code](https://github.com/r-three/t-few)

1. **Powering Finetuning in Few-Shot Learning: Domain-Agnostic Bias Reduction With Selected Sampling,** in AAAI, 2022.
*R. Tao, H. Zhang, Y. Zheng, and M. Savvides.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20823/20582)

1. **SELECTION-INFERENCE: Exploiting Large Language Models for Interpretable Logical Reasoning,** in ICLR, 2023.
*A. Creswell, M. Shannahan, and I. Higgins.*
[paper](https://arxiv.org/pdf/2205.09712.pdf)

1. **Revisit Finetuning Strategy for Few-Shot Learning to Transfer the Emdeddings,** in ICLR, 2023.
*H. Wang, T. Yue, X. Ye, Z. He, B. Li, and Y. Li.*
[paper](https://openreview.net/pdf?id=tXc-riXhmx)
[code](https://github.com/whzyf951620/ LinearProbingFinetuningFirthBias)

1. **Model Ensemble Instead of Prompt Fusion: A Sample-Specific Knowledge Transfer Method for Few-Shot Prompt Tuning,** in ICLR, 2023.
*X. PENG, C. Xing, P. K. Choubey, C.-S. Wu, and C. Xiong.*
[paper](https://openreview.net/pdf?id=p0yrSRbN5Bu)

1. **Bidirectional Language Models Are Also Few-Shot Learners,** in ICLR, 2023.
*A. Patel, B. Li, M. S. Rasooli, N. Constant, C. Raffel, and C. Callison-Burch.*
[paper](https://openreview.net/pdf?id=wCFB37bzud4)

1. **Prototypical Calibration for Few-Shot Learning of Language Models,** in ICLR, 2023.
*Z. Han, Y. Hao, L. Dong, Y. Sun, and F. Wei.*
[paper](https://openreview.net/pdf?id=nUsP9lFADUF)
[code](https://github.com/zhixhan/ProCa)

1. **Prompt, Generate, Then Cache: Cascade of Foundation Models Makes Strong Few-Shot Learners,** in CVPR, 2023.
*R. Zhang, X. Hu, B. Li, S. Huang, H. Deng, Y. Qiao, P. Gao, and H. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Prompt_Generate_Then_Cache_Cascade_of_Foundation_Models_Makes_Strong_CVPR_2023_paper.pdf)
[code](https://github.com/ZrrSkywalker/CaFo)

1. **Supervised Masked Knowledge Distillation for Few-Shot Transformers,** in CVPR, 2023.
*H. Lin, G. Han, J. Ma, S. Huang, X. Lin, and S.-F. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Supervised_Masked_Knowledge_Distillation_for_Few-Shot_Transformers_CVPR_2023_paper.pdf)
[code](https://github.com/HL-hanlin/SMKD)

1. **Boosting Transductive Few-Shot Fine-Tuning With Margin-Based Uncertainty Weighting and Probability Regularization,** in CVPR, 2023.
*R. Tao, H. Chen, and M. Savvides.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tao_Boosting_Transductive_Few-Shot_Fine-Tuning_With_Margin-Based_Uncertainty_Weighting_and_Probability_CVPR_2023_paper.pdf)

1. **Hint-Aug: Drawing Hints From Foundation Vision Transformers Towards Boosted Few-Shot Parameter-Efficient Tuning,** in CVPR, 2023.
*Z. Yu, S. Wu, Y. Fu, S. Zhang, and Y. C. Lin.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Hint-Aug_Drawing_Hints_From_Foundation_Vision_Transformers_Towards_Boosted_Few-Shot_CVPR_2023_paper.pdf)
[code](https://github.com/GATECH-EIC/Hint-Aug)

1. **ProD: Prompting-to-Disentangle Domain Knowledge for Cross-Domain Few-Shot Image Classification,** in CVPR, 2023.
*T. Ma, Y. Sun, Z. Yang, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_ProD_Prompting-To-Disentangle_Domain_Knowledge_for_Cross-Domain_Few-Shot_Image_Classification_CVPR_2023_paper.pdf)

1. **Few-Shot Learning With Visual Distribution Calibration and Cross-Modal Distribution Alignment,** in CVPR, 2023.
*R. Wang, H. Zheng, X. Duan, J. Liu, Y. Lu, T. Wang, S. Xu, and B. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Few-Shot_Learning_With_Visual_Distribution_Calibration_and_Cross-Modal_Distribution_Alignment_CVPR_2023_paper.pdf)
[code](https://gitee.com/mindspore/models/tree/master/research/cv)

1. **MetricPrompt: Prompting Model as a Relevance Metric for Few-Shot Text Classification,** in KDD, 2023.
*H. Dong, W. Zhang, and W. Che.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599430)
[code](https://github.com/Dousia/MetricPrompt)

1. **Efficient Training of Language Models Using Few-Shot Learning,** in ICML, 2023.
*S. J. Reddi, S. Miryoosefi, S. Karp, S. Krishnan, S. Kale, S. Kim, and S. Kumar.*
[paper](https://proceedings.mlr.press/v202/j-reddi23a/j-reddi23a.pdf)

1. **Multitask Pre-Training of Modular Prompt for Chinese Few-Shot Learning,** in ACL, 2023.
*T. Sun, Z. He, Q. Zhu, X. Qiu, and X. Huang.*
[paper](https://aclanthology.org/2023.acl-long.625.pdf)
[code](https://github.com/Hzfinfdu/MPMP)

1. **Cold-Start Data Selection for Better Few-Shot Language Model Fine-Tuning: A Prompt-Based Uncertainty Propagation Approach,** in ACL, 2023.
*Y. Yu, R. Zhang, R. Xu, J. Zhang, J. Shen, and C. Zhang.*
[paper](https://aclanthology.org/2023.acl-long.141.pdf)
[code](https://github.com/yueyu1030/Patron)

1. **Instruction Induction: From Few Examples to Natural Language Task Descriptions,** in ACL, 2023.
*O. Honovich, U. Shaham, S. R. Bowman, and O. Levy.*
[paper](https://aclanthology.org/2023.acl-long.108.pdf)
[code](https://github.com/orhonovich/instruction-induction)

1. **Few-Shot Adaptation Works With Unpredictable Data,** in ACL, 2023.
*J. S. Chan, M. Pieler, J. Jao, J. Scheurer, and E. Perez.*
[paper](https://aclanthology.org/2023.acl-long.102.pdf)

1. **Hierarchical Verbalizer for Few-Shot Hierarchical Text Classification,** in ACL, 2023.
*K. Ji, Y. Lian, J. Gao, and B. Wang.*
[paper](https://aclanthology.org/2023.acl-long.164.pdf)
[code](https://github.com/1KE-JI/HierVerb)

1. **Black Box Few-Shot Adaptation for Vision-Language Models,** in ICCV, 2023.
*Y. Ouali, A. Bulat, B. Matinez, and G. Tzimiropoulos.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.pdf)
[code](https://github.com/saic-fi/LFA)

1. **Read-Only Prompt Optimization for Vision-Language Few-Shot Learning,** in ICCV, 2023.
*D. Lee, S. Song, J. Suh, J. Choi, S. Lee, and H. J. Kim.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Read-only_Prompt_Optimization_for_Vision-Language_Few-shot_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/mlvlab/RPO)

1. **Not All Features Matter: Enhancing Few-Shot CLIP With Adaptive Prior Refinement,** in ICCV, 2023.
*X. Zhu, R. Zhang, B. He, A. Zhou, D. Wang, B. Zhao, and P. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Not_All_Features_Matter_Enhancing_Few-shot_CLIP_with_Adaptive_Prior_ICCV_2023_paper.pdf)
[code](https://github.com/yangyangyang127/APE)

1. **One-Shot Generative Domain Adaptation,** in ICCV, 2023.
*C. Yang, Y. Shen, Z. Zhang, Y. Xu, J. Zhu, Z. Wu, and B. Zhou.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_One-Shot_Generative_Domain_Adaptation_ICCV_2023_paper.pdf)
[code](https://genforce.github.io/genda/)

1. **Smoothness Similarity Regularization for Few-Shot GAN Adaptation,** in ICCV, 2023.
*V. Sushko, R. Wang, and J. Gall.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Sushko_Smoothness_Similarity_Regularization_for_Few-Shot_GAN_Adaptation_ICCV_2023_paper.pdf)

1. **Task-Aware Adaptive Learning for Cross-Domain Few-Shot Learning,** in ICCV, 2023.
*Y. Guo, R. Du, Y. Dong, T. Hospedales, Y. Song, and Z. Ma.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Task-aware_Adaptive_Learning_for_Cross-domain_Few-shot_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/PRIS-CV/TA2-Net)

1. **Defending Pre-Trained Language Models as Few-Shot Learners Against Backdoor Attacks,** in NeurIPS, 2023.
*Z. Xi, T. Du, C. Li, R. Pang, S. Ji, J. Chen, F. Ma, and T. Wang.*
[paper](https://openreview.net/attachment?id=GqXbfVmEPW&name=pdf)
[code](https://github.com/zhaohan-xi/PLM-prompt-defense)

1. **FD-Align: Feature Discrimination Alignment for Fine-Tuning Pre-Trained Models in Few-Shot Learning,** in NeurIPS, 2023.
*K. Song, H. Ma, B. Zou, H. Zhang, and W. Huang.*
[paper](https://openreview.net/attachment?id=shXnfALjuH&name=pdf)
[code](https://github.com/skingorz/FD-Align)

1. **Fairness-Guided Few-Shot Prompting for Large Language Models,** in NeurIPS, 2023.
*H. Ma, C. Zhang, Y. Bian, L. Liu, Z. Zhang, P. Zhao, S. Zhang, H. Fu, Q. Hu, and B. Wu.*
[paper](https://openreview.net/attachment?id=D8oHQ2qSTj&name=pdf)

1. **Meta-Adapter: An Online Few-Shot Learner for Vision-Language Model,** in NeurIPS, 2023.
*C. Cheng, L. Song, R. Xue, H. Wang, H. Sun, Y. Ge, and Y. Shan.*
[paper](https://openreview.net/attachment?id=Ts0d8PvTeB&name=pdf)

1. **Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning,** in NeurIPS, 2023.
*X. Shi, S. Xue, K. Wang, F. Zhou, J. Y. Zhang, J. ZHOU, C. Tan, and H. Mei.*
[paper](https://openreview.net/attachment?id=aW9BqtRQkh&name=pdf)
[code](https://github.com/iLampard/lamp)

1. **ExPT: Synthetic Pretraining for Few-Shot Experimental Design,** in NeurIPS, 2023.
*T. Nguyen, S. Agrawal, and A. Grover.*
[paper](https://openreview.net/attachment?id=7qfkImn0dL&name=pdf)
[code](https://github.com/tung-nd/ExPT)

1. **LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning,** in NeurIPS, 2023.
*A. Miyai, Q. Yu, G. Irie, and K. Aizawa.*
[paper](https://openreview.net/attachment?id=UjtiLdXGMC&name=pdf)
[code](https://github.com/AtsuMiyai/LoCoOp)

1. **Embroid: Unsupervised Prediction Smoothing Can Improve Few-Shot Classification,** in NeurIPS, 2023.
*N. Guha, M. F. Chen, K. Bhatia, A. Mirhoseini, F. Sala, and C. Re.*
[paper](https://openreview.net/attachment?id=4iMpwAlza1&name=pdf)

1. **Domain Re-Modulation for Few-Shot Generative Domain Adaptation,** in NeurIPS, 2023.
*Y. Wu, Z. Li, C. Wang, H. Zheng, S. Zhao, B. Li, and D. Tao.*
[paper](https://openreview.net/attachment?id=jown9RvYn7&name=pdf)
[code](https://github.com/wuyi2020/DoRM)

1. **Focus Your Attention When Few-Shot Classification,** in NeurIPS, 2023.
*H. Wang, S. Jie, and Z. Deng.*
[paper](https://openreview.net/attachment?id=uFlE0qgtRO&name=pdf)
[code](https://github.com/Haoqing-Wang/FORT)

1. **The Effect of Diversity in Meta-Learning,** in AAAI, 2023.
*R. Kumar, T. Deleu, and Y. Bengio.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26012/25784)
[code](https://github.com/RamnathKumar181/Task-Diversity-meta-learning)

1. **FEditNet: Few-Shot Editing of Latent Semantics in GAN Spaces,** in AAAI, 2023.
*M. Xia, Y. Shu, Y. Wang, Y.-K. Lai, Q. Li, P. Wan, Z. Wang, and Y.-J. Liu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25394/25166)
[code](https://github.com/THU-LYJ-Lab/FEditNet)

1. **Better Generalized Few-Shot Learning Even Without Base Data.,** in AAAI, 2023.
*S.-W. Kim, and D.-W. Choi.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25999/25771)
[code](https://github.com/bigdata-inha/Zero-Base-GFSL)

1. **Prompt-Augmented Linear Probing: Scaling Beyond the Limit of Few-Shot in-Context Learners,** in AAAI, 2023.
*H. Cho, H. J. Kim, J. Kim, S.-W. Lee, S.-g. Lee, K. M. Yoo, and T. Kim.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26495/26267)

1. **Anchoring Fine-Tuning of Sentence Transformer With Semantic Label Information for Efficient Truly Few-Shot Classification,** in EMNLP, 2023.
*A. Pauli, L. Derczynski, and I. Assent.*
[paper](https://aclanthology.org/2023.emnlp-main.692.pdf)
[code](https://github.com/AmaliePauli/AncSetfit)

1. **Skill-Based Few-Shot Selection for in-Context Learning,** in EMNLP, 2023.
*S. An, B. Zhou, Z. Lin, Q. Fu, B. Chen, N. Zheng, W. Chen, and J.-G. Lou.*
[paper](https://aclanthology.org/2023.emnlp-main.831.pdf)

1. **Transductive Learning for Textual Few-Shot Classification in API-based Embedding Models,** in EMNLP, 2023.
*P. Colombo, V. Pellegrain, M. Boudiaf, M. Tami, V. Storchan, I. Ayed, and P. Piantanida.*
[paper](https://aclanthology.org/2023.emnlp-main.257.pdf)

1. **AdaSent: Efficient Domain-Adapted Sentence Embeddings for Few-Shot Classification,** in EMNLP, 2023.
*Y. Huang, K. Wang, S. Dutta, R. Patel, G. Glavaš, and I. Gurevych.*
[paper](https://aclanthology.org/2023.emnlp-main.208.pdf)
[code](https://github.com/UKPLab/AdaSent)

1. **A Hard-to-Beat Baseline for Training-Free CLIP-based Adaptation,** in ICLR, 2024.
*Z. Wang, J. Liang, L. Sheng, R. He, Z. Wang, and T. Tan.*
[paper](https://openreview.net/attachment?id=Js5PJPHDyY&name=pdf)

1. **Group Preference Optimization: Few-Shot Alignment of Large Language Models,** in ICLR, 2024.
*S. Zhao, J. Dang, and A. Grover.*
[paper](https://openreview.net/attachment?id=DpFeMH4l8Q&name=pdf)

1. **Consistency-Guided Prompt Learning for Vision-Language Models,** in ICLR, 2024.
*S. Roy, and A. Etemad.*
[paper](https://openreview.net/attachment?id=wsRXwlwx4w&name=pdf)
[code](https://github.com/ShuvenduRoy/CoPrompt)

1. **BayesPrompt: Prompting Large-Scale Pre-Trained Language Models on Few-Shot Inference via Debiased Domain Abstraction,** in ICLR, 2024.
*J. Li, F. Song, Y. Jin, W. Qiang, C. Zheng, F. Sun, and H. Xiong.*
[paper](https://openreview.net/attachment?id=DmD1wboID9&name=pdf)

1. **Neural Fine-Tuning Search for Few-Shot Learning,** in ICLR, 2024.
*P. Eustratiadis, Ł. Dudziak, D. Li, and T. Hospedales.*
[paper](https://openreview.net/pdf?id=T7YV5UZKBc)

1. **DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-Tuning,** in ICLR, 2024.
*Z. Shi, and A. Lipani.*
[paper](https://openreview.net/attachment?id=KjegfPGRde&name=pdf)
[code](https://github.com/ZhengxiangShi/DePT)

1. **Few-Shot Hybrid Domain Adaptation of Image Generator,** in ICLR, 2024.
*H. Li, Y. Liu, L. Xia, Y. Lin, T. Zheng, Z. Yang, W. Wang, X. Zhong, X. Ren, and X. He.*
[paper](https://openreview.net/attachment?id=FE2e8664Sl&name=pdf)

1. **Bayesian Exploration of Pre-Trained Models for Low-Shot Image Classification,** in CVPR, 2024.
*Y. Miao, Y. Lei, F. Zhou, and Z. Deng.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02251)

1. **Large Language Models are Good Prompt Learners for Low-Shot Image Classification,** in CVPR, 2024.
*Z. Zheng, J. Wei, X. Hu, H. Zhu, and R. Nevatia.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02688)

1. **A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models,** in CVPR, 2024.
*J. Silva-Rodríguez, S. Hajimiri, I. B. Ayed, and J. Dolz.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02235)

1. **Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation for Cross-Domain Few-Shot Learning,** in CVPR, 2024.
*R. Perera, and S. K. Halgamuge.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02246)

1. **AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning,** in CVPR, 2024.
*Y. Tang, Z. Lin, Q. Wang, P. Zhu, and Q. Hu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02201)

1. **Simple Semantic-Aided Few-Shot Learning,** in CVPR, 2024.
*H. Zhang, J. Xu, S. Jiang, and Z. He.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02701)

1. **PELMS: Pre-training for Effective Low-Shot Multi-Document Summarization,** in NAACL, 2024.
*J. Peper, W. Qiu, and L. Wang.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.423)

1. **Adaptive Cross-lingual Text Classification through In-Context One-Shot Demonstrations,** in NAACL, 2024.
*E. Villa-Cueva, A. P. López-Monroy, F. Sánchez-Vega, and T. Solorio.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.460)

1. **LLMs Are Few-Shot In-Context Low-Resource Language Learners,** in NAACL, 2024.
*S. Cahyawijaya, H. Lovenia, and P. Fung.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.24)

1. **TrojFSP: Trojan Insertion in Few-shot Prompt Tuning,** in NAACL, 2024.
*M. Zheng, J. Xue, X. Chen, Y. Wang, Q. Lou, and L. Jiang.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.64)

1. **PromptFix: Few-shot Backdoor Removal via Adversarial Prompt Tuning,** in NAACL, 2024.
*T. Zhang, Z. Xi, T. Wang, P. Mitra, and J. Chen.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.177)

1. **Take One Step at a Time to Know Incremental Utility of Demonstration: An Analysis on Reranking for Few-Shot In-Context Learning,** in NAACL, 2024.
*K. Hashimoto, K. Raman, and M. Bendersky.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.221)

1. **Leveraging Normalization Layer in Adapters with Progressive Learning and Adaptive Distillation for Cross-Domain Few-Shot Learning,** in AAAI, 2024.
*Y. Yang, T. Kim, and S.-Y. Yun.*
[paper](https://doi.org/10.1609/aaai.v38i15.29573)

1. **Few-Shot Learning via Repurposing Ensemble of Black-Box Models,** in AAAI, 2024.
*M. Hoang, and T. N. Hoang.*
[paper](https://doi.org/10.1609/aaai.v38i11.29137)

1. **Task-Adaptive Prompted Transformer for Cross-Domain Few-Shot Learning,** in AAAI, 2024.
*J. Wu, X. Liu, X. Yin, T. Zhang, and Y. Zhang.*
[paper](https://doi.org/10.1609/aaai.v38i6.28416)

1. **One-Shot Learning as Instruction Data Prospector for Large Language Models,** in ACL, 2024.
*Y. Li, B. Hui, X. Xia, J. Yang, M. Yang, L. Zhang, S. Si, L.-H. Chen, J. Liu, T. Liu, F. Huang, and Y. Li.*
[paper](https://aclanthology.org/2024.acl-long.252)

1. **Strong Baselines for Parameter-Efficient Few-Shot Fine-Tuning,** in AAAI, 2024.
*S. Basu, S. X. Hu, D. Massiceti, and S. Feizi.*
[paper](https://doi.org/10.1609/aaai.v38i10.28978)

1. **Pushing the Limit of Fine-Tuning for Few-Shot Learning: Where Feature Reusing Meets Cross-Scale Attention,** in AAAI, 2024.
*Y.-Y. Chen, J.-W. Hsieh, X. Li, and M.-C. Chang.*
[paper](https://doi.org/10.1609/aaai.v38i10.29024)

1. **The Representation Landscape of Few-Shot Learning and Fine-Tuning in Large Language Models.,** in NeurIPS, 2024.
*D. Doimo, A. Serra, A. Ansuini, and A. Cazzaniga.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/206018a258033def63607fbdf364bd2d-Paper-Conference.pdf)

1. **MOKD: Cross-domain Finetuning for Few-shot Classification via Maximizing Optimized Kernel Dependence.,** in ICML, 2024.
*H. Tian, F. Liu, T. Liu, B. Du, Y.-m. Cheung, and B. Han.*
[paper](https://openreview.net/forum?id=fz9PaJNViP)

1. **Few-shot Adaptation to Distribution Shifts By Mixing Source and Target Embeddings.,** in ICML, 2024.
*Y. Xue, A. Payani, Y. Yang, and B. Mirzasoleiman.*
[paper](https://openreview.net/forum?id=ePDnv4xESI)

1. **Why In-Context Learning Models are Good Few-Shot Learners?,** in ICLR, 2025.
*S. Wu, Y. Wang, and Q. Yao.*
[paper](https://openreview.net/forum?id=iLUcsecZJp)

1. **LiFT: Learning to Fine-Tune via Bayesian Parameter Efficient Meta Fine-Tuning.,** in ICLR, 2025.
*M. Kim, and T. M. Hospedales.*
[paper](https://openreview.net/forum?id=7nyJBVCTGQ)

1. **Making Text Embedders Few-Shot Learners.,** in ICLR, 2025.
*C. Li, M. Qin, S. Xiao, J. Chen, K. Luo, D. Lian, Y. Shao, and Z. Liu.*
[paper](https://openreview.net/forum?id=wfLuiDjQ0u)

1. **Few-shot Personalization of LLMs with Mis-aligned Responses,** in NAACL, 2025.
*J. Kim, and Y. Yang.*
[paper](https://doi.org/10.18653/v1/2025.naacl-long.598)



### Refining Meta-learned Parameters

1. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks,** in ICML, 2017.
*C. Finn, P. Abbeel, and S. Levine.*
[paper](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf?source=post_page---------------------------)

1. **Bayesian Model-Agnostic Meta-Learning,** in NeurIPS, 2018.
*J. Yoon, T. Kim, O. Dia, S. Kim, Y. Bengio, and S. Ahn.*
[paper](https://papers.nips.cc/paper/7963-bayesian-model-agnostic-meta-learning.pdf)

1. **Probabilistic Model-Agnostic Meta-Learning,** in NeurIPS, 2018.
*C. Finn, K. Xu, and S. Levine.*
[paper](https://papers.nips.cc/paper/8161-probabilistic-model-agnostic-meta-learning.pdf)

1. **Gradient-Based Meta-Learning With Learned Layerwise Metric and Subspace,** in ICML, 2018.
*Y. Lee and S. Choi.*
[paper](http://proceedings.mlr.press/v80/lee18a/lee18a.pdf)

1. **Recasting Gradient-Based Meta-Learning as Hierarchical Bayes,** in ICLR, 2018.
*E. Grant, C. Finn, S. Levine, T. Darrell, and T. Griffiths.*
[paper](https://openreview.net/forum?id=BJ_UL-k0b)

1. **Few-Shot Human Motion Prediction via Meta-Learning,** in ECCV, 2018.
*L.-Y. Gui, Y.-X. Wang, D. Ramanan, and J. Moura.*
[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangyan_Gui_Few-Shot_Human_Motion_ECCV_2018_paper.pdf)

1. **The effects of negative adaptation in model-agnostic meta-learning,** arXiv preprint, 2018.
*T. Deleu and Y. Bengio.*
[paper](http://metalearning.ml/2018/papers/metalearn2018_paper76.pdf)

1. **Unsupervised Meta-Learning for Few-Shot Image Classification,** in NeurIPS, 2019.
*S. Khodadadeh, L. Bölöni, and M. Shah.*
[paper](https://papers.nips.cc/paper/9203-unsupervised-meta-learning-for-few-shot-image-classification.pdf)

1. **Amortized Bayesian Meta-Learning,** in ICLR, 2019.
*S. Ravi and A. Beatson.*
[paper](https://openreview.net/forum?id=rkgpy3C5tX)

1. **Meta-Learning With Latent Embedding Optimization,** in ICLR, 2019.
*A. A. Rusu, D. Rao, J. Sygnowski, O. Vinyals, R. Pascanu, S. Osindero, and R. Hadsell.* 
[paper](https://openreview.net/forum?id=BJgklhAcK7)
[code](https://github.com/deepmind/leo)

1. **LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning,** in ICML, 2019.
*H. Li, W. Dong, X. Mei, C. Ma, F. Huang, and B.-G. Hu.*
[paper](http://proceedings.mlr.press/v97/li19c/li19c.pdf)
[code](https://github.com/likesiwell/LGM-Net/)

1. **Meta R-Cnn: Towards General Solver for Instance-Level Low-Shot Learning,** in ICCV, 2019.
*X. Yan, Z. Chen, A. Xu, X. Wang, X. Liang, and L. Lin.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Meta_R-CNN_Towards_General_Solver_for_Instance-Level_Low-Shot_Learning_ICCV_2019_paper.pdf)

1. **Task Agnostic Meta-Learning for Few-Shot Learning,** in CVPR, 2019.
*M. A. Jamal, and G.-J. Qi.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jamal_Task_Agnostic_Meta-Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

1. **Meta-Transfer Learning for Few-Shot Learning,** in CVPR, 2019.
*Q. Sun, Y. Liu, T.-S. Chua, and B. Schiele.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)
[code](https://github.com/yaoyao-liu/meta-transfer-learning)

1. **Meta-Learning of Neural Architectures for Few-Shot Learning,** in CVPR, 2020.
*T. Elsken, B. Staffler, J. H. Metzen, and F. Hutter.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Elsken_Meta-Learning_of_Neural_Architectures_for_Few-Shot_Learning_CVPR_2020_paper.pdf)

1. **Attentive Weights Generation for Few Shot Learning via Information Maximization,** in CVPR, 2020.
*Y. Guo, and N.-M. Cheung.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Attentive_Weights_Generation_for_Few_Shot_Learning_via_Information_Maximization_CVPR_2020_paper.pdf)

1. **Few-Shot Open-Set Recognition Using Meta-Learning,** in CVPR, 2020.
*B. Liu, H. Kang, H. Li, G. Hua, and N. Vasconcelos.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Few-Shot_Open-Set_Recognition_Using_Meta-Learning_CVPR_2020_paper.pdf)

1. **Incremental Few-Shot Object Detection,** in CVPR, 2020.
*J.-M. Perez-Rua, X. Zhu, T. M. Hospedales, and T. Xiang.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Perez-Rua_Incremental_Few-Shot_Object_Detection_CVPR_2020_paper.pdf)

1. **Automated Relational Meta-Learning,** in ICLR, 2020.
*H. Yao, X. Wu, Z. Tao, Y. Li, B. Ding, R. Li, and Z. Li.*
[paper](https://openreview.net/pdf?id=rklp93EtwH)

1. **Meta-Learning With Warped Gradient Descent,** in ICLR, 2020.
*S. Flennerhag, A. A. Rusu, R. Pascanu, F. Visin, H. Yin, and R. Hadsell.*
[paper](https://openreview.net/pdf?id=rkeiQlBFPB)

1. **Meta-Learning Without Memorization,** in ICLR, 2020.
*M. Yin, G. Tucker, M. Zhou, S. Levine, and C. Finn.*
[paper](https://openreview.net/pdf?id=BklEFpEYwS)

1. **ES-MAML: Simple Hessian-Free Meta Learning,** in ICLR, 2020.
*X. Song, W. Gao, Y. Yang, K. Choromanski, A. Pacchiano, and Y. Tang.*
[paper](https://openreview.net/pdf?id=S1exA2NtDB)

1. **Self-Supervised Tuning for Few-Shot Segmentation,** in IJCAI, 2020.
*K. Zhu, W. Zhai, and Y. Cao.*
[paper](https://www.ijcai.org/Proceedings/2020/0142.pd)

1. **Multi-Attention Meta Learning for Few-Shot Fine-Grained Image Recognition,** in IJCAI, 2020.
*Y. Zhu, C. Liu, and S. Jiang.*
[paper](https://www.ijcai.org/Proceedings/2020/0152.pdf)

1. **An Ensemble of Epoch-Wise Empirical Bayes for Few-Shot Learning,** in ECCV, 2020.
*Y. Liu, B. Schiele, and Q. Sun.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610392.pdf)
[code](https://gitlab.mpi-klsb.mpg.de/yaoyaoliu/e3bm)

1. **Incremental Few-Shot Meta-Learning via Indirect Discriminant Alignment,** in ECCV, 2020.
*Q. Liu, O. Majumder, A. Achille, A. Ravichandran, R. Bhotika, and S. Soatto.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520664.pdf)

1. **Model-Agnostic Boundary-Adversarial Sampling for Test-Time Generalization in Few-Shot Learning,** in ECCV, 2020.
*J. Kim, H. Kim, and G. Kim.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460579.pdf)
[code](https://github.com/jaekyeom/MABAS)

1. **Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels,** in NeurIPS, 2020.
*M. Patacchiola, J. Turner, E. J. Crowley, M. O'Boyle, and A. J. Storkey.*
[paper](https://proceedings.neurips.cc/paper/2020/file/b9cfe8b6042cf759dc4c0cccb27a6737-Paper.pdf)
[code](https://github.com/BayesWatch/deep-kernel-transfer)

1. **OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification,** in NeurIPS, 2020.
*T. Jeong, and H. Kim.*
[paper](https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf)
[code](https://github.com/twj-KAIST/OOD-MAML)

1. **Unraveling Meta-Learning: Understanding Feature Representations for Few-Shot Tasks,** in ICML, 2020.
*M. Goldblum, S. Reich, L. Fowl, R. Ni, V. Cherepanova, and T. Goldstein.*
[paper](http://proceedings.mlr.press/v119/goldblum20a/goldblum20a.pdf)
[code](https://github.com/goldblum/FeatureClustering)

1. **Node Classification on Graphs With Few-Shot Novel Labels via Meta Transformed Network Embedding,** in NeurIPS, 2020.
*L. Lan, P. Wang, X. Du, K. Song, J. Tao, and X. Guan.*
[paper](https://proceedings.neurips.cc/paper/2020/file/c055dcc749c2632fd4dd806301f05ba6-Paper.pdf)

1. **Adversarially Robust Few-Shot Learning: A Meta-Learning Approach,** in NeurIPS, 2020.
*M. Goldblum, L. Fowl, and T. Goldstein.*
[paper](https://proceedings.neurips.cc/paper/2020/file/cfee398643cbc3dc5eefc89334cacdc1-Paper.pdf)
[code](https://github.com/goldblum/AdversarialQuerying)

1. **BOIL: Towards Representation Change for Few-Shot Learning,** in ICLR, 2021.
*J. Oh, H. Yoo, C. Kim, and S. Yun.*
[paper](https://openreview.net/pdf?id=umIdUL8rMH)
[code](https://github.com/flennerhag/warpgrad)

1. **Few-Shot Open-Set Recognition by Transformation Consistency,** in CVPR, 2021.
*M. Jeong, S. Choi, and C. Kim.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jeong_Few-Shot_Open-Set_Recognition_by_Transformation_Consistency_CVPR_2021_paper.pdf)

1. **Improving Generalization in Meta-Learning via Task Augmentation,** in ICML, 2021.
*H. Yao, L. Huang, L. Zhang, Y. Wei, L. Tian, J. Zou, J. Huang, and Z. Li.*
[paper](http://proceedings.mlr.press/v139/yao21b/yao21b.pdf)

1. **A Representation Learning Perspective on the Importance of Train-Validation Splitting in Meta-Learning,** in ICML, 2021.
*N. Saunshi, A. Gupta, and W. Hu.*
[paper](http://proceedings.mlr.press/v139/saunshi21a/saunshi21a.pdf)
[code](https://github.com/nsaunshi/meta_tr_val_split)

1. **Data Augmentation for Meta-Learning,** in ICML, 2021.
*R. Ni, M. Goldblum, A. Sharaf, K. Kong, and T. Goldstein.*
[paper](http://proceedings.mlr.press/v139/ni21a/ni21a.pdf)
[code](https://github.com/RenkunNi/MetaAug)

1. **Task Cooperation for Semi-Supervised Few-Shot Learning,** in AAAI, 2021.
*H. Ye, X. Li, and D.-C. Zhan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17277/17084)

1. **Conditional Self-Supervised Learning for Few-Shot Classification,** in IJCAI, 2021.
*Y. An, H. Xue, X. Zhao, and L. Zhang.*
[paper](https://www.ijcai.org/proceedings/2021/0295.pdf)

1. **Cross-Domain Few-Shot Classification via Adversarial Task Augmentation,** in IJCAI, 2021.
*H. Wang, and Z.-H. Deng.*
[paper](https://www.ijcai.org/proceedings/2021/0149.pdf)
[code](https://github.com/Haoqing-Wang/CDFSL-ATA)

1. **DReCa: A General Task Augmentation Strategy for Few-Shot Natural Language Inference,** in NAACL-HLT, 2021.
*S. Murty, T. Hashimoto, and C. D. Manning.*
[paper](https://aclanthology.org/2021.naacl-main.88.pdf)

1. **MetaXL: Meta Representation Transformation for Low-Resource Cross-Lingual Learning,** in NAACL-HLT, 2021.
*M. Xia, G. Zheng, S. Mukherjee, M. Shokouhi, G. Neubig, and A. H. Awadallah.*
[paper](https://aclanthology.org/2021.naacl-main.42.pdf)
[code](https://github.com/microsoft/MetaXL)

1. **Meta-Learning With Task-Adaptive Loss Function for Few-Shot Learning,** in ICCV, 2021.
*S. Baik, J. Choi, H. Kim, D. Cho, J. Min, and K. M. Lee.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Baik_Meta-Learning_With_Task-Adaptive_Loss_Function_for_Few-Shot_Learning_ICCV_2021_paper.pdf)
[code](https://github.com/baiksung/MeTAL)

1. **Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning,** in ICCV, 2021.
*Y. Chen, Z. Liu, H. Xu, T. Darrell, and X. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Meta-Baseline_Exploring_Simple_Meta-Learning_for_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **A Lazy Approach to Long-Horizon Gradient-Based Meta-Learning,** in ICCV, 2021.
*M. A. Jamal, L. Wang, and B. Gong.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jamal_A_Lazy_Approach_to_Long-Horizon_Gradient-Based_Meta-Learning_ICCV_2021_paper.pdf)

1. **Task-Aware Part Mining Network for Few-Shot Learning,** in ICCV, 2021.
*J. Wu, T. Zhang, Y. Zhang, and F. Wu.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Task-Aware_Part_Mining_Network_for_Few-Shot_Learning_ICCV_2021_paper.pdf)

1. **Binocular Mutual Learning for Improving Few-Shot Classification,** in ICCV, 2021.
*Z. Zhou, X. Qiu, J. Xie, J. Wu, and C. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Binocular_Mutual_Learning_for_Improving_Few-Shot_Classification_ICCV_2021_paper.pdf)
[code](https://github.com/zzqzzq/bml)

1. **Meta-Learning With an Adaptive Task Scheduler,** in NeurIPS, 2021.
*H. Yao, Y. Wang, Y. Wei, P. Zhao, M. Mahdavi, D. Lian, and C. Finn.*
[paper](https://proceedings.neurips.cc/paper/2021/file/3dc4876f3f08201c7c76cb71fa1da439-Paper.pdf)

1. **Memory Efficient Meta-Learning With Large Images,** in NeurIPS, 2021.
*J. Bronskill, D. Massiceti, M. Patacchiola, K. Hofmann, S. Nowozin, and R. Turner.*
[paper](https://proceedings.neurips.cc/paper/2021/file/cc1aa436277138f61cda703991069eaf-Paper.pdf)

1. **EvoGrad: Efficient Gradient-Based Meta-Learning and Hyperparameter Optimization,** in NeurIPS, 2021.
*O. Bohdal, Y. Yang, and T. Hospedales.*
[paper](https://proceedings.neurips.cc/paper/2021/file/bac49b876d5dfc9cd169c22ef5178ca7-Paper.pdf)

1. **Towards Enabling Meta-Learning From Target Models,** in NeurIPS, 2021.
*S. Lu, H. Ye, L. Gan, and D. Zhan.*
[paper](https://proceedings.neurips.cc/paper/2021/file/43baa6762fa81bb43b39c62553b2970d-Paper.pdf)

1. **The Role of Global Labels in Few-Shot Classification and How to Infer Them,** in NeurIPS, 2021.
*R. Wang, M. Pontil, and C. Ciliberto.*
[paper](https://proceedings.neurips.cc/paper/2021/file/e3b6fb0fd4df098162eede3313c54a8d-Paper.pdf)

1. **How to Train Your MAML to Excel in Few-Shot Classification,** in ICLR, 2022.
*H. Ye, and W. Chao.*
[paper](https://openreview.net/pdf?id=49h_IkpJtaE)
[code](https://github.com/Han-Jia/UNICORN-MAML)

1. **Meta-Learning With Fewer Tasks Through Task Interpolation,** in ICLR, 2022.
*H. Yao, L. Zhang, and C. Finn.*
[paper](https://openreview.net/pdf?id=ajXWF7bVR8d)
[code](https://github.com/huaxiuyao/MLTI)

1. **Continuous-Time Meta-Learning With Forward Mode Differentiation,** in ICLR, 2022.
*T. Deleu, D. Kanaa, L. Feng, G. Kerg, Y. Bengio, G. Lajoie, and P. Bacon.*
[paper](https://openreview.net/pdf?id=57PipS27Km)

1. **Bootstrapped Meta-Learning,** in ICLR, 2022.
*S. Flennerhag, Y. Schroecker, T. Zahavy, H. v. Hasselt, D. Silver, and S. Singh.*
[paper](https://openreview.net/pdf?id=b-ny3x071E5)

1. **Learning Prototype-Oriented Set Representations for Meta-Learning,** in ICLR, 2022.
*D. d. Guo, L. Tian, M. Zhang, M. Zhou, and H. Zha.*
[paper](https://openreview.net/pdf?id=WH6u2SvlLp4)

1. **Dynamic Kernel Selection for Improved Generalization and Memory Efficiency in Meta-Learning,** in CVPR, 2022.
*A. Chavan, R. Tiwari, U. Bamba, and D. K. Gupta.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Dynamic_Kernel_Selection_for_Improved_Generalization_and_Memory_Efficiency_in_CVPR_2022_paper.pdf)
[code](https://github.com/transmuteAI/MetaDOCK)

1. **What Matters for Meta-Learning Vision Regression Tasks?,** in CVPR, 2022.
*N. Gao, H. Ziesche, N. A. Vien, M. Volpp, and G. Neumann.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_What_Matters_for_Meta-Learning_Vision_Regression_Tasks_CVPR_2022_paper.pdf)
[code](https://github.com/boschresearch/what-matters-for-meta-learning)

1. **Multidimensional Belief Quantification for Label-Efficient Meta-Learning,** in CVPR, 2022.
*D. S. Pandey, and Q. Yu.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Pandey_Multidimensional_Belief_Quantification_for_Label-Efficient_Meta-Learning_CVPR_2022_paper.pdf)

1. **Few-Shot Node Classification on Attributed Networks With Graph Meta-Learning,** in SIGIR, 2022.
*Y. Liu, M. Li, X. Li, F. Giunchiglia, X. Feng, and R. Guan.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531978)

1. **The Role of Deconfounding in Meta-Learning,** in ICML, 2022.
*Y. Jiang, Z. Chen, K. Kuang, L. Yuan, X. Ye, Z. Wang, F. Wu, and Y. Wei.*
[paper](https://proceedings.mlr.press/v162/jiang22a/jiang22a.pdf)

1. **Stochastic Deep Networks With Linear Competing Units for Model-Agnostic Meta-Learning,** in ICML, 2022.
*K. Kalais, and S. Chatzis.*
[paper](https://proceedings.mlr.press/v162/kalais22a/kalais22a.pdf)
[code](https://github.com/Kkalais/StochLWTA-ML)

1. **Efficient Variance Reduction for Meta-Learning,** in ICML, 2022.
*H. Yang, and J. T. Kwok.*
[paper](https://proceedings.mlr.press/v162/yang22g/yang22g.pdf)

1. **Subspace Learning for Effective Meta-Learning,** in ICML, 2022.
*W. Jiang, J. Kwok, and Y. Zhang.*
[paper](https://proceedings.mlr.press/v162/jiang22b/jiang22b.pdf)

1. **Robust Meta-Learning With Sampling Noise and Label Noise via Eigen-Reptile,** in ICML, 2022.
*D. Chen, L. Wu, S. Tang, X. Yun, B. Long, and Y. Zhuang.*
[paper](https://proceedings.mlr.press/v162/chen22aa/chen22aa.pdf)
[code](https://github.com/Anfeather/Eigen-Reptile)

1. **Attentional Meta-Learners for Few-Shot Polythetic Classification,** in ICML, 2022.
*B. J. Day, R. V. Torné, N. Simidjievski, and P. Lió.*
[paper](https://proceedings.mlr.press/v162/day22a/day22a.pdf)
[code](https://github.com/rvinas/polythetic_metalearning)

1. **PLATINUM: Semi-Supervised Model Agnostic Meta-Learning Using Submodular Mutual Information,** in ICML, 2022.
*C. Li, S. Kothawade, F. Chen, and R. K. Iyer.*
[paper](https://proceedings.mlr.press/v162/li22k/li22k.pdf)
[code](https://github.com/Hugo101/PLATINUM)

1. **Finding Meta Winning Ticket to Train Your MAML,** in KDD, 2022.
*D. Gao, Y. Xie, Z. Zhou, Z. Wang, Y. Li, and B. Ding.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539467)

1. **P-Meta: Towards on-Device Deep Model Adaptation,** in KDD, 2022.
*Z. Qu, Z. Zhou, Y. Tong, and L. Thiele.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539293)

1. **FAITH: Few-Shot Graph Classification With Hierarchical Task Graphs,** in IJCAI, 2022.
*S. Wang, Y. Dong, X. Huang, C. Chen, and J. Li.*
[paper](https://www.ijcai.org/proceedings/2022/0317.pdf)
[code](https://github.com/SongW-SW/FAITH)

1. **Meta-Learning Fast Weight Language Models,** in EMNLP, 2022.
*K. Clark, K. Guu, M.-W. Chang, P. Pasupat, G. Hinton, and M. Norouzi.*
[paper](https://aclanthology.org/2022.emnlp-main.661.pdf)

1. **Understanding Benign Overfitting in Gradient-Based Meta Learning,** in NeurIPS, 2022.
*L. Chen, S. Lu, and T. Chen.*
[paper](https://openreview.net/pdf?id=oW4Zz0zlbFF)

1. **Meta-Learning With Self-Improving Momentum Target,** in NeurIPS, 2022.
*J. Tack, J. Park, H. Lee, J. Lee, and J. Shin.*
[paper](https://github.com/jihoontack/SiMT)

1. **Adversarial Task Up-Sampling for Meta-Learning,** in NeurIPS, 2022.
*Y. Wu, L.-K. Huang, and Y. Wei.*
[paper](https://openreview.net/pdf?id=pFqgUJxXXz)

1. **PAC Prediction Sets for Meta-Learning,** in NeurIPS, 2022.
*S. Park, E. Dobriban, I. Lee, and O. Bastani.*
[paper](https://openreview.net/pdf?id=s6ygs1UCOw1)

1. **A Contrastive Rule for Meta-Learning,** in NeurIPS, 2022.
*N. Zucchet, S. Schug, J. V. Oswald, D. Zhao, and J. Sacramento.*
[paper](https://openreview.net/pdf?id=NIJFp_n4MXt)
[code](https://github.com/smonsays/contrastive-meta-learning)

1. **On Enforcing Better Conditioned Meta-Learning for Rapid Few-Shot Adaptation,** in NeurIPS, 2022.
*M. Hiller, M. Harandi, and T. Drummond.*
[paper](https://openreview.net/pdf?id=G6cJsOOx2R3)

1. **Conditional Meta-Learning of Linear Representations,** in NeurIPS, 2022.
*G. Denevi, m. pontil, and C. Ciliberto.*
[paper](https://openreview.net/pdf?id=0Uejkm1GB1U)

1. **Meta-Ticket: Finding Optimal Subnetworks for Few-Shot Learning Within Randomly Initialized Neural Networks,** in NeurIPS, 2022.
*D. Chijiwa, S. Yamaguchi, A. Kumagai, and Y. Ida.*
[paper](https://openreview.net/pdf?id=Cr4_3ptitj)
[code](https://www.catalyzex.com/paper/arxiv:2205.15619/code)

1. **MetaNODE: Prototype Optimization as a Neural ODE for Few-Shot Learning,** in AAAI, 2022.
*B. Zhang, X. Li, S. Feng, Y. Ye, and R. Ye.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20885/20644)

1. **A Nested Bi-Level Optimization Framework for Robust Few Shot Learning,** in AAAI, 2022.
*K. Killamsetty, C. Li, C. Zhao, F. Chen, and R. K. Iyer.*
[paper](https://arxiv.org/pdf/2011.06782.pdf)

1. **Enhancing Meta Learning via Multi-Objective Soft Improvement Functions,** in ICLR, 2023.
*R. Yu, W. Chen, X. Wang, and J. Kwok.*
[paper](https://openreview.net/pdf?id=hCmjBJeGXcu)

1. **Understanding Train-Validation Split in Meta-Learning With Neural Networks,** in ICLR, 2023.
*X. Zuo, Z. Chen, H. Yao, Y. Cao, and Q. Gu.*
[paper](https://openreview.net/pdf?id=JVlyfHEEm0k)

1. **Bi-Level Meta-Learning for Few-Shot Domain Generalization,** in CVPR, 2023.
*X. Qin, X. Song, and S. Jiang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Qin_Bi-Level_Meta-Learning_for_Few-Shot_Domain_Generalization_CVPR_2023_paper.pdf)

1. **SHOT: Suppressing the Hessian Along the Optimization Trajectory for Gradient-Based Meta-Learning,** in NeurIPS, 2023.
*J. Lee, J. Yoo, and N. Kwak.*
[paper](https://openreview.net/attachment?id=PXsqbAjpQd&name=pdf)
[code](https://github.com/JunHoo-Lee/SHOT)

1. **Meta-AdaM: An Meta-Learned Adaptive Optimizer With Momentum for Few-Shot Learning,** in NeurIPS, 2023.
*S. Sun, and H. Gao.*
[paper](https://openreview.net/attachment?id=d85pPNBHLt&name=pdf)

1. **ESPT: A Self-Supervised Episodic Spatial Pretext Task for Improving Few-Shot Learning,** in AAAI, 2023.
*Y. Rong, X. Lu, Z. Sun, Y. Chen, and S. Xiong.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26148/25920)
[code](https://github.com/Whut-YiRong/ESPT)

1. **Scalable Bayesian Meta-Learning Through Generalized Implicit Gradients,** in AAAI, 2023.
*Y. Zhang, B. Li, S. Gao, and G. B. Giannakis.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26337/26109)
[code](https://github.com/zhangyilang/iBaML)

1. **A Hierarchical Bayesian Model for Few-Shot Meta Learning,** in ICLR, 2024.
*M. Kim, and T. Hospedales.*
[paper](https://openreview.net/attachment?id=mQ72XRfYRZ&name=pdf)

1. **First-Order ANIL Provably Learns Representations Despite Overparametrisation,** in ICLR, 2024.
*O. K. Yüksel, E. Boursier, and N. Flammarion.*
[paper](https://openreview.net/attachment?id=if2vRbS8Ew&name=pdf)

1. **Meta-Learning Priors Using Unrolled Proximal Neural Networks,** in ICLR, 2024.
*Y. Zhang, and G. B. Giannakis.*
[paper](https://openreview.net/attachment?id=b3Cu426njo&name=pdf)

1. **MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning,** in AAAI, 2024.
*B. Zhang, C. Luo, D. Yu, X. Li, H. Lin, Y. Ye, and B. Zhang.*
[paper](https://doi.org/10.1609/aaai.v38i15.29608)

1. **Dual-Level Curriculum Meta-Learning for Noisy Few-Shot Learning Tasks,** in AAAI, 2024.
*X. Que, and Q. Yu.*
[paper](https://doi.org/10.1609/aaai.v38i13.29392)

1. **One Meta-tuned Transformer is What You Need for Few-shot Learning.,** in ICML, 2024.
*X. Yang, H. Yao, and Y. Wei.*
[paper](https://openreview.net/forum?id=01ahsMovBx)

1. **Unleashing the Power of Meta-tuning for Few-shot Generalization Through Sparse Interpolated Experts.,** in ICML, 2024.
*S. Chen, J. Tack, Y. Yang, Y. W. Teh, J. R. Schwarz, and Y. Wei.*
[paper](https://openreview.net/forum?id=QhHMx51ir6)

1. **Enabling Few-Shot Learning with PID Control: A Layer Adaptive Optimizer.,** in ICML, 2024.
*L. Yu, X. Li, P. Zhang, Z. Zhang, and F. Dunkin.*
[paper](https://openreview.net/forum?id=LabSWooau0)

1. **Task-Specific Preconditioner for Cross-Domain Few-Shot Learning,** in AAAI, 2025.
*S. Kang, J. Park, W. Lee, and W. Rhee.*
[paper](https://doi.org/10.1609/aaai.v39i17.33953)

### Learning Search Steps

1. **Optimization as a Model for Few-Shot Learning,** in ICLR, 2017.
*S. Ravi and H. Larochelle.*
[paper](https://openreview.net/forum?id=rJY0-Kcll)
[code](https://github.com/twitter/meta-learning-lstm)

1. **Meta Navigator: Search for a Good Adaptation Policy for Few-Shot Learning,** in ICCV, 2021.
*C. Zhang, H. Ding, G. Lin, R. Li, C. Wang, and C. Shen.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Meta_Navigator_Search_for_a_Good_Adaptation_Policy_for_Few-Shot_ICCV_2021_paper.pdf)


## [Applications](#content)


### Computer Vision

1. **Learning Robust Visual-Semantic Embeddings,** in CVPR, 2017.
*Y.-H. Tsai, L.-K. Huang, and R. Salakhutdinov.*
[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tsai_Learning_Robust_Visual-Semantic_ICCV_2017_paper.pdf)

1. **One-Shot Action Localization by Learning Sequence Matching Network,** in CVPR, 2018.
*H. Yang, X. He, and F. Porikli.*
[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_One-Shot_Action_Localization_CVPR_2018_paper.pdf)

1. **Few-Shot Video-to-Video Synthesis,** in NeurIPS, 2019.
*T.-C. Wang, M.-Y. Liu, A. Tao, G. Liu, J. Kautz, and B. Catanzaro.*
[paper](https://papers.nips.cc/paper/8746-few-shot-video-to-video-synthesis.pdf)
[code](https://github.com/NVlabs/few-shot-vid2vid)

1. **Few-Shot Object Detection via Feature Reweighting,** in ICCV, 2019.
*B. Kang, Z. Liu, X. Wang, F. Yu, J. Feng, and T. Darrell.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.pdf)
[code](https://github.com/bingykang/Fewshot_Detection)

1. **Few-Shot Unsupervised Image-to-Image Translation,** in ICCV, 2019.
*M.-Y. Liu, X. Huang, A. Mallya, T. Karras, T. Aila, J. Lehtinen, and J. Kautz.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Few-Shot_Unsupervised_Image-to-Image_Translation_ICCV_2019_paper.pdf)
[code](https://github.com/NVlabs/FUNIT)

1. **Feature Weighting and Boosting for Few-Shot Segmentation,** in ICCV, 2019.
*K. Nguyen, and S. Todorovic.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Feature_Weighting_and_Boosting_for_Few-Shot_Segmentation_ICCV_2019_paper.pdf)

1. **Few-Shot Adaptive Gaze Estimation,** in ICCV, 2019.
*S. Park, S. D. Mello, P. Molchanov, U. Iqbal, O. Hilliges, and J. Kautz.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf)

1. **AMP: Adaptive Masked Proxies for Few-Shot Segmentation,** in ICCV, 2019.
*M. Siam, B. N. Oreshkin, and M. Jagersand.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Siam_AMP_Adaptive_Masked_Proxies_for_Few-Shot_Segmentation_ICCV_2019_paper.pdf)
[code](https://github.com/MSiam/AdaptiveMaskedProxies)

1. **Few-Shot Generalization for Single-Image 3D Reconstruction via Priors,** in ICCV, 2019.
*B. Wallace, and B. Hariharan.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wallace_Few-Shot_Generalization_for_Single-Image_3D_Reconstruction_via_Priors_ICCV_2019_paper.pdf)

1. **Few-Shot Adversarial Learning of Realistic Neural Talking Head Models,** in ICCV, 2019.
*E. Zakharov, A. Shysheya, E. Burkov, and V. Lempitsky.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zakharov_Few-Shot_Adversarial_Learning_of_Realistic_Neural_Talking_Head_Models_ICCV_2019_paper.pdf)
[code](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models)

1. **Pyramid Graph Networks With Connection Attentions for Region-Based One-Shot Semantic Segmentation,** in ICCV, 2019.
*C. Zhang, G. Lin, F. Liu, J. Guo, Q. Wu, and R. Yao.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Pyramid_Graph_Networks_With_Connection_Attentions_for_Region-Based_One-Shot_Semantic_ICCV_2019_paper.pdf)

1. **Time-Conditioned Action Anticipation in One Shot,** in CVPR, 2019.
*Q. Ke, M. Fritz, and B. Schiele.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ke_Time-Conditioned_Action_Anticipation_in_One_Shot_CVPR_2019_paper.pdf)

1. **Few-Shot Learning With Localization in Realistic Settings,** in CVPR, 2019.
*D. Wertheimer, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wertheimer_Few-Shot_Learning_With_Localization_in_Realistic_Settings_CVPR_2019_paper.pdf)
[code](https://github.com/daviswer/fewshotlocal)

1. **Improving Few-Shot User-Specific Gaze Adaptation via Gaze Redirection Synthesis,** in CVPR, 2019.
*Y. Yu, G. Liu, and J.-M. Odobez.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Improving_Few-Shot_User-Specific_Gaze_Adaptation_via_Gaze_Redirection_Synthesis_CVPR_2019_paper.pdf)

1. **CANet: Class-Agnostic Segmentation Networks With Iterative Refinement and Attentive Few-Shot Learning,** in CVPR, 2019.
*C. Zhang, G. Lin, F. Liu, R. Yao, and C. Shen.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_CANet_Class-Agnostic_Segmentation_Networks_With_Iterative_Refinement_and_Attentive_Few-Shot_CVPR_2019_paper.pdf)
[code](https://github.com/icoz69/CaNet)

1. **Multi-Level Semantic Feature Augmentation for One-Shot Learning,** in TIP, 2019.
*Z. Chen, Y. Fu, Y. Zhang, Y.-G. Jiang, X. Xue, and L. Sigal.*
[paper](https://arxiv.org/abs/1804.05298)
[code](https://github.com/tankche1/Semantic-Feature-Augmentation-in-Few-shot-Learning)

1. **3FabRec: Fast Few-Shot Face Alignment by Reconstruction,** in CVPR, 2020.
*B. Browatzki, and C. Wallraven.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Browatzki_3FabRec_Fast_Few-Shot_Face_Alignment_by_Reconstruction_CVPR_2020_paper.pdf)

1. **Few-Shot Video Classification via Temporal Alignment,** in CVPR, 2020.
*K. Cao, J. Ji, Z. Cao, C.-Y. Chang, J. C. Niebles.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf)

1. **One-Shot Adversarial Attacks on Visual Tracking With Dual Attention,** in CVPR, 2020.
*X. Chen, X. Yan, F. Zheng, Y. Jiang, S.-T. Xia, Y. Zhao, and R. Ji.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_One-Shot_Adversarial_Attacks_on_Visual_Tracking_With_Dual_Attention_CVPR_2020_paper.pdf)

1. **FGN: Fully Guided Network for Few-Shot Instance Segmentation,** in CVPR, 2020.
*Z. Fan, J.-G. Yu, Z. Liang, J. Ou, C. Gao, G.-S. Xia, and Y. Li.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_FGN_Fully_Guided_Network_for_Few-Shot_Instance_Segmentation_CVPR_2020_paper.pdf)

1. **CRNet: Cross-Reference Networks for Few-Shot Segmentation,** in CVPR, 2020.
*W. Liu, C. Zhang, G. Lin, and F. Liu.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_CRNet_Cross-Reference_Networks_for_Few-Shot_Segmentation_CVPR_2020_paper.pdf)

1. **Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition,** in CVPR, 2020.
*L. Tang, D. Wertheimer, and B. Hariharan.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_Revisiting_Pose-Normalization_for_Fine-Grained_Few-Shot_Recognition_CVPR_2020_paper.pdf)

1. **Few-Shot Learning of Part-Specific Probability Space for 3D Shape Segmentation,** in CVPR, 2020.
*L. Wang, X. Li, and Y. Fang.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Few-Shot_Learning_of_Part-Specific_Probability_Space_for_3D_Shape_Segmentation_CVPR_2020_paper.pdf)

1. **Semi-Supervised Learning for Few-Shot Image-to-Image Translation,** in CVPR, 2020.
*Y. Wang, S. Khan, A. Gonzalez-Garcia, J. van de Weijer, and F. S. Khan.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Semi-Supervised_Learning_for_Few-Shot_Image-to-Image_Translation_CVPR_2020_paper.pdf)

1. **Multi-Domain Learning for Accurate and Few-Shot Color Constancy,** in CVPR, 2020.
*J. Xiao, S. Gu, and L. Zhang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xiao_Multi-Domain_Learning_for_Accurate_and_Few-Shot_Color_Constancy_CVPR_2020_paper.pdf)

1. **One-Shot Domain Adaptation for Face Generation,** in CVPR, 2020.
*C. Yang, and S.-N. Lim.*
[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_One-Shot_Domain_Adaptation_for_Face_Generation_CVPR_2020_paper.pdf)

1. **MetaPix: Few-Shot Video Retargeting,** in ICLR, 2020.
*J. Lee, D. Ramanan, and R. Girdhar.*
[paper](https://openreview.net/pdf?id=SJx1URNKwH)

1. **Few-Shot Human Motion Prediction via Learning Novel Motion Dynamics,** in IJCAI, 2020.
*C. Zang, M. Pei, and Y. Kong.*
[paper](https://www.ijcai.org/Proceedings/2020/0118.pdf)

1. **Shaping Visual Representations With Language for Few-Shot Classification,** in ACL, 2020.
*J. Mu, P. Liang, and N. D. Goodman.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.436.pdf)

1. **MarioNETte: Few-Shot Face Reenactment Preserving Identity of Unseen Targets,** in AAAI, 2020.
*S. Ha, M. Kersner, B. Kim, S. Seo, and D. Kim.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6721)

1. **One-Shot Learning for Long-Tail Visual Relation Detection,** in AAAI, 2020.
*W. Wang, M. Wang, S. Wang, G. Long, L. Yao, G. Qi, and Y. Chen.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6904)
[code](https://github.com/Witt-Wang/oneshot)

1. **Differentiable Meta-Learning Model for Few-Shot Semantic Segmentation,** in AAAI, 2020.
*P. Tian, Z. Wu, L. Qi, L. Wang, Y. Shi, and Y. Gao.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6887)

1. **Part-Aware Prototype Network for Few-Shot Semantic Segmentation,** in ECCV, 2020.
*Y. Liu, X. Zhang, S. Zhang, and X. He.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540137.pdf)
[code](https://github.com/Xiangyi1996/PPNet-PyTorch)

1. **Prototype Mixture Models for Few-Shot Semantic Segmentation,** in ECCV, 2020.
*B. Yang, C. Liu, B. Li, J. Jiao, and Q. Ye.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530749.pdf)
[code](https://github.com/Yang-Bob/PMMs)

1. **Few-Shot Action Recognition With Permutation-Invariant Attention,** in ECCV, 2020.
*H. Zhang, L. Zhang, X. Qi, H. Li, P. H. S. Torr, and P. Koniusz.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf)

1. **Few-Shot Compositional Font Generation With Dual Memory,** in ECCV, 2020.
*J. Cha, S. Chun, G. Lee, B. Lee, S. Kim, and H. Lee.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640715.pdf)
[code](https://github.com/clovaai/dmfont)

1. **Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild,** in ECCV, 2020.
*Y. Xiao, and R. Marlet.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620188.pdf)

1. **Few-Shot Semantic Segmentation With Democratic Attention Networks,** in ECCV, 2020.
*H. Wang, X. Zhang, Y. Hu, Y. Yang, X. Cao, and X. Zhen.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580715.pdf)

1. **Few-Shot Single-View 3-D Object Reconstruction With Compositional Priors,** in ECCV, 2020.
*M. Michalkiewicz, S. Parisot, S. Tsogkas, M. Baktashmotlagh, A. Eriksson, and E. Belilovsky.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700613.pdf)

1. **COCO-FUNIT: Few-Shot Unsupervised Image Translation With a Content Conditioned Style Encoder,** in ECCV, 2020.
*K. Saito, K. Saenko, and M. Liu.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480392.pdf)
[code](https://nvlabs.github.io/COCO-FUNIT/)

1. **Multi-Scale Positive Sample Refinement for Few-Shot Object Detection,** in ECCV, 2020.
*J. Wu, S. Liu, D. Huang, and Y. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610443.pdf)
[code](https://github.com/jiaxi-wu/MPSR)

1. **Large-Scale Few-Shot Learning via Multi-Modal Knowledge Discovery,** in ECCV, 2020.
*S. Wang, J. Yue, J. Liu, Q. Tian, and M. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550715.pdf)

1. **Graph Convolutional Networks for Learning With Few Clean and Many Noisy Labels,** in ECCV, 2020.
*A. Iscen, G. Tolias, Y. Avrithis, O. Chum, and C. Schmid.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550290.pdf)

1. **Self-Supervised Few-Shot Learning on Point Clouds,** in NeurIPS, 2020.
*C. Sharma, and M. Kaul.*
[paper](https://proceedings.neurips.cc/paper/2020/file/50c1f44e426560f3f2cdcb3e19e39903-Paper.pdf)
[code](https://github.com/charusharma1991/SSL_PointClouds)

1. **Restoring Negative Information in Few-Shot Object Detection,** in NeurIPS, 2020.
*Y. Yang, F. Wei, M. Shi, and G. Li.*
[paper](https://proceedings.neurips.cc/paper/2020/file/240ac9371ec2671ae99847c3ae2e6384-Paper.pdf)
[code](https://github.com/yang-yk/NP-RepMet)

1. **Few-Shot Image Generation With Elastic Weight Consolidation,** in NeurIPS, 2020.
*Y. Li, R. Zhang, J. Lu, and E. Shechtman.*
[paper](https://proceedings.neurips.cc/paper/2020/file/b6d767d2f8ed5d21a44b0e5886680cb9-Paper.pdf)

1. **Few-Shot Visual Reasoning With Meta-Analogical Contrastive Learning,** in NeurIPS, 2020.
*Y. Kim, J. Shin, E. Yang, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/c39e1a03859f9ee215bc49131d0caf33-Paper.pdf)

1. **CrossTransformers: Spatially-Aware Few-Shot Transfer,** in NeurIPS, 2020.
*C. Doersch, A. Gupta, and A. Zisserman.*
[paper](https://proceedings.neurips.cc/paper/2020/file/fa28c6cdf8dd6f41a657c3d7caa5c709-Paper.pdf)

1. **Make One-Shot Video Object Segmentation Efficient Again,** in NeurIPS, 2020.
*T. Meinhardt, and L. Leal-Taixé.*
[paper](https://proceedings.neurips.cc/paper/2020/file/781397bc0630d47ab531ea850bddcf63-Paper.pdf)
[code](https://github.com/dvl-tum/e-osvos)

1. **Frustratingly Simple Few-Shot Object Detection,** in ICML, 2020.
*X. Wang, T. E. Huang, J. Gonzalez, T. Darrell, and F. Yu.*
[paper](http://proceedings.mlr.press/v119/wang20j/wang20j.pdf)
[code](https://github.com/ucbdrive/few-shot-object-detection)

1. **Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation,** in NeurIPS, 2020.
*Y. Luo, P. Liu, T. Guan, J. Yu, and Y. Yang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/781397bc0630d47ab531ea850bddcf63-Paper.pdf)
[code](https://github.com/RoyalVane/ASM)

1. **Disentangling 3D Prototypical Networks for Few-Shot Concept Learning,** in ICLR, 2021.
*M. Prabhudesai, S. Lal, D. Patil, H. Tung, A. W. Harley, and K. Fragkiadaki.*
[paper](https://openreview.net/pdf?id=-Lr-u0b42he)

1. **Learning Normal Dynamics in Videos With Meta Prototype Network,** in CVPR, 2021.
*H. Lv, C. Chen, Z. Cui, C. Xu, Y. Li, and J. Yang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lv_Learning_Normal_Dynamics_in_Videos_With_Meta_Prototype_Network_CVPR_2021_paper.pdf)
[code](https://github.com/ktr-hubrt/MPN/)

1. **Learning Dynamic Alignment via Meta-Filter for Few-Shot Learning,** in CVPR, 2021.
*C. Xu, Y. Fu, C. Liu, C. Wang, J. Li, F. Huang, L. Zhang, and X. Xue.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Learning_Dynamic_Alignment_via_Meta-Filter_for_Few-Shot_Learning_CVPR_2021_paper.pdf)

1. **Delving Deep Into Many-to-Many Attention for Few-Shot Video Object Segmentation,** in CVPR, 2021.
*H. Chen, H. Wu, N. Zhao, S. Ren, and S. He.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Delving_Deep_Into_Many-to-Many_Attention_for_Few-Shot_Video_Object_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/scutpaul/DANet)

1. **Adaptive Prototype Learning and Allocation for Few-Shot Segmentation,** in CVPR, 2021.
*G. Li, V. Jampani, L. Sevilla-Lara, D. Sun, J. Kim, and J. Kim.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Adaptive_Prototype_Learning_and_Allocation_for_Few-Shot_Segmentation_CVPR_2021_paper.pdf)
[code](https://git.io/ASGNet)

1. **FAPIS: A Few-Shot Anchor-Free Part-Based Instance Segmenter,** in CVPR, 2021.
*K. Nguyen, and S. Todorovic.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_FAPIS_A_Few-Shot_Anchor-Free_Part-Based_Instance_Segmenter_CVPR_2021_paper.pdf)

1. **FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding,** in CVPR, 2021.
*B. Sun, B. Li, S. Cai, Y. Yuan, and C. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_FSCE_Few-Shot_Object_Detection_via_Contrastive_Proposal_Encoding_CVPR_2021_paper.pdf)
[code](https://github.com/MegviiDetection/FSCE)

1. **Few-Shot 3D Point Cloud Semantic Segmentation,** in CVPR, 2021.
*N. Zhao, T. Chua, and G. H. Lee.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Few-Shot_3D_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/Na-Z/attMPTI)

1. **Generalized Few-Shot Object Detection Without Forgetting,** in CVPR, 2021.
*Z. Fan, Y. Ma, Z. Li, and J. Sun.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.pdf)

1. **Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling,** in CVPR, 2021.
*Z. Huang, X. Han, J. Xu, and T. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Few-Shot_Human_Motion_Transfer_by_Personalized_Geometry_and_Texture_Modeling_CVPR_2021_paper.pdf)
[code](https://github.com/HuangZhiChao95/FewShotMotionTransfer)

1. **Labeled From Unlabeled: Exploiting Unlabeled Data for Few-Shot Deep HDR Deghosting,** in CVPR, 2021.
*K. R. Prabhakar, G. Senthil, S. Agrawal, R. V. Babu, and R. K. S. S. Gorthi.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Prabhakar_Labeled_From_Unlabeled_Exploiting_Unlabeled_Data_for_Few-Shot_Deep_HDR_CVPR_2021_paper.pdf)

1. **Few-Shot Transformation of Common Actions Into Time and Space,** in CVPR, 2021.
*P. Yang, P. Mettes, and C. G. M. Snoek.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Few-Shot_Transformation_of_Common_Actions_Into_Time_and_Space_CVPR_2021_paper.pdf)
[code](https://github.com/PengWan-Yang/few-shot-transformer)

1. **Temporal-Relational CrossTransformers for Few-Shot Action Recognition,** in CVPR, 2021.
*T. Perrett, A. Masullo, T. Burghardt, M. Mirmehdi, and D. Damen.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Perrett_Temporal-Relational_CrossTransformers_for_Few-Shot_Action_Recognition_CVPR_2021_paper.pdf)

1. **pixelNeRF: Neural Radiance Fields From One or Few Images,** in CVPR, 2021.
*A. Yu, V. Ye, M. Tancik, and A. Kanazawa.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yu_pixelNeRF_Neural_Radiance_Fields_From_One_or_Few_Images_CVPR_2021_paper.pdf)
[code](https://alexyu.net/pixelnerf)

1. **Hallucination Improves Few-Shot Object Detection,** in CVPR, 2021.
*W. Zhang, and Y. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Hallucination_Improves_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)

1. **Few-Shot Object Detection via Classification Refinement and Distractor Retreatment,** in CVPR, 2021.
*Y. Li, H. Zhu, Y. Cheng, W. Wang, C. S. Teo, C. Xiang, P. Vadakkepat, and T. H. Lee.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Few-Shot_Object_Detection_via_Classification_Refinement_and_Distractor_Retreatment_CVPR_2021_paper.pdf)

1. **Dense Relation Distillation With Context-Aware Aggregation for Few-Shot Object Detection,** in CVPR, 2021.
*H. Hu, S. Bai, A. Li, J. Cui, and L. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)
[code](https://github.com/hzhupku/DCNet)

1. **Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need?,** in CVPR, 2021.
*M. Boudiaf, H. Kervadec, Z. I. Masud, P. Piantanida, I. B. Ayed, and J. Dolz.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.pdf)
[code](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation)

1. **Few-Shot Image Generation via Cross-Domain Correspondence,** in CVPR, 2021.
*U. Ojha, Y. Li, J. Lu, A. A. Efros, Y. J. Lee, E. Shechtman, and R. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ojha_Few-Shot_Image_Generation_via_Cross-Domain_Correspondence_CVPR_2021_paper.pdf)

1. **Self-Guided and Cross-Guided Learning for Few-Shot Segmentation,** in CVPR, 2021.
*B. Zhang, J. Xiao, and T. Qin.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Self-Guided_and_Cross-Guided_Learning_for_Few-Shot_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/zbf1991/SCL)

1. **Anti-Aliasing Semantic Reconstruction for Few-Shot Semantic Segmentation,** in CVPR, 2021.
*B. Liu, Y. Ding, J. Jiao, X. Ji, and Q. Ye.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Anti-Aliasing_Semantic_Reconstruction_for_Few-Shot_Semantic_Segmentation_CVPR_2021_paper.pdf)

1. **Beyond Max-Margin: Class Margin Equilibrium for Few-Shot Object Detection,** in CVPR, 2021.
*B. Li, B. Yang, C. Liu, F. Liu, R. Ji, and Q. Ye.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Beyond_Max-Margin_Class_Margin_Equilibrium_for_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)
[code](https://github.com/Bohao-Lee/CME)

1. **Scale-Aware Graph Neural Network for Few-Shot Semantic Segmentation,** in CVPR, 2021.
*G. Xie, J. Liu, H. Xiong, and L. Shao.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Scale-Aware_Graph_Neural_Network_for_Few-Shot_Semantic_Segmentation_CVPR_2021_paper.pdf)

1. **Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection,** in CVPR, 2021.
*C. Zhu, F. Chen, U. Ahmed, Z. Shen, and M. Savvides.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Semantic_Relation_Reasoning_for_Shot-Stable_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)

1. **Accurate Few-Shot Object Detection With Support-Query Mutual Guidance and Hybrid Loss,** in CVPR, 2021.
*L. Zhang, S. Zhou, J. Guan, and J. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Accurate_Few-Shot_Object_Detection_With_Support-Query_Mutual_Guidance_and_Hybrid_CVPR_2021_paper.pdf)

1. **Transformation Invariant Few-Shot Object Detection,** in CVPR, 2021.
*A. Li, and Z. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Transformation_Invariant_Few-Shot_Object_Detection_CVPR_2021_paper.pdf)

1. **MetaHTR: Towards Writer-Adaptive Handwritten Text Recognition,** in CVPR, 2021.
*A. K. Bhunia, S. Ghose, A. Kumar, P. N. Chowdhury, A. Sain, and Y. Song.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhunia_MetaHTR_Towards_Writer-Adaptive_Handwritten_Text_Recognition_CVPR_2021_paper.pdf)

1. **What if We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels,** in CVPR, 2021.
*J. Baek, Y. Matsui, and K. Aizawa.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf)
[code](https://github.com/ku21fan/STR-Fewer-Labels)

1. **Few-Shot Font Generation With Localized Style Representations and Factorization,** in AAAI, 2021.
*S. Park, S. Chun, J. Cha, B. Lee, and H. Shim.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16340/16147)
[code](https://github.com/clovaai/lffont)

1. **Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition,** in AAAI, 2021.
*S. Huang, M. Zhang, Y. Kang, and D. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16957/16764)
[code](https://github.com/bighuang624/AGAM)

1. **One-Shot Face Reenactment Using Appearance Adaptive Normalization,** in AAAI, 2021.
*G. Yao, Y. Yuan, T. Shao, S. Li, S. Liu, Y. Liu, M. Wang, and K. Zhou.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16427/16234)


1. **StarNet: Towards Weakly Supervised Few-Shot Object Detection,** in AAAI, 2021.
*L. Karlinsky, J. Shtok, A. Alfassy, M. Lichtenstein, S. Harary, E. Schwartz, S. Doveh, P. Sattigeri, R. Feris, A. Bronstein, and R. Giryes.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16268/16075)
[code](https://github.com/leokarlin/StarNet)

1. **Progressive One-Shot Human Parsing,** in AAAI, 2021.
*H. He, J. Zhang, B. Thuraisingham, and D. Tao.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16243/16050)
[code](https://github.com/Charleshhy/One-shot-Human-Parsing)

1. **Knowledge Is Power: Hierarchical-Knowledge Embedded Meta-Learning for Visual Reasoning in Artistic Domains,** in KDD, 2021.
*W. Zheng, L. Yan, C. Gou, and F.-Y. Wang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467285)

1. **MEDA: Meta-Learning With Data Augmentation for Few-Shot Text Classification,** in IJCAI, 2021.
*P. Sun, Y. Ouyang, W. Zhang, and X.-Y. Dai.*
[paper](https://www.ijcai.org/proceedings/2021/0541.pdf)

1. **Learning Implicit Temporal Alignment for Few-Shot Video Classification,** in IJCAI, 2021.
*S. Zhang, J. Zhou, and X. He.*
[paper](https://www.ijcai.org/proceedings/2021/0181.pdf)
[code](https://github.com/tonysy/PyAction)

1. **Few-Shot Neural Human Performance Rendering From Sparse RGBD Videos,** in IJCAI, 2021.
*A. Pang, X. Chen, H. Luo, M. Wu, J. Yu, and L. Xu.*
[paper](https://www.ijcai.org/proceedings/2021/0130.pdf)

1. **Uncertainty-Aware Few-Shot Image Classification,** in IJCAI, 2021.
*Z. Zhang, C. Lan, W. Zeng, Z. Chen, and S. Chan.*
[paper](https://www.ijcai.org/proceedings/2021/0471.pdf)

1. **Few-Shot Learning With Part Discovery and Augmentation From Unlabeled Images,** in IJCAI, 2021.
*W. Chen, C. Si, W. Wang, L. Wang, Z. Wang, and T. Tan.*
[paper](https://www.ijcai.org/proceedings/2021/0313.pdf)

1. **Few-Shot Partial-Label Learning,** in IJCAI, 2021.
*Y. Zhao, G. Yu, L. Liu, Z. Yan, L. Cui, and C. Domeniconi.*
[paper](https://www.ijcai.org/proceedings/2021/0475.pdf)

1. **One-Shot Affordance Detection,** in IJCAI, 2021.
*H. Luo, W. Zhai, J. Zhang, Y. Cao, and D. Tao.*
[paper](https://www.ijcai.org/proceedings/2021/0124.pdf)

1. **DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection,** in ICCV, 2021.
*L. Qiao, Y. Zhao, Z. Li, X. Qiu, J. Wu, and C. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiao_DeFRCN_Decoupled_Faster_R-CNN_for_Few-Shot_Object_Detection_ICCV_2021_paper.pdf)

1. **Learning Meta-Class Memory for Few-Shot Semantic Segmentation,** in ICCV, 2021.
*Z. Wu, X. Shi, G. Lin, and J. Cai.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Meta-Class_Memory_for_Few-Shot_Semantic_Segmentation_ICCV_2021_paper.pdf)

1. **UVStyle-Net: Unsupervised Few-Shot Learning of 3D Style Similarity Measure for B-Reps,** in ICCV, 2021.
*P. Meltzer, H. Shayani, A. Khasahmadi, P. K. Jayaraman, A. Sanghi, and J. Lambourne.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Meltzer_UVStyle-Net_Unsupervised_Few-Shot_Learning_of_3D_Style_Similarity_Measure_for_ICCV_2021_paper.pdf)

1. **LoFGAN: Fusing Local Representations for Few-Shot Image Generation,** in ICCV, 2021.
*Z. Gu, W. Li, J. Huo, L. Wang, and Y. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_LoFGAN_Fusing_Local_Representations_for_Few-Shot_Image_Generation_ICCV_2021_paper.pdf)

1. **H3d-Net: Few-Shot High-Fidelity 3D Head Reconstruction,** in ICCV, 2021.
*E. Ramon, G. Triginer, J. Escur, A. Pumarola, J. Garcia, X. Giró-i-Nieto, and F. Moreno-Noguer.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ramon_H3D-Net_Few-Shot_High-Fidelity_3D_Head_Reconstruction_ICCV_2021_paper.pdf)

1. **Learned Spatial Representations for Few-Shot Talking-Head Synthesis,** in ICCV, 2021.
*M. Meshry, S. Suri, L. S. Davis, and A. Shrivastava.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Meshry_Learned_Spatial_Representations_for_Few-Shot_Talking-Head_Synthesis_ICCV_2021_paper.pdf)

1. **Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis,** in ICCV, 2021.
*A. Jain, M. Tancik, and P. Abbeel.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.pdf)

1. **Hypercorrelation Squeeze for Few-Shot Segmentation,** in ICCV, 2021.
*J. Min, D. Kang, and M. Cho.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Hypercorrelation_Squeeze_for_Few-Shot_Segmentation_ICCV_2021_paper.pdf)
[code](https://github.com/juhongm999/hsnet)

1. **Few-Shot Semantic Segmentation With Cyclic Memory Network,** in ICCV, 2021.
*G. Xie, H. Xiong, J. Liu, Y. Yao, and L. Shao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Few-Shot_Semantic_Segmentation_With_Cyclic_Memory_Network_ICCV_2021_paper.pdf)

1. **Simpler Is Better: Few-Shot Semantic Segmentation With Classifier Weight Transformer,** in ICCV, 2021.
*Z. Lu, S. He, X. Zhu, L. Zhang, Y. Song, and T. Xiang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Simpler_Is_Better_Few-Shot_Semantic_Segmentation_With_Classifier_Weight_Transformer_ICCV_2021_paper.pdf)
[code](https://github.com/zhiheLu/CWT-for-FSS)

1. **Unsupervised Few-Shot Action Recognition via Action-Appearance Aligned Meta-Adaptation,** in ICCV, 2021.
*J. Patravali, G. Mittal, Y. Yu, F. Li, and M. Chen.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Patravali_Unsupervised_Few-Shot_Action_Recognition_via_Action-Appearance_Aligned_Meta-Adaptation_ICCV_2021_paper.pdf)

1. **Multiple Heads Are Better Than One: Few-Shot Font Generation With Multiple Localized Experts,** in ICCV, 2021.
*S. Park, S. Chun, J. Cha, B. Lee, and H. Shim.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Multiple_Heads_Are_Better_Than_One_Few-Shot_Font_Generation_With_ICCV_2021_paper.pdf)
[code](https://github.com/clovaai/mxfont)

1. **Mining Latent Classes for Few-Shot Segmentation,** in ICCV, 2021.
*L. Yang, W. Zhuo, L. Qi, Y. Shi, and Y. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Mining_Latent_Classes_for_Few-Shot_Segmentation_ICCV_2021_paper.pdf)
[code](https://github.com/LiheYoung/MiningFSS)

1. **Partner-Assisted Learning for Few-Shot Image Classification,** in ICCV, 2021.
*J. Ma, H. Xie, G. Han, S. Chang, A. Galstyan, and W. Abd-Almageed.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ma_Partner-Assisted_Learning_for_Few-Shot_Image_Classification_ICCV_2021_paper.pdf)

1. **Hierarchical Graph Attention Network for Few-Shot Visual-Semantic Learning,** in ICCV, 2021.
*C. Yin, K. Wu, Z. Che, B. Jiang, Z. Xu, and J. Tang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yin_Hierarchical_Graph_Attention_Network_for_Few-Shot_Visual-Semantic_Learning_ICCV_2021_paper.pdf)

1. **Video Pose Distillation for Few-Shot, Fine-Grained Sports Action Recognition,** in ICCV, 2021.
*J. Hong, M. Fisher, M. Gharbi, and K. Fatahalian.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Hong_Video_Pose_Distillation_for_Few-Shot_Fine-Grained_Sports_Action_Recognition_ICCV_2021_paper.pdf)

1. **Universal-Prototype Enhancing for Few-Shot Object Detection,** in ICCV, 2021.
*A. Wu, Y. Han, L. Zhu, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Universal-Prototype_Enhancing_for_Few-Shot_Object_Detection_ICCV_2021_paper.pdf)
[code](https://github.com/amingwu/up-fsod)

1. **Query Adaptive Few-Shot Object Detection With Heterogeneous Graph Convolutional Networks,** in ICCV, 2021.
*G. Han, Y. He, S. Huang, J. Ma, and S. Chang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Han_Query_Adaptive_Few-Shot_Object_Detection_With_Heterogeneous_Graph_Convolutional_Networks_ICCV_2021_paper.pdf)

1. **Few-Shot Visual Relationship Co-Localization,** in ICCV, 2021.
*R. Teotia, V. Mishra, M. Maheshwari, and A. Mishra.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Teotia_Few-Shot_Visual_Relationship_Co-Localization_ICCV_2021_paper.pdf)
[code](https://github.com/vl2g/VRC)

1. **Shallow Bayesian Meta Learning for Real-World Few-Shot Recognition,** in ICCV, 2021.
*X. Zhang, D. Meng, H. Gouk, and T. M. Hospedales.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Shallow_Bayesian_Meta_Learning_for_Real-World_Few-Shot_Recognition_ICCV_2021_paper.pdf)
[code](https://github.com/open-debin/bayesian_mqda)

1. **Super-Resolving Cross-Domain Face Miniatures by Peeking at One-Shot Exemplar,** in ICCV, 2021.
*P. Li, X. Yu, and Y. Yang.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Super-Resolving_Cross-Domain_Face_Miniatures_by_Peeking_at_One-Shot_Exemplar_ICCV_2021_paper.pdf)

1. **Few-Shot Segmentation via Cycle-Consistent Transformer,** in NeurIPS, 2021.
*G. Zhang, G. Kang, Y. Yang, and Y. Wei.*
[paper](https://proceedings.neurips.cc/paper/2021/file/b8b12f949378552c21f28deff8ba8eb6-Paper.pdf)

1. **Generalized and Discriminative Few-Shot Object Detection via SVD-dictionary Enhancement,** in NeurIPS, 2021.
*A. WU, S. Zhao, C. Deng, and W. Liu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/325995af77a0e8b06d1204a171010b3a-Paper.pdf)

1. **Re-Ranking for Image Retrieval and Transductive Few-Shot Classification,** in NeurIPS, 2021.
*X. SHEN, Y. Xiao, S. Hu, O. Sbai, and M. Aubry.*
[paper](https://proceedings.neurips.cc/paper/2021/file/d9fc0cdb67638d50f411432d0d41d0ba-Paper.pdf)

1. **Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose,** in NeurIPS, 2021.
*A. Wang, S. Mei, A. L. Yuille, and A. Kortylewski.*
[paper](https://proceedings.neurips.cc/paper/2021/file/3a61ed715ee66c48bacf237fa7bb5289-Paper.pdf)

1. **MetaAvatar: Learning Animatable Clothed Human Models From Few Depth Images,** in NeurIPS, 2021.
*S. Wang, M. Mihajlovic, Q. Ma, A. Geiger, and S. Tang.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1680829293f2a8541efa2647a0290f88-Paper.pdf)

1. **Few-Shot Object Detection via Association and Discrimination,** in NeurIPS, 2021.
*Y. Cao, J. Wang, Y. Jin, T. Wu, K. Chen, Z. Liu, and D. Lin.*
[paper](https://proceedings.neurips.cc/paper/2021/file/8a1e808b55fde9455cb3d8857ed88389-Paper.pdf)

1. **Rectifying the Shortcut Learning of Background for Few-Shot Learning,** in NeurIPS, 2021.
*X. Luo, L. Wei, L. Wen, J. Yang, L. Xie, Z. Xu, and Q. Tian.*
[paper](https://proceedings.neurips.cc/paper/2021/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf)

1. **D2C: Diffusion-Decoding Models for Few-Shot Conditional Generation,** in NeurIPS, 2021.
*A. Sinha, J. Song, C. Meng, and S. Ermon.*
[paper](https://proceedings.neurips.cc/paper/2021/file/682e0e796084e163c5ca053dd8573b0c-Paper.pdf)

1. **Few-Shot Backdoor Attacks on Visual Object Tracking,** in ICLR, 2022.
*Y. Li, H. Zhong, X. Ma, Y. Jiang, and S. Xia.*
[paper](https://openreview.net/pdf?id=qSV5CuSaK_a)
[code](https://github.com/HXZhong1997/FSBA)

1. **Temporal Alignment Prediction for Supervised Representation Learning and Few-Shot Sequence Classification,** in ICLR, 2022.
*B. Su, and J. Wen.*
[paper](https://openreview.net/pdf?id=p3DKPQ7uaAi)
[code](https://github.com/BingSu12/TAP)

1. **Learning Non-Target Knowledge for Few-Shot Semantic Segmentation,** in CVPR, 2022.
*Y. Liu, N. Liu, Q. Cao, X. Yao, J. Han, and L. Shao.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Learning_Non-Target_Knowledge_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf)

1. **Learning What Not to Segment: A New Perspective on Few-Shot Segmentation,** in CVPR, 2022.
*C. Lang, G. Cheng, B. Tu, and J. Han.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lang_Learning_What_Not_To_Segment_A_New_Perspective_on_Few-Shot_CVPR_2022_paper.pdf)
[code](https://github.com/chunbolang/BAM)

1. **Few-Shot Keypoint Detection With Uncertainty Learning for Unseen Species,** in CVPR, 2022.
*C. Lu, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Few-Shot_Keypoint_Detection_With_Uncertainty_Learning_for_Unseen_Species_CVPR_2022_paper.pdf)



1. **Spatio-Temporal Relation Modeling for Few-Shot Action Recognition,** in CVPR, 2022.
*A. Thatipelli, S. Narayan, S. Khan, R. M. Anwer, F. S. Khan, and B. Ghanem.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Thatipelli_Spatio-Temporal_Relation_Modeling_for_Few-Shot_Action_Recognition_CVPR_2022_paper.pdf)
[code](https://github.com/Anirudh257/strm)

1. **Attribute Group Editing for Reliable Few-Shot Image Generation,** in CVPR, 2022.
*G. Ding, X. Han, S. Wang, S. Wu, X. Jin, D. Tu, and Q. Huang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Attribute_Group_Editing_for_Reliable_Few-Shot_Image_Generation_CVPR_2022_paper.pdf)
[code](https://github.com/UniBester/AGE)

1. **Few-Shot Backdoor Defense Using Shapley Estimation,** in CVPR, 2022.
*J. Guan, Z. Tu, R. He, and D. Tao.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guan_Few-Shot_Backdoor_Defense_Using_Shapley_Estimation_CVPR_2022_paper.pdf)

1. **Hybrid Relation Guided Set Matching for Few-Shot Action Recognition,** in CVPR, 2022.
*X. Wang, S. Zhang, Z. Qing, M. Tang, Z. Zuo, C. Gao, R. Jin, and N. Sang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Hybrid_Relation_Guided_Set_Matching_for_Few-Shot_Action_Recognition_CVPR_2022_paper.pdf)
[code](https://hyrsm-cvpr2022.github.io/)

1. **Label, Verify, Correct: A Simple Few Shot Object Detection Method,** in CVPR, 2022.
*P. Kaul, W. Xie, and A. Zisserman.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kaul_Label_Verify_Correct_A_Simple_Few_Shot_Object_Detection_Method_CVPR_2022_paper.pdf)

1. **InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering,** in CVPR, 2022.
*M. Kim, S. Seo, and B. Han.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_InfoNeRF_Ray_Entropy_Minimization_for_Few-Shot_Neural_Volume_Rendering_CVPR_2022_paper.pdf)

1. **A Closer Look at Few-Shot Image Generation,** in CVPR, 2022.
*Y. Zhao, H. Ding, H. Huang, and N. Cheung.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_A_Closer_Look_at_Few-Shot_Image_Generation_CVPR_2022_paper.pdf)
[code](https://github.com/mseitzer/pytorch-fid)

1. **Motion-Modulated Temporal Fragment Alignment Network for Few-Shot Action Recognition,** in CVPR, 2022.
*J. Wu, T. Zhang, Z. Zhang, F. Wu, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Motion-Modulated_Temporal_Fragment_Alignment_Network_for_Few-Shot_Action_Recognition_CVPR_2022_paper.pdf)

1. **Kernelized Few-Shot Object Detection With Efficient Integral Aggregation,** in CVPR, 2022.
*S. Zhang, L. Wang, N. Murray, and P. Koniusz.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Kernelized_Few-Shot_Object_Detection_With_Efficient_Integral_Aggregation_CVPR_2022_paper.pdf)
[code](https://github.com/ZS123-lang/KFSOD)

1. **FS6D: Few-Shot 6D Pose Estimation of Novel Objects,** in CVPR, 2022.
*Y. He, Y. Wang, H. Fan, J. Sun, and Q. Chen.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_FS6D_Few-Shot_6D_Pose_Estimation_of_Novel_Objects_CVPR_2022_paper.pdf)

1. **Look Closer to Supervise Better: One-Shot Font Generation via Component-Based Discriminator,** in CVPR, 2022.
*Y. Kong, C. Luo, W. Ma, Q. Zhu, S. Zhu, N. Yuan, and L. Jin.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_Look_Closer_To_Supervise_Better_One-Shot_Font_Generation_via_Component-Based_CVPR_2022_paper.pdf)

1. **Generalized Few-Shot Semantic Segmentation,** in CVPR, 2022.
*Z. Tian, X. Lai, L. Jiang, S. Liu, M. Shu, H. Zhao, and J. Jia.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tian_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf)
[code](https://github.com/dvlab-research/GFS-Seg)

1. **Dynamic Prototype Convolution Network for Few-Shot Semantic Segmentation,** in CVPR, 2022.
*J. Liu, Y. Bao, G. Xie, H. Xiong, J. Sonke, and E. Gavves.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Dynamic_Prototype_Convolution_Network_for_Few-Shot_Semantic_Segmentation_CVPR_2022_paper.pdf)

1. **OSOP: A Multi-Stage One Shot Object Pose Estimation Framework,** in CVPR, 2022.
*I. Shugurov, F. Li, B. Busam, and S. Ilic.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shugurov_OSOP_A_Multi-Stage_One_Shot_Object_Pose_Estimation_Framework_CVPR_2022_paper.pdf)

1. **Semantic-Aligned Fusion Transformer for One-Shot Object Detection,** in CVPR, 2022.
*Y. Zhao, X. Guo, and Y. Lu.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Semantic-Aligned_Fusion_Transformer_for_One-Shot_Object_Detection_CVPR_2022_paper.pdf)

1. **OnePose: One-Shot Object Pose Estimation Without CAD Models,** in CVPR, 2022.
*J. Sun, Z. Wang, S. Zhang, X. He, H. Zhao, G. Zhang, and X. Zhou.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_OnePose_One-Shot_Object_Pose_Estimation_Without_CAD_Models_CVPR_2022_paper.pdf)
[code](https://zju3dv.github.io/onepose/)

1. **Few-Shot Object Detection With Fully Cross-Transformer,** in CVPR, 2022.
*G. Han, J. Ma, S. Huang, L. Chen, and S. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Few-Shot_Object_Detection_With_Fully_Cross-Transformer_CVPR_2022_paper.pdf)

1. **Learning to Memorize Feature Hallucination for One-Shot Image Generation,** in CVPR, 2022.
*Y. Xie, Y. Fu, Y. Tai, Y. Cao, J. Zhu, and C. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Learning_To_Memorize_Feature_Hallucination_for_One-Shot_Image_Generation_CVPR_2022_paper.pdf)

1. **Few-Shot Font Generation by Learning Fine-Grained Local Styles,** in CVPR, 2022.
*L. Tang, Y. Cai, J. Liu, Z. Hong, M. Gong, M. Fan, J. Han, J. Liu, E. Ding, and J. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Few-Shot_Font_Generation_by_Learning_Fine-Grained_Local_Styles_CVPR_2022_paper.pdf)

1. **Balanced and Hierarchical Relation Learning for One-Shot Object Detection,** in CVPR, 2022.
*H. Yang, S. Cai, H. Sheng, B. Deng, J. Huang, X. Hua, Y. Tang, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Balanced_and_Hierarchical_Relation_Learning_for_One-Shot_Object_Detection_CVPR_2022_paper.pdf)

1. **XMP-Font: Self-Supervised Cross-Modality Pre-Training for Few-Shot Font Generation,** in CVPR, 2022.
*W. Liu, F. Liu, F. Ding, Q. He, and Z. Yi.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_XMP-Font_Self-Supervised_Cross-Modality_Pre-Training_for_Few-Shot_Font_Generation_CVPR_2022_paper.pdf)


1. **Few-Shot Head Swapping in the Wild,** in CVPR, 2022.
*C. Shu, H. Wu, H. Zhou, J. Liu, Z. Hong, C. Ding, J. Han, J. Liu, E. Ding, and J. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shu_Few-Shot_Head_Swapping_in_the_Wild_CVPR_2022_paper.pdf)

1. **Integrative Few-Shot Learning for Classification and Segmentation,** in CVPR, 2022.
*D. Kang, and M. Cho.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_Integrative_Few-Shot_Learning_for_Classification_and_Segmentation_CVPR_2022_paper.pdf)

1. **Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-Shot Learning,** in CVPR, 2022.
*Y. He, W. Liang, D. Zhao, H. Zhou, W. Ge, Y. Yu, and W. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Attribute_Surrogates_Learning_and_Spectral_Tokens_Pooling_in_Transformers_for_CVPR_2022_paper.pdf)
[code](https://github.com/StomachCold/HCTransformers)

1. **Task Discrepancy Maximization for Fine-Grained Few-Shot Classification,** in CVPR, 2022.
*S. Lee, W. Moon, and J. Heo.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Task_Discrepancy_Maximization_for_Fine-Grained_Few-Shot_Classification_CVPR_2022_paper.pdf)

1. **Channel Importance Matters in Few-Shot Image Classification,** in ICML, 2022.
*X. Luo, J. Xu, and Z. Xu.*
[paper](https://proceedings.mlr.press/v162/luo22c/luo22c.pdf)

1. **Long-Short Term Cross-Transformer in Compressed Domain for Few-Shot Video Classification,** in IJCAI, 2022.
*W. Luo, Y. Liu, B. Li, W. Hu, Y. Miao, and Y. Li.*
[paper](https://www.ijcai.org/proceedings/2022/0174.pdf)

1. **HifiHead: One-Shot High Fidelity Neural Head Synthesis With 3D Control,** in IJCAI, 2022.
*F. Zhu, J. Zhu, W. Chu, Y. Tai, Z. Xie, X. Huang, and C. Wang.*
[paper](https://www.ijcai.org/proceedings/2022/0244.pdf)
[code](https://github.com/TencentYoutuResearch/HeadSynthesis-HifHea)

1. **Iterative Few-Shot Semantic Segmentation From Image Label Text,** in IJCAI, 2022.
*H. Wang, L. Liu, W. Zhang, J. Zhang, Z. Gan, Y. Wang, C. Wang, and H. Wang.*
[paper](https://www.ijcai.org/proceedings/2022/0193.pdf)
[code](https://github.com/Whileherham/IMR-HSNet)

1. **Beyond the Prototype: Divide-and-Conquer Proxies for Few-Shot Segmentation,** in IJCAI, 2022.
*C. Lang, B. Tu, G. Cheng, and J. Han.*
[paper](https://www.ijcai.org/proceedings/2022/0143.pdf)
[code](https://github.com/chunbolang/DCP)

1. **CATrans: Context and Affinity Transformer for Few-Shot Segmentation,** in IJCAI, 2022.
*S. Zhang, T. Wu, S. Wu, and G. Guo.*
[paper](https://www.ijcai.org/proceedings/2022/0231.pdf)

1. **Masked Feature Generation Network for Few-Shot Learning,** in IJCAI, 2022.
*Y. Yu, D. Zhang, and Z. Ji.*
[paper](https://www.ijcai.org/proceedings/2022/0513.pdf)

1. **Decoupling Classifier for Boosting Few-Shot Object Detection and Instance Segmentation,** in NeurIPS, 2022.
*B.-B. Gao, X. Chen, Z. Huang, C. Nie, J. Liu, J. Lai, G. JIANG, X. Wang, and C. Wang.*
[paper](https://openreview.net/pdf?id=dVXO3Orjmxk)
[code](https://csgaobb.github.io/Projects/DCFS)

1. **Searching for Better Spatio-Temporal Alignment in Few-Shot Action Recognition,** in NeurIPS, 2022.
*Y. Cao, X. Su, Q. Tang, S. You, X. Lu, and C. Xu.*
[paper](https://openreview.net/pdf?id=IlYS1pLa9y)

1. **Feature-Proxy Transformer for Few-Shot Segmentation,** in NeurIPS, 2022.
*J.-W. Zhang, Y. Sun, Y. Yang, and W. Chen.*
[paper](https://openreview.net/pdf?id=hBaI5MY0CBz)
[code](https://github.com/Jarvis73/FPTrans)

1. **Intermediate Prototype Mining Transformer for Few-Shot Semantic Segmentation,** in NeurIPS, 2022. 
*Y. liu, N. Liu, X. Yao, J. Han,*
[paper](https://openreview.net/pdf?id=NyAJzgHLAr)
[code](https://github.com/LIUYUANWEI98/IPMT)

1. **OnePose++: Keypoint-Free One-Shot Object Pose Estimation Without CAD Models,** in NeurIPS, 2022.
*X. He, J. Sun, Y. Wang, D. Huang, H. Bao, and X. Zhou.*
[paper](https://openreview.net/pdf?id=BZ92dxDS3tO)
[code](https://zju3dv.github.io/onepose_plus_plus/)

1. **Mask Matching Transformer for Few-Shot Segmentation,** in NeurIPS, 2022.
*S. Jiao, G. Zhang, S. Navasardyan, L. Chen, Y. Zhao, Y. Wei, and H. Shi.*
[paper](https://openreview.net/pdf?id=zt4xNo0lF8W)
[code](https://github.com/Picsart-AI-Research/Mask-Matching-Transformer)

1. **Learning Dense Object Descriptors From Multiple Views for Low-Shot Category Generalization,** in NeurIPS, 2022.
*S. Stojanov, N. A. Thai, Z. Huang, and J. M. Rehg.*
[paper](https://openreview.net/pdf?id=KJemAi9fymT)
[code](https://github.com/rehg-lab/dope_selfsup)

1. **Pose Adaptive Dual Mixup for Few-Shot Single-View 3D Reconstruction,** in AAAI, 2022.
*T. Y. Cheng, H.-R. Yang, N. Trigoni, H.-T. Chen, and T.-L. Liu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19920/19679)

1. **Meta Faster R-Cnn: Towards Accurate Few-Shot Object Detection With Attentive Feature Alignment,** in AAAI, 2022.
*G. Han, S. Huang, J. Ma, Y. He, and S.-F. Chang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19959/19718)
[code](https://github.com/GuangxingHan/Meta-Faster-R-CNN)

1. **TA2N: Two-Stage Action Alignment Network for Few-Shot Action Recognition,** in AAAI, 2022.
*S. Li, H. Liu, R. Qian, Y. Li, J. See, M. Fei, X. Yu, and W. Lin.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20029/19788)

1. **Learning From the Target: Dual Prototype Network for Few Shot Semantic Segmentation,** in AAAI, 2022.
*B. Mao, X. Zhang, L. Wang, Q. Zhang, S. Xiang, and C. Pan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20090/19849)

1. **OA-FSUI2IT: A Novel Few-Shot Cross Domain Object Detection Framework With Object-Aware Few-Shot Unsupervised Image-to-Image Translation,** in AAAI, 2022.
*L. Zhao, Y. Meng, and L. Xu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20253)

1. **When Facial Expression Recognition Meets Few-Shot Learning: A Joint and Alternate Learning Framework,** in AAAI, 2022.
*X. Zou, Y. Yan, J.-H. Xue, S. Chen, and H. Wang.*
[paper](https://arxiv.org/abs/2201.06781)

1. **Dual Attention Networks for Few-Shot Fine-Grained Recognition,** in AAAI, 2022.
*S.-L. Xu, F. Zhang, X.-S. Wei, and J. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20196)

1. **Inferring Prototypes for Multi-Label Few-Shot Image Classification With Word Vector Guided Attention,** in AAAI, 2022.
*K. Yan, C. Zhang, J. Hou, P. Wang, Z. Bouraoui, S. Jameel, and S. Schockaert.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20205/19964)

1. **Analogy-Forming Transformers for Few-Shot 3D Parsing,** in ICLR, 2023.
*N. Gkanatsios, M. Singh, Z. Fang, S. Tulsiani, and K. Fragkiadaki.*
[paper](https://openreview.net/pdf?id=SRIQZTh0IK)
[code](http://analogicalnets.github.io)

1. **Suppressing the Heterogeneity: A Strong Feature Extractor for Few-Shot Segmentation,** in ICLR, 2023.
*Z. Hu, Y. Sun, and Y. Yang.*
[paper](https://openreview.net/pdf?id=CGuvK3U09LH)

1. **Universal Few-Shot Learning of Dense Prediction Tasks With Visual Token Matching,** in ICLR, 2023.
*D. Kim, J. Kim, S. Cho, C. Luo, and S. Hong.*
[paper](https://openreview.net/pdf?id=88nT0j5jAn)
[code](https://github.com/GitGyun/visual_token_matching)

1. **Few-Shot Semantic Image Synthesis With Class Affinity Transfer,** in CVPR, 2023.
*M. Careil, J. Verbeek, and S. Lathuilière.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Careil_Few-Shot_Semantic_Image_Synthesis_With_Class_Affinity_Transfer_CVPR_2023_paper.pdf)

1. **Semantic Prompt for Few-Shot Image Recognition,** in CVPR, 2023.
*W. Chen, C. Si, Z. Zhang, L. Wang, Z. Wang, and T. Tan.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Semantic_Prompt_for_Few-Shot_Image_Recognition_CVPR_2023_paper.pdf)

1. **ViewNet: A Novel Projection-Based Backbone With View Pooling for Few-Shot Point Cloud Classification,** in CVPR, 2023.
*J. Chen, M. Yang, and S. Velipasalar.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_ViewNet_A_Novel_Projection-Based_Backbone_With_View_Pooling_for_Few-Shot_CVPR_2023_paper.pdf)

1. **Meta-Tuning Loss Functions and Data Augmentation for Few-Shot Object Detection,** in CVPR, 2023.
*B. Demirel, O. B. Baran, and R. G. Cinbis.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Demirel_Meta-Tuning_Loss_Functions_and_Data_Augmentation_for_Few-Shot_Object_Detection_CVPR_2023_paper.pdf)

1. **Few-Shot Geometry-Aware Keypoint Localization,** in CVPR, 2023.
*X. He, G. Bharaj, D. Ferman, H. Rhodin, and P. Garrido.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/He_Few-Shot_Geometry-Aware_Keypoint_Localization_CVPR_2023_paper.pdf)
[code](https://xingzhehe.github.io/FewShot3DKP/)

1. **Meta Learning to Bridge Vision and Language Models for Multimodal Few-Shot Learning,** in ICLR, 2023.
*I. Najdenkoska, X. Zhen, and M. Worring.*
[paper](https://openreview.net/pdf?id=3oWo92cQyxL)
[code](https://github.com/ivonajdenkoska/multimodal-meta-learn)

1. **Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning With Multimodal Models,** in CVPR, 2023.
*Z. Lin, S. Yu, Z. Kuang, D. Pathak, and D. Ramanan.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Multimodality_Helps_Unimodality_Cross-Modal_Few-Shot_Learning_With_Multimodal_Models_CVPR_2023_paper.pdf)
[code](https://github.com/linzhiqiu/cross_modal_adaptation)

1. **Active Exploration of Multimodal Complementarity for Few-Shot Action Recognition,** in CVPR, 2023.
*Y. Wanyan, X. Yang, C. Chen, and C. Xu.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wanyan_Active_Exploration_of_Multimodal_Complementarity_for_Few-Shot_Action_Recognition_CVPR_2023_paper.pdf)

1. **AsyFOD: An Asymmetric Adaptation Paradigm for Few-Shot Domain Adaptive Object Detection,** in CVPR, 2023.
*Y. Gao, K.-Y. Lin, J. Yan, Y. Wang, and W.-S. Zheng.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object_CVPR_2023_paper.pdf)
[code](https://github.com/Hlings/AsyFOD)

1. **NIFF: Alleviating Forgetting in Generalized Few-Shot Object Detection via Neural Instance Feature Forging,** in CVPR, 2023.
*K. Guirguis, J. Meier, G. Eskandar, M. Kayser, B. Yang, and J. Beyerer.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guirguis_NIFF_Alleviating_Forgetting_in_Generalized_Few-Shot_Object_Detection_via_Neural_CVPR_2023_paper.pdf)

1. **A Strong Baseline for Generalized Few-Shot Semantic Segmentation,** in CVPR, 2023.
*S. Hajimiri, M. Boudiaf, I. B. Ayed, and J. Dolz.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hajimiri_A_Strong_Baseline_for_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2023_paper.pdf)
[code](https://github.com/sinahmr/DIaM)

1. **StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning,** in CVPR, 2023.
*Y. Fu, Y. Xie, Y. Fu, and Y.-G. Jiang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_StyleAdv_Meta_Style_Adversarial_Training_for_Cross-Domain_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/lovelyqian/StyleAdv-CDFSL)

1. **BlendFields: Few-Shot Example-Driven Facial Modeling,** in CVPR, 2023.
*K. Kania, S. J. Garbin, A. Tagliasacchi, V. Estellers, K. M. Yi, J. Valentin, T. Trzcinski, and M. Kowalski.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kania_BlendFields_Few-Shot_Example-Driven_Facial_Modeling_CVPR_2023_paper.pdf)

1. **Learning Orthogonal Prototypes for Generalized Few-Shot Semantic Segmentation,** in CVPR, 2023.
*S. Liu, Y. Zhang, Z. Qiu, H. Xie, Y. Zhang, and T. Yao.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Learning_Orthogonal_Prototypes_for_Generalized_Few-Shot_Semantic_Segmentation_CVPR_2023_paper.pdf)

1. **DiGeo: Discriminative Geometry-Aware Learning for Generalized Few-Shot Object Detection,** in CVPR, 2023.
*J. Ma, Y. Niu, J. Xu, S. Huang, G. Han, and S.-F. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_DiGeo_Discriminative_Geometry-Aware_Learning_for_Generalized_Few-Shot_Object_Detection_CVPR_2023_paper.pdf)
[code](https://github.com/Phoenix-V/DiGeo)

1. **Hierarchical Dense Correlation Distillation for Few-Shot Segmentation,** in CVPR, 2023.
*B. Peng, Z. Tian, X. Wu, C. Wang, S. Liu, J. Su, and J. Jia.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Peng_Hierarchical_Dense_Correlation_Distillation_for_Few-Shot_Segmentation_CVPR_2023_paper.pdf)
[code](https://github.com/Pbihao/HDMNet)

1. **Rethinking the Correlation in Few-Shot Segmentation: A Buoys View,** in CVPR, 2023.
*Y. Wang, R. Sun, and T. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Rethinking_the_Correlation_in_Few-Shot_Segmentation_A_Buoys_View_CVPR_2023_paper.pdf)

1. **CF-Font: Content Fusion for Few-Shot Font Generation,** in CVPR, 2023.
*C. Wang, M. Zhou, T. Ge, Y. Jiang, H. Bao, and W. Xu.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_CF-Font_Content_Fusion_for_Few-Shot_Font_Generation_CVPR_2023_paper.pdf)
[code](https://github.com/wangchi95/CF-Font)

1. **MoLo: Motion-Augmented Long-Short Contrastive Learning for Few-Shot Action Recognition,** in CVPR, 2023.
*X. Wang, S. Zhang, Z. Qing, C. Gao, Y. Zhang, D. Zhao, and N. Sang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MoLo_Motion-Augmented_Long-Short_Contrastive_Learning_for_Few-Shot_Action_Recognition_CVPR_2023_paper.pdf)
[code](https://github.com/alibaba-mmai-research/MoLo)


1. **Generating Features With Increased Crop-Related Diversity for Few-Shot Object Detection,** in CVPR, 2023.
*J. Xu, H. Le, and D. Samaras.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Generating_Features_With_Increased_Crop-Related_Diversity_for_Few-Shot_Object_Detection_CVPR_2023_paper.pdf)

1. **SMAE: Few-Shot Learning for HDR Deghosting With Saturation-Aware Masked Autoencoders,** in CVPR, 2023.
*Q. Yan, S. Zhang, W. Chen, H. Tang, Y. Zhu, J. Sun, L. V. Gool, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_SMAE_Few-Shot_Learning_for_HDR_Deghosting_With_Saturation-Aware_Masked_Autoencoders_CVPR_2023_paper.pdf)

1. **MIANet: Aggregating Unbiased Instance and General Information for Few-Shot Semantic Segmentation,** in CVPR, 2023.
*Y. Yang, Q. Chen, Y. Feng, and T. Huang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_MIANet_Aggregating_Unbiased_Instance_and_General_Information_for_Few-Shot_Semantic_CVPR_2023_paper.pdf)
[code](https://github.com/Aldrich2y/MIANet)

1. **FreeNeRF: Improving Few-Shot Neural Rendering With Free Frequency Regularization,** in CVPR, 2023.
*J. Yang, M. Pavone, and Y. Wang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_FreeNeRF_Improving_Few-Shot_Neural_Rendering_With_Free_Frequency_Regularization_CVPR_2023_paper.pdf)
[code](https://github.com/Jiawei-Yang/FreeNeRF)

1. **Exploring Incompatible Knowledge Transfer in Few-Shot Image Generation,** in CVPR, 2023.
*Y. Zhao, C. Du, M. Abdollahzadeh, T. Pang, M. Lin, S. Yan, and N.-M. Cheung.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Exploring_Incompatible_Knowledge_Transfer_in_Few-Shot_Image_Generation_CVPR_2023_paper.pdf)
[code](https://github.com/yunqing-me/RICK)

1. **Where Is My Spot? Few-Shot Image Generation via Latent Subspace Optimization,** in CVPR, 2023.
*C. Zheng, B. Liu, H. Zhang, X. Xu, and S. He.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_Where_Is_My_Spot_Few-Shot_Image_Generation_via_Latent_Subspace_CVPR_2023_paper.pdf)
[code](https://github.com/chansey0529/LSO)

1. **Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification & Segmentation,** in CVPR, 2023.
*D. Kang, P. Koniusz, M. Cho, and N. Murray.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Distilling_Self-Supervised_Vision_Transformers_for_Weakly-Supervised_Few-Shot_Classification__Segmentation_CVPR_2023_paper.pdf)

1. **FGNet: Towards Filling the Intra-Class and Inter-Class Gaps for Few-Shot Segmentation,** in IJCAI, 2023.
*Y. Zhang, W. Yang, and S. Wang.*
[paper](https://www.ijcai.org/proceedings/2023/0194.pdf)
[code](https://github.com/YXZhang979/FGNet)

1. **Clustered-Patch Element Connection for Few-Shot Learning,** in IJCAI, 2023.
*J. Lai, S. Yang, J. Zhou, W. Wu, X. Chen, J. Liu, B.-B. Gao, and C. Wang.*
[paper](https://www.ijcai.org/proceedings/2023/0110.pdf)

1. **GeCoNeRF: Few-Shot Neural Radiance Fields via Geometric Consistency,** in ICML, 2023.
*M. Kwak, J. Song, and S. Kim.*
[paper](https://proceedings.mlr.press/v202/kwak23a/kwak23a.pdf)
[code](https://github.com/KU-CVLAB/GeCoNeRF)

1. **Few-Shot Common Action Localization via Cross-Attentional Fusion of Context and Temporal Dynamics,** in ICCV, 2023.
*J. Lee, M. Jain, and S. Yun.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Few-Shot_Common_Action_Localization_via_Cross-Attentional_Fusion_of_Context_and_ICCV_2023_paper.pdf)

1. **StyleDomain: Efficient and Lightweight Parameterizations of StyleGAN for One-Shot and Few-Shot Domain Adaptation,** in ICCV, 2023.
*A. Alanov, V. Titov, M. Nakhodnov, and D. Vetrov.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Alanov_StyleDomain_Efficient_and_Lightweight_Parameterizations_of_StyleGAN_for_One-shot_and_ICCV_2023_paper.pdf)
[code](https://github.com/AIRI-Institute/StyleDomain)

1. **FlipNeRF: Flipped Reflection Rays for Few-Shot Novel View Synthesis,** in ICCV, 2023.
*S. Seo, Y. Chang, and N. Kwak.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Seo_FlipNeRF_Flipped_Reflection_Rays_for_Few-shot_Novel_View_Synthesis_ICCV_2023_paper.pdf)

1. **Few-Shot Physically-Aware Articulated Mesh Generation via Hierarchical Deformation,** in ICCV, 2023.
*X. Liu, B. Wang, H. Wang, and L. Yi.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Few-Shot_Physically-Aware_Articulated_Mesh_Generation_via_Hierarchical_Deformation_ICCV_2023_paper.pdf)
[code](meowuu7.github.io/few-arti-obj-gen)

1. **SparseNeRF: Distilling Depth Ranking for Few-Shot Novel View Synthesis,** in ICCV, 2023.
*G. Wang, Z. Chen, C. C. Loy, and Z. Liu.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SparseNeRF_Distilling_Depth_Ranking_for_Few-shot_Novel_View_Synthesis_ICCV_2023_paper.pdf)
[code](https://sparsenerf.github.io/)

1. **Few-Shot Video Classification via Representation Fusion and Promotion Learning,** in ICCV, 2023.
*H. Xia, K. Li, M. R. Min, and Z. Ding.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Few-Shot_Video_Classification_via_Representation_Fusion_and_Promotion_Learning_ICCV_2023_paper.pdf)

1. **Augmenting and Aligning Snippets for Few-Shot Video Domain Adaptation,** in ICCV, 2023.
*Y. Xu, J. Yang, Y. Zhou, Z. Chen, M. Wu, and X. Li.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Augmenting_and_Aligning_Snippets_for_Few-Shot_Video_Domain_Adaptation_ICCV_2023_paper.pdf)
[code](https://github.com/xuyu0010/SSA2lign)

1. **One-Shot Recognition of Any Material Anywhere Using Contrastive Learning With Physics-Based Rendering,** in ICCV, 2023.
*M. S. Drehwald, S. Eppel, J. Li, H. Hao, and A. Aspuru-Guzik.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Drehwald_One-Shot_Recognition_of_Any_Material_Anywhere_Using_Contrastive_Learning_with_ICCV_2023_paper.pdf)
[code](https://github.com/ZuseZ4/MatSim-Dataset-Generator-Scripts-And-Neural-net)


1. **FS-DETR: Few-Shot Detection Transformer With Prompting and Without Re-Training,** in ICCV, 2023.
*A. Bulat, R. Guerrero, B. Martinez, and G. Tzimiropoulos.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bulat_FS-DETR_Few-Shot_DEtection_TRansformer_with_Prompting_and_without_Re-Training_ICCV_2023_paper.pdf)

1. **Confidence-Based Visual Dispersal for Few-Shot Unsupervised Domain Adaptation,** in ICCV, 2023.
*Y. Xiong, H. Chen, Z. Lin, S. Zhao, and G. Ding.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiong_Confidence-based_Visual_Dispersal_for_Few-shot_Unsupervised_Domain_Adaptation_ICCV_2023_paper.pdf)
[code](https://github.com/Bostoncake/C-VisDiT)

1. **CDFSL-V: Cross-Domain Few-Shot Learning for Videos,** in ICCV, 2023.
*S. Samarasinghe, M. N. Rizve, N. Kardan, and M. Shah.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Samarasinghe_CDFSL-V_Cross-Domain_Few-Shot_Learning_for_Videos_ICCV_2023_paper.pdf)
[code](https://github.com/Sarinda251/CDFSL-V)

1. **Generalized Few-Shot Point Cloud Segmentation via Geometric Words,** in ICCV, 2023.
*Y. Xu, C. Hu, N. Zhao, and G. H. Lee.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Generalized_Few-Shot_Point_Cloud_Segmentation_via_Geometric_Words_ICCV_2023_paper.pdf)
[code](https://github.com/Pixie8888/GFS-3DSeg_GWs)

1. **Invariant Training 2d-3d Joint Hard Samples for Few-Shot Point Cloud Recognition,** in ICCV, 2023.
*X. Yi, J. Deng, Q. Sun, X. Hua, J. Lim, and H. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yi_Invariant_Training_2D-3D_Joint_Hard_Samples_for_Few-Shot_Point_Cloud_ICCV_2023_paper.pdf)
[code](https://github.com/yxymessi/InvJoint)

1. **CIRI: Curricular Inactivation for Residue-Aware One-Shot Video Inpainting,** in ICCV, 2023.
*W. Zheng, C. Xu, X. Xu, W. Liu, and S. He.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_CIRI_Curricular_Inactivation_for_Residue-aware_One-shot_Video_Inpainting_ICCV_2023_paper.pdf)
[code](https://github.com/Arise-zwy/CIRI)

1. **S-Adaptive Decoupled Prototype for Few-Shot Object Detection,** in ICCV, 2023.
*J. Du, S. Zhang, Q. Chen, H. Le, Y. Sun, Y. Ni, J. Wang, B. He, and J. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Du_s-Adaptive_Decoupled_Prototype_for_Few-Shot_Object_Detection_ICCV_2023_paper.pdf)

1. **Parallel Attention Interaction Network for Few-Shot Skeleton-Based Action Recognition,** in ICCV, 2023.
*X. Liu, S. Zhou, L. Wang, and G. Hua.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Parallel_Attention_Interaction_Network_for_Few-Shot_Skeleton-Based_Action_Recognition_ICCV_2023_paper.pdf)

1. **Robust One-Shot Face Video Re-Enactment Using Hybrid Latent Spaces of StyleGAN2,** in ICCV, 2023.
*T. Oorloff, and Y. Yacoob.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Oorloff_Robust_One-Shot_Face_Video_Re-enactment_using_Hybrid_Latent_Spaces_of_ICCV_2023_paper.pdf)
[code](https://trevineoorloff.github.io/FaceVideoReenactment_HybridLatents.io/)

1. **Informative Data Mining for One-Shot Cross-Domain Semantic Segmentation,** in ICCV, 2023.
*Y. Wang, J. Liang, J. Xiao, S. Mei, Y. Yang, and Z. Zhang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Informative_Data_Mining_for_One-Shot_Cross-Domain_Semantic_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/yxiwang/IDM)

1. **The Euclidean Space Is Evil: Hyperbolic Attribute Editing for Few-Shot Image Generation,** in ICCV, 2023.
*L. Li, Y. Zhang, and S. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_The_Euclidean_Space_is_Evil_Hyperbolic_Attribute_Editing_for_Few-shot_ICCV_2023_paper.pdf)
[code](https://github.com/lingxiao-li/HAE)

1. **Few Shot Font Generation via Transferring Similarity Guided Global Style and Quantization Local Style,** in ICCV, 2023.
*W. Pan, A. Zhu, X. Zhou, B. K. Iwana, and S. Li.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_Few_Shot_Font_Generation_Via_Transferring_Similarity_Guided_Global_Style_ICCV_2023_paper.pdf)
[code](https://github.com/awei669/VQ-Font)

1. **Boosting Few-Shot Action Recognition With Graph-Guided Hybrid Matching,** in ICCV, 2023.
*J. Xing, M. Wang, Y. Ruan, B. Chen, Y. Guo, B. Mu, G. Dai, J. Wang, and Y. Liu.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xing_Boosting_Few-shot_Action_Recognition_with_Graph-guided_Hybrid_Matching_ICCV_2023_paper.pdf)
[code](https://github.com/jiazheng-xing/GgHM)

1. **MSI: Maximize Support-Set Information for Few-Shot Segmentation,** in ICCV, 2023.
*S. Moon, S. S. Sohn, H. Zhou, S. Yoon, V. Pavlovic, M. H. Khan, and M. Kapadia.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Moon_MSI_Maximize_Support-Set_Information_for_Few-Shot_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/moonsh/MSI-Maximize-Support-Set-Information-ICCV2023)


1. **Self-Calibrated Cross Attention Network for Few-Shot Segmentation,** in ICCV, 2023.
*Q. Xu, W. Zhao, G. Lin, and C. Long.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Self-Calibrated_Cross_Attention_Network_for_Few-Shot_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/Sam1224/SCCAN)

1. **Multi-Grained Temporal Prototype Learning for Few-Shot Video Object Segmentation,** in ICCV, 2023.
*N. Liu, K. Nan, W. Zhao, Y. Liu, X. Yao, S. Khan, H. Cholakkal, R. M. Anwer, J. Han, and F. S. Khan.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Multi-grained_Temporal_Prototype_Learning_for_Few-shot_Video_Object_Segmentation_ICCV_2023_paper.pdf)
[code](https://github.com/nankepan/VIPMT)

1. **HyperReenact: One-Shot Reenactment via Jointly Learning to Refine and Retarget Faces,** in ICCV, 2023.
*S. Bounareli, C. Tzelepis, V. Argyriou, I. Patras, and G. Tzimiropoulos.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bounareli_HyperReenact_One-Shot_Reenactment_via_Jointly_Learning_to_Refine_and_Retarget_ICCV_2023_paper.pdf)
[code](https://github.com/StelaBou/HyperReenact)

1. **General Image-to-Image Translation With One-Shot Image Guidance,** in ICCV, 2023.
*B. Cheng, Z. Liu, Y. Peng, and Y. Lin.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_General_Image-to-Image_Translation_with_One-Shot_Image_Guidance_ICCV_2023_paper.pdf)
[code](https://github.com/CrystalNeuro/visual-concept-translator)

1. **ActorsNeRF: Animatable Few-Shot Human Rendering With Generalizable NeRFs,** in ICCV, 2023.
*J. Mu, S. Sang, N. Vasconcelos, and X. Wang.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Mu_ActorsNeRF_Animatable_Few-shot_Human_Rendering_with_Generalizable_NeRFs_ICCV_2023_paper.pdf)
[code](https://jitengmu.github.io/ActorsNeRF/)

1. **One-Shot Implicit Animatable Avatars With Model-Based Priors,** in ICCV, 2023.
*Y. Huang, H. Yi, W. Liu, H. Wang, B. Wu, W. Wang, B. Lin, D. Zhang, and D. Cai.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_One-shot_Implicit_Animatable_Avatars_with_Model-based_Priors_ICCV_2023_paper.pdf)
[code](https://huangyangyi.github.io/ELICIT)

1. **Preface: A Data-Driven Volumetric Prior for Few-Shot Ultra High-Resolution Face Synthesis,** in ICCV, 2023.
*M. C. Bühler, K. Sarkar, T. Shah, G. Li, D. Wang, L. Helminger, S. Orts-Escolano, D. Lagun, O. Hilliges, T. Beeler, and A. Meka.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Buhler_Preface_A_Data-driven_Volumetric_Prior_for_Few-shot_Ultra_High-resolution_Face_ICCV_2023_paper.pdf)

1. **DINAR: Diffusion Inpainting of Neural Textures for One-Shot Human Avatars,** in ICCV, 2023.
*D. Svitov, D. Gudkov, R. Bashirov, and V. Lempitsky.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Svitov_DINAR_Diffusion_Inpainting_of_Neural_Textures_for_One-Shot_Human_Avatars_ICCV_2023_paper.pdf)

1. **Tune-a-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,** in ICCV, 2023.
*J. Z. Wu, Y. Ge, X. Wang, S. W. Lei, Y. Gu, Y. Shi, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.pdf)
[code](https://tuneavideo.github.io)

1. **Phasic Content Fusing Diffusion Model With Directional Distribution Consistency for Few-Shot Model Adaption,** in ICCV, 2023.
*T. Hu, J. Zhang, L. Liu, R. Yi, S. Kou, H. Zhu, X. Chen, Y. Wang, C. Wang, and L. Ma.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Phasic_Content_Fusing_Diffusion_Model_with_Directional_Distribution_Consistency_for_ICCV_2023_paper.pdf)
[code](https://github.com/sjtuplayer/few-shot-diffusion)

1. **Prototypical Variational Autoencoder for 3D Few-Shot Object Detection,** in NeurIPS, 2023.
*W. Tang, B. YANG, X. Li, Y. Liu, P. Heng, and C. Fu.*
[paper](https://openreview.net/attachment?id=fljrZsJ2I8&name=pdf)

1. **Generalizable One-Shot 3D Neural Head Avatar,** in NeurIPS, 2023.
*X. Li, S. D. Mello, S. Liu, K. Nagano, U. Iqbal, and J. Kautz.*
[paper](https://openreview.net/attachment?id=95q46MpBGZ&name=pdf)
[code](https://research.nvidia.com/labs/lpr/one-shot-avatar/)

1. **Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation,** in NeurIPS, 2023.
*Y. Wang, N. Luo, and T. Zhang.*
[paper](https://openreview.net/attachment?id=hxJu0386if&name=pdf)
[code](https://github.com/Wyxdm/AMNet)

1. **Bi-Directional Feature Reconstruction Network for Fine-Grained Few-Shot Image Classification,** in AAAI, 2023.
*J. Wu, D. Chang, A. Sain, X. Li, Z. Ma, J. Cao, J. Guo, and Y.-Z. Song.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25383/25155)
[code](https://github.com/PRIS-CV/Bi-FRN)

1. **Revisiting the Spatial and Temporal Modeling for Few-Shot Action Recognition,** in AAAI, 2023.
*J. Xing, M. Wang, Y. Liu, and B. Mu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25403/25175)

1. **Disentangle and Remerge: Interventional Knowledge Distillation for Few-Shot Object Detection From a Conditional Causal Perspective,** in AAAI, 2023.
*J. Li, Y. Zhang, W. Qiang, L. Si, C. Jiao, X. Hu, C. Zheng, and F. Sun.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25216/24988)
[code](https://github.com/ZYN-1101/DandR)

1. **Breaking Immutable: Information-Coupled Prototype Elaboration for Few-Shot Object Detection,** in AAAI, 2023.
*X. Lu, W. Diao, Y. Mao, J. Li, P. Wang, X. Sun, and K. Fu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25274/25046)
[code](https://github.com/lxn96/ICPE)

1. **Few-Shot Object Detection via Variational Feature Aggregation,** in AAAI, 2023.
*J. Han, Y. Ren, J. Ding, K. Yan, and G.-S. Xia.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25153/24925)

1. **Few-Shot 3D Point Cloud Semantic Segmentation via Stratified Class-Specific Attention Based Transformer Network,** in AAAI, 2023.
*C. Zhang, Z. Wu, X. Wu, Z. Zhao, and S. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25449/25221)
[code](https://github.com/czzhang179/SCAT)

1. **Few-Shot Composition Learning for Image Retrieval With Prompt Tuning,** in AAAI, 2023.
*J. Wu, R. Wang, H. Zhao, R. Zhang, C. Lu, S. Li, and R. Henao.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25597/25369)

1. **Real3D-Portrait: One-Shot Realistic 3D Talking Portrait Synthesis,** in ICLR, 2024.
*Z. Ye, T. Zhong, Y. Ren, J. Yang, W. Li, J. Huang, Z. Jiang, J. He, R. Huang, J. Liu, C. Zhang, X. Yin, Z. MA, and Z. Zhao.*
[paper](https://openreview.net/attachment?id=7ERQPyR2eb&name=pdf)
[code](https://github.com/yerfor/Real3DPortrait)

1. **Personalize Segment Anything Model With One Shot,** in ICLR, 2024.
*R. Zhang, Z. Jiang, Z. Guo, S. Yan, J. Pan, H. Dong, Y. Qiao, P. Gao, and H. Li.*
[paper](https://openreview.net/attachment?id=6Gzkhoc6YS&name=pdf)

1. **Matcher: Segment Anything With One Shot Using All-Purpose Feature Matching,** in ICLR, 2024.
*Y. Liu, M. Zhu, H. Li, H. Chen, X. Wang, and C. Shen.*
[paper](https://openreview.net/attachment?id=yzRXdhk2he&name=pdf)

1. **SparseDFF: Sparse-View Feature Distillation for One-Shot Dexterous Manipulation,** in ICLR, 2024.
*Q. Wang, H. Zhang, C. Deng, Y. You, H. Dong, Y. Zhu, and L. Guibas.*
[paper](https://openreview.net/attachment?id=HHWlwxDeRn&name=pdf)

1. **Deformable One-Shot Face Stylization via DINO Semantic Guidance,** in CVPR, 2024.
*Y. Zhou, Z. Chen, and H. Huang.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00744)

1. **One-Shot Structure-Aware Stylized Image Synthesis,** in CVPR, 2024.
*H. Cho, J. Lee, S. Chang, and Y. Jeong.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00793)

1. **Bi-level Learning of Task-Specific Decoders for Joint Registration and One-Shot Medical Image Segmentation,** in CVPR, 2024.
*X. Fan, X. Wang, J. Gao, J. Wang, Z. Luo, and R. Liu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01114)

1. **LLaFS: When Large Language Models Meet Few-Shot Segmentation,** in CVPR, 2024.
*L. Zhu, T. Chen, D. Ji, J. Ye, and J. Liu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00296)

1. **Addressing Background Context Bias in Few-Shot Segmentation Through Iterative Modulation,** in CVPR, 2024.
*L. Zhu, T. Chen, J. Yin, S. See, and J. Liu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00324)

1. **Cross-Domain Few-Shot Segmentation via Iterative Support-Query Correspondence Mining,** in CVPR, 2024.
*J. Nie, Y. Xing, G. Zhang, P. Yan, A. Xiao, Y.-P. Tan, A. C. Kot, and S. Lu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00325)

1. **Unlocking the Potential of Pre-Trained Vision Transformers for Few-Shot Semantic Segmentation through Relationship Descriptors,** in CVPR, 2024.
*Z. Zhou, H.-M. Xu, Y. Shu, and L. Liu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00366)

1. **No Time to Train: Empowering Non-Parametric Networks for Few-Shot 3D Scene Segmentation,** in CVPR, 2024.
*X. Zhu, R. Zhang, B. He, Z. Guo, J. Liu, H. Xiao, C. Fu, H. Dong, and P. Gao.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00368)

1. **Rethinking Prior Information Generation with CLIP for Few-Shot Segmentation,** in CVPR, 2024.
*J. Wang, B. Zhang, J. Pang, H. Chen, and W. Liu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00378)

1. **Rethinking Few-shot 3D Point Cloud Semantic Segmentation,** in CVPR, 2024.
*Z. An, G. Sun, Y. Liu, F. Liu, Z. Wu, D. Wang, L. V. Gool, and S. J. Belongie.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00383)

1. **Back to 3D: Few-Shot 3D Keypoint Detection with Back-Projected 2D Features,** in CVPR, 2024.
*T. Wimmer, P. Wonka, and M. Ovsjanikov.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00398)

1. **Multiplane Prior Guided Few-Shot Aerial Scene Rendering,** in CVPR, 2024.
*Z. Gao, L. Jiao, L. Li, X. Liu, F. Liu, P. Chen, and Y. Guo.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00479)

1. **LAMP: Learn A Motion Pattern for Few-Shot Video Generation,** in CVPR, 2024.
*R. Wu, L. Chen, T. Yang, C. Guo, C. Li, and X. Zhang.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00677)

1. **Exact Fusion via Feature Distribution Matching for Few-Shot Image Generation,** in CVPR, 2024.
*Y. Zhou, Y. Ye, P. Zhang, X. Wei, and M. Chen.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00801)

1. **SNIDA: Unlocking Few-Shot Object Detection with Non-Linear Semantic Decoupling Augmentation,** in CVPR, 2024.
*Y. Wang, X. Zou, L. Yan, S. Zhong, and J. Zhou.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01192)

1. **Is Vanilla MLP in Neural Radiance Field Enough for Few-Shot View Synthesis? 20288-20298,** in CVPR, 2024.
*H. Zhu, T. He, X. Li, B. Li, and Z. Chen.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01918)

1. **Learning with Unreliability: Fast Few-Shot Voxel Radiance Fields with Relative Geometric Consistency,** in CVPR, 2024.
*Y. Xu, B. Liu, H. Tang, B. Deng, and S. He.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01923)

1. **Global and Hierarchical Geometry Consistency Priors for Few-Shot NeRFs in Indoor Scenes,** in CVPR, 2024.
*X. Sun, Q. Xu, X. Yang, Y. Zang, and C. Wang.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01940)

1. **APSeg: Auto-Prompt Network for Cross-Domain Few-Shot Semantic Segmentation,** in CVPR, 2024.
*W. He, Y. Zhang, W. Zhuo, L. Shen, J. Yang, S. Deng, and L. Sun.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02243)

1. **LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP,** in CVPR, 2024.
*Y. Huang, F. Shakeri, J. Dolz, M. Boudiaf, H. Bahig, and I. B. Ayed.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02244)

1. **Domain-Rectifying Adapter for Cross-Domain Few-Shot Segmentation,** in CVPR, 2024.
*J. Su, Q. Fan, W. Pei, G. Lu, and F. Chen.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02269)

1. **Few-Shot Object Detection with Foundation Models,** in CVPR, 2024.
*G. Han, and S.-N. Lim.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02703)

1. **Instance-based Max-margin for Practical Few-shot Recognition,** in CVPR, 2024.
*M. Fu, and K. Zhu.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02709)

1. **OHTA: One-shot Hand Avatar via Data-driven Implicit Priors,** in CVPR, 2024.
*X. Zheng, C. Wen, Z. Su, Z. Xu, Z. Li, Y. Zhao, and Z. Xue.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00082)

1. **Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data,** in CVPR, 2024.
*Y. Deng, D. Wang, X. Ren, X. Chen, and B. Wang.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00680)

1. **HAVE-FUN: Human Avatar Reconstruction from Few-Shot Unconstrained Images,** in CVPR, 2024.
*X. Yang, X. Chen, D. Gao, S. Wang, X. Han, and B. Wang.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00077)

1. **FaceChain-SuDe: Building Derived Class to Inherit Category Attributes for One-Shot Subject-Driven Generation,** in CVPR, 2024.
*P. Qiao, L. Shang, C. Liu, B. Sun, X. Ji, and J. Chen.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00689)



1. **VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D Head Reenactment,** in CVPR, 2024.
*P. Tran, E. Zakharov, L.-N. Ho, A. T. Tran, L. Hu, and H. Li.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00984)

1. **Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation,** in CVPR, 2024.
*and J. Herzog.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02228)

1. **Flatten Long-Range Loss Landscapes for Cross-Domain Few-Shot Learning,** in CVPR, 2024.
*Y. Zou, Y. Liu, Y. Hu, Y. Li, and R. Li.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02225)

1. **Aggregation and Purification: Dual Enhancement Network for Point Cloud Few-shot Segmentation,** in IJCAI, 2024.
*G. Xiong, Y. Wang, Z. Li, W. Yang, T. Zhang, X. Zhou, S. Zhang, and Y. Zhang.*
[paper](https://www.ijcai.org/proceedings/2024/164)

1. **Learning Spatial Similarity Distribution for Few-shot Object Counting,** in IJCAI, 2024.
*Y. Xu, F. Song, and H. Zhang.*
[paper](https://www.ijcai.org/proceedings/2024/167)

1. **Multi-Attention Based Visual-Semantic Interaction for Few-Shot Learning,** in IJCAI, 2024.
*P. Zhao, Y. Wang, W. Wang, J. Mu, H. Liu, C. Wang, and X. Cao.*
[paper](https://www.ijcai.org/proceedings/2024/194)

1. **Cross-Domain Few-Shot Semantic Segmentation via Doubly Matching Transformation,** in IJCAI, 2024.
*J. Chen, R. Quan, and J. Qin.*
[paper](https://www.ijcai.org/proceedings/2024/71)

1. **A Transformer-Based Adaptive Prototype Matching Network for Few-Shot Semantic Segmentation,** in IJCAI, 2024.
*S. Chen, Y. Chen, Y. Zheng, Z.-X. Yang, and E. Wu.*
[paper](https://www.ijcai.org/proceedings/2024/73)

1. **Task-Agnostic Self-Distillation for Few-Shot Action Recognition,** in IJCAI, 2024.
*B. Zhang, Y. Dang, P. Chen, R. Liang, N. Gao, R. Huan, and X. He.*
[paper](https://www.ijcai.org/proceedings/2024/600)

1. **QDETRv: Query-Guided DETR for One-Shot Object Localization in Videos,** in AAAI, 2024.
*Y. Kumar, S. Mallick, A. Mishra, S. Rasipuram, A. Maitra, and R. R. Ramnani.*
[paper](https://doi.org/10.1609/aaai.v38i3.28063)

1. **MFOS: Model-Free & One-Shot Object Pose Estimation,** in AAAI, 2024.
*J. Lee, Y. Cabon, R. Brégier, S. Yoo, and J. Revaud.*
[paper](https://doi.org/10.1609/aaai.v38i4.28072)

1. **FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning,** in AAAI, 2024.
*Z. Yang, D. Peng, Y. Kong, Y. Zhang, C. Yao, and L. Jin.*
[paper](https://doi.org/10.1609/aaai.v38i7.28482)

1. **Exploring Base-Class Suppression with Prior Guidance for Bias-Free One-Shot Object Detection,** in AAAI, 2024.
*W. Zhang, Y. Hu, H. Shan, and E. Liu.*
[paper](https://doi.org/10.1609/aaai.v38i7.28561)

1. **Relevant Intrinsic Feature Enhancement Network for Few-Shot Semantic Segmentation,** in AAAI, 2024.
*X. Bao, J. Qin, S. Sun, X. Wang, and Y. Zheng.*
[paper](https://doi.org/10.1609/aaai.v38i2.27834)

1. **FeatWalk: Enhancing Few-Shot Classification through Local View Leveraging,** in AAAI, 2024.
*D. Chen, J. Zhang, W.-S. Zheng, and R. Wang.*
[paper](https://doi.org/10.1609/aaai.v38i2.27862)

1. **WeditGAN: Few-Shot Image Generation via Latent Space Relocation,** in AAAI, 2024.
*Y. Duan, L. Niu, Y. Hong, and L. Zhang.*
[paper](https://doi.org/10.1609/aaai.v38i2.27932)

1. **MaskDiff: Modeling Mask Distribution with Diffusion Probabilistic Model for Few-Shot Instance Segmentation,** in AAAI, 2024.
*M.-Q. Le, T. V. Nguyen, T.-N. Le, T.-T. Do, M. N. Do, and M.-T. Tran.*
[paper](https://doi.org/10.1609/aaai.v38i3.28068)

1. **Few-Shot Neural Radiance Fields under Unconstrained Illumination,** in AAAI, 2024.
*S. Lee, J. Choi, S. Kim, I.-J. Kim, and J. Cho.*
[paper](https://doi.org/10.1609/aaai.v38i4.28075)

1. **Few-Shot Learning from Augmented Label-Uncertain Queries in Bongard-HOI,** in AAAI, 2024.
*Q. Lei, B. Wang, and R. T. Tan.*
[paper](https://doi.org/10.1609/aaai.v38i4.28079)

1. **Label-Efficient Few-Shot Semantic Segmentation with Unsupervised Meta-Training,** in AAAI, 2024.
*J. Li, K. Shi, G.-S. Xie, X. Liu, J. Zhang, and T. Zhou.*
[paper](https://doi.org/10.1609/aaai.v38i4.28094)

1. **Detect Any Keypoints: An Efficient Light-Weight Few-Shot Keypoint Detector,** in AAAI, 2024.
*C. Lu, and P. Koniusz.*
[paper](https://doi.org/10.1609/aaai.v38i4.28180)

1. **Cross-Layer and Cross-Sample Feature Optimization Network for Few-Shot Fine-Grained Image Classification,** in AAAI, 2024.
*Z.-X. Ma, Z.-D. Chen, L.-J. Zhao, Z.-C. Zhang, X. Luo, and X.-S. Xu.*
[paper](https://doi.org/10.1609/aaai.v38i5.28208)

1. **Task-Disruptive Background Suppression for Few-Shot Segmentation,** in AAAI, 2024.
*S. Park, S. B. Lee, S. Hyun, H. S. Seong, and J.-P. Heo.*
[paper](https://doi.org/10.1609/aaai.v38i5.28242)

1. **Boosting Few-Shot Learning via Attentive Feature Regularization,** in AAAI, 2024.
*X. Zhu, S. Wang, J. Lu, Y. Hao, H. Liu, and X. He.*
[paper](https://doi.org/10.1609/aaai.v38i7.28614)

1. **CGMGM: A Cross-Gaussian Mixture Generative Model for Few-Shot Semantic Segmentation,** in AAAI, 2024.
*J. Shen, K. Kuang, J. Wang, X. Wang, T. Feng, and W. Zhang.*
[paper](https://doi.org/10.1609/aaai.v38i5.28280)

1. **Self-Prompt Mechanism for Few-Shot Image Recognition,** in AAAI, 2024.
*M. Song, H. Wang, and G. Zhong.*
[paper](https://doi.org/10.1609/aaai.v38i5.28297)

1. **UniAP: Towards Universal Animal Perception in Vision via Few-Shot Learning,** in AAAI, 2024.
*M. Sun, Z. Zhao, W. Chai, H. Luo, S. Cao, Y. Zhang, J.-N. Hwang, and G. Wang.*
[paper](https://doi.org/10.1609/aaai.v38i5.28305)

1. **VQ-FONT: Few-Shot Font Generation with Structure-Aware Enhancement and Quantization,** in AAAI, 2024.
*M. Yao, Y. Zhang, X. Lin, X. Li, and W. Zuo.*
[paper](https://doi.org/10.1609/aaai.v38i15.29577)

1. **Weakly Supervised Few-Shot Object Detection with DETR,** in AAAI, 2024.
*C. Zhang, Y. Zhang, L. Zhang, J. Zhao, J. Guan, and S. Zhou.*
[paper](https://doi.org/10.1609/aaai.v38i7.28528)

1. **Cross-Modal Feature Distribution Calibration for Few-Shot Visual Question Answering,** in AAAI, 2024.
*J. Zhang, X. Liu, M. Chen, and Z. Wang.*
[paper](https://doi.org/10.1609/aaai.v38i7.28543)

1. **Adaptive FSS: A Novel Few-Shot Segmentation Framework via Prototype Enhancement,** in AAAI, 2024.
*J. Wang, J. Li, C. Chen, Y. Zhang, H. Shen, and T. Zhang.*
[paper](https://doi.org/10.1609/aaai.v38i6.28355)

1. **Fine-Grained Prototypes Distillation for Few-Shot Object Detection,** in AAAI, 2024.
*Z. Wang, B. Yang, H. Yue, and Z. Ma.*
[paper](https://doi.org/10.1609/aaai.v38i6.28399)

1. **AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model,** in AAAI, 2024.
*T. Hu, J. Zhang, R. Yi, Y. Du, X. Chen, L. Liu, Y. Wang, and C. Wang.*
[paper](https://doi.org/10.1609/aaai.v38i8.28696)


1. **Revisiting Few-Shot Object Detection with Vision-Language Models.,** in NeurIPS, 2024.
*A. Madan, N. Peri, S. Kong, and D. Ramanan.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/22b2067b8f680812624032025864c5a1-Abstract-Datasets_and_Benchmarks_Track.html)

1. **A Surprisingly Simple Approach to Generalized Few-Shot Semantic Segmentation.,** in NeurIPS, 2024.
*T. Sakai, H. Qiu, T. Katsuki, D. Kimura, T. Osogami, and T. Inoue.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/2f75a57e9c71e8369da0150ea769d5a2-Paper-Conference.pdf)

1. **Generated and Pseudo Content guided Prototype Refinement for Few-shot Point Cloud Segmentation.,** in NeurIPS, 2024.
*L. Wei, C. Lang, Z. Chen, T. Wang, Y. Li, and J. Liu.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/377d0752059d3d4686aa021b664a25dd-Paper-Conference.pdf)

1. **Unleashing the Potential of the Diffusion Model in Few-shot Semantic Segmentation.,** in NeurIPS, 2024.
*M. Zhu, Y. Liu, Z. Luo, C. Jing, H. Chen, G. Xu, X. Wang, and C. Shen.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/4b2a917e30e1bb1aff055b4d8c6c081c-Paper-Conference.pdf)

1. **Hybrid Mamba for Few-Shot Segmentation.,** in NeurIPS, 2024.
*Q. Xu, X. Liu, L. Zhu, G. Lin, C. Long, Z. Li, and R. Zhao.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/86fe62e3b315d2578721562d9fd1a433-Paper-Conference.pdf)

1. **Bridge the Points: Graph-based Few-shot Segment Anything Semantically.,** in NeurIPS, 2024.
*A. Zhang, G. Gao, J. Jiao, C. Liu, and Y. Wei.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/3a2ef31a1e45908901adc0ca853a8faf-Paper-Conference.pdf)

1. **DomainGallery: Few-shot Domain-driven Image Generation by Attribute-centric Finetuning.,** in NeurIPS, 2024.
*Y. Duan, Y. Hong, B. Zhang, J. Lan, H. Zhu, W. Wang, J. Zhang, L. Niu, and L. Zhang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/013d743db3c684957305d32017f13339-Paper-Conference.pdf)

1. **Learning Interaction-aware 3D Gaussian Splatting for One-shot Hand Avatars.,** in NeurIPS, 2024.
*X. Huang, H. Li, W. Liu, X. Liang, Y. Yan, Y. Cheng, and C. Gao.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/19c9708f31ec44b5b1cbd67f91d05d95-Paper-Conference.pdf)

1. **Latent Representation Matters: Human-like Sketches in One-shot Drawing Tasks.,** in NeurIPS, 2024.
*V. Boutin, R. Mukherji, A. Agrawal, S. Muzellec, T. Fel, T. Serre, and R. VanRullen.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/ae90d88755e0eaeb9121712fbac4e8de-Paper-Conference.pdf)

1. **IODA: Instance-Guided One-shot Domain Adaptation for Super-Resolution.,** in NeurIPS, 2024.
*Z. Tang, and Y.-B. Yang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/d4ce6738e84876aa79f13c8bc8b7c5eb-Paper-Conference.pdf)

1. **Few-Shot Adversarial Prompt Learning on Vision-Language Models.,** in NeurIPS, 2024.
*Y. Zhou, X. Xia, Z. Lin, B. Han, and T. Liu.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/05aedcaf4bc6e78a5e22b4cf9114c5e8-Paper-Conference.pdf)

1. **Lightweight Frequency Masker for Cross-Domain Few-Shot Semantic Segmentation.,** in NeurIPS, 2024.
*J. Tong, Y. Zou, Y. Li, and R. Li.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/af536fa4312b5b2e5c541156cd31e93b-Paper-Conference.pdf)

1. **Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis.,** in NeurIPS, 2024.
*R. Peng, W. Xu, L. Tang, L. Leo, J. Jiao, and R. Wang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/b09df3a10e26204136540ca59bc5a646-Paper-Conference.pdf)

1. **FAST: A Dual-tier Few-Shot Learning Paradigm for Whole Slide Image Classification.,** in NeurIPS, 2024.
*K. Fu, X. Luo, L. Qu, S. Wang, Y. Xiong, I. Maglogiannis, L. Gao, and M. Wang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/bdcdf38389d7fcefc73c4c3720217155-Paper-Conference.pdf)

1. **Attention Temperature Matters in ViT-Based Cross-Domain Few-Shot Learning.,** in NeurIPS, 2024.
*Y. Zou, R. Ma, Y. Li, and R. Li.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/d2fe3a5711a6d488da9e9a78b84ee24c-Paper-Conference.pdf)

1. **Meta-Exploiting Frequency Prior for Cross-Domain Few-Shot Learning.,** in NeurIPS, 2024.
*F. Zhou, P. Wang, L. Zhang, Z. Chen, W. Wei, C. Ding, G. Lin, and Y. Zhang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/d3b8ce5e27b1c622d1b3da22b215e59b-Paper-Conference.pdf)

1. **Vision Transformer Off-the-Shelf: A Surprising Baseline for Few-Shot Class-Agnostic Counting,** in AAAI, 2024.
*Z. Wang, L. Xiao, Z. Cao, and H. Lu.*
[paper](https://doi.org/10.1609/aaai.v38i6.28396)

1. **Personalization as a Shortcut for Few-Shot Backdoor Attack against Text-to-Image Diffusion Models,** in AAAI, 2024.
*Y. Huang, F. Juefei-Xu, Q. Guo, J. Zhang, Y. Wu, M. Hu, T. Li, G. Pu, and Y. Liu.*
[paper](https://doi.org/10.1609/aaai.v38i19.30110)

1. **Few-Shot Unsupervised Implicit Neural Shape Representation Learning with Spatial Adversaries.,** in ICML, 2024.
*A. Ouasfi, and A. Boukhayma.*
[paper](https://openreview.net/forum?id=SLqdDWwibH)

1. **Bidirectional Reciprocative Information Communication for Few-Shot Semantic Segmentation.,** in ICML, 2024.
*Y. Liu, J. Han, X. Yao, S. Khan, H. Cholakkal, R. M. Anwer, N. Liu, and F. S. Khan.*
[paper](https://openreview.net/forum?id=uRz9GZN17X)

1. **Few-Shot Character Understanding in Movies as an Assessment to Meta-Learning of Theory-of-Mind.,** in ICML, 2024.
*M. Yu, Q. Wang, S. Zhang, Y. Sang, K. Pu, Z. Wei, H. Wang, L. Xu, J. Li, Y. Yu, and J. Zhou.*
[paper](https://openreview.net/forum?id=ZZ7UKgK4c1)

1. **Learning Causal Domain-Invariant Temporal Dynamics for Few-Shot Action Recognition.,** in ICML, 2024.
*Y. Li, G. Chen, B. Abramowitz, S. Anzellotti, and D. Wei.*
[paper](https://openreview.net/forum?id=LvuuYqU0BW)

1. **DeepCalliFont: Few-Shot Chinese Calligraphy Font Synthesis by Integrating Dual-Modality Generative Models,** in AAAI, 2024.
*Y. Liu, and Z. Lian.*
[paper](https://doi.org/10.1609/aaai.v38i4.28168)

1. **EMOPortraits: Emotion-Enhanced Multimodal One-Shot Head Avatars,** in CVPR, 2024.
*N. Drobyshev, A. B. Casademunt, K. Vougioukas, Z. Landgraf, S. Petridis, and M. Pantic.*
[paper](https://doi.org/10.1109/CVPR52733.2024.00812)

1. **Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation.,** in ICLR, 2025.
*Z. An, G. Sun, Y. Liu, R. Li, M. Wu, M.-M. Cheng, E. Konukoglu, and S. J. Belongie.*
[paper](https://openreview.net/forum?id=jXvwJ51vcK)

1. **Multi-Perspective Data Augmentation for Few-shot Object Detection.,** in ICLR, 2025.
*A.-K. N. Vu, Q.-T. Truong, V.-T. Nguyen, T. D. Ngo, T.-T. Do, and T. V. Nguyen.*
[paper](https://openreview.net/forum?id=qG0WCAhZE0)

1. **Enhancing Identity-Deformation Disentanglement in StyleGAN for One-Shot Face Video Re-Enactment,** in AAAI, 2025.
*Q. Chang, Y.-X. Ding, and K. Zhou.*
[paper](https://doi.org/10.1609/aaai.v39i2.32113)

1. **StyO: Stylize Your Face in Only One-Shot,** in AAAI, 2025.
*B. Li, Z. Zhang, X. Nie, C. Han, Y. Hu, X. Qiu, and T. Guo.*
[paper](https://doi.org/10.1609/aaai.v39i5.32488)

1. **Guided and Variance-Corrected Fusion with One-shot Style Alignment for Large-Content Image Generation,** in AAAI, 2025.
*S. Sun, M. Xian, T. Yao, F. Xu, and L. Capriotti.*
[paper](https://doi.org/10.1609/aaai.v39i7.32764)

1. **One-Shot Reference-based Structure-Aware Image to Sketch Synthesis,** in AAAI, 2025.
*R. Yang, H. Yang, L. Zhao, Q. Lei, M. Dong, K. Ota, and X. Wu.*
[paper](https://doi.org/10.1609/aaai.v39i9.33000)

1. **Enhancing Generalized Few-Shot Semantic Segmentation via Effective Knowledge Transfer,** in AAAI, 2025.
*X. Chen, M. Shi, Z. Zhou, L. He, and S. Tsoka.*
[paper](https://doi.org/10.1609/aaai.v39i2.32225)

1. **Manta: Enhancing Mamba for Few-Shot Action Recognition of Long Sub-Sequence,** in AAAI, 2025.
*W. Huang, J. Zhang, G. Li, L. Zhang, S. Wang, F. Dong, J. Jin, T. Ogawa, and M. Haseyama.*
[paper](https://doi.org/10.1609/aaai.v39i4.32391)

1. **Making Large Vision Language Models to Be Good Few-Shot Learners,** in AAAI, 2025.
*F. Liu, W. Cai, J. Huo, C. Zhang, D. Chen, and J. Zhou.*
[paper](https://doi.org/10.1609/aaai.v39i5.32576)

1. **Multi-Label Few-Shot Image Classification via Pairwise Feature Augmentation and Flexible Prompt Learning,** in AAAI, 2025.
*H. Liu, Y. Wang, X. Zhang, F. Zhang, W. Wang, F. Ma, and H. Yu.*
[paper](https://doi.org/10.1609/aaai.v39i5.32578)

1. **Beyond Pixel and Object: Part Feature as Reference for Few-Shot Video Object Segmentation,** in AAAI, 2025.
*N. Luo, G. Xiong, and T. Zhang.*
[paper](https://doi.org/10.1609/aaai.v39i6.32626)

1. **MVREC: A General Few-shot Defect Classification Model Using Multi-View Region-Context,** in AAAI, 2025.
*S. Lyu, R. Zhang, Z. Ma, F. Liao, D. Mo, and W. Wong.*
[paper](https://doi.org/10.1609/aaai.v39i6.32634)

1. **Few-Shot Fine-Grained Image Classification with Progressively Feature Refinement and Continuous Relationship Modeling,** in AAAI, 2025.
*Z.-X. Ma, Z.-D. Chen, T. Zheng, X. Luo, Z. Jia, and X.-S. Xu.*
[paper](https://doi.org/10.1609/aaai.v39i6.32645)

1. **Foreground-Covering Prototype Generation and Matching for SAM-Aided Few-Shot Segmentation,** in AAAI, 2025.
*S. Park, S. Lee, H. S. Seong, J. Yoo, and J.-P. Heo.*
[paper](https://doi.org/10.1609/aaai.v39i6.32688)

1. **SAM-Aware Graph Prompt Reasoning Network for Cross-Domain Few-Shot Segmentation,** in AAAI, 2025.
*S.-F. Peng, G. Sun, Y. Li, H. Wang, and G.-S. Xie.*
[paper](https://doi.org/10.1609/aaai.v39i6.32695)

1. **Few-Shot Domain Adaptation for Learned Image Compression,** in AAAI, 2025.
*T. Zhang, H. Zhang, Y. Li, L. Li, and D. Liu.*
[paper](https://doi.org/10.1609/aaai.v39i10.33100)

1. **SVasP: Self-Versatility Adversarial Style Perturbation for Cross-Domain Few-Shot Learning,** in AAAI, 2025.
*W. Li, P. Fang, and H. Xue.*
[paper](https://doi.org/10.1609/aaai.v39i15.33676)

1. **Frame Order Matters: A Temporal Sequence-Aware Model for Few-Shot Action Recognition,** in AAAI, 2025.
*B. Li, M. Liu, G. Wang, and Y. Yu.*
[paper](https://doi.org/10.1609/aaai.v39i17.34004)

1. **Spatial Annealing for Efficient Few-shot Neural Rendering,** in AAAI, 2025.
*Y. Xiao, D. Zhai, W. Zhao, K. Jiang, J. Jiang, and X. Liu.*
[paper](https://doi.org/10.1609/aaai.v39i8.32939)

1. **Vision-aware Multimodal Prompt Tuning for Uploadable Multi-source Few-shot Domain Adaptation,** in AAAI, 2025.
*K. Liu, J. Wang, K. He, D. Xu, and X. Zhang.*
[paper](https://doi.org/10.1609/aaai.v39i18.34080)

1. **Reconstruction Target Matters in Masked Image Modeling for Cross-Domain Few-Shot Learning,** in AAAI, 2025.
*R. Ma, Y. Zou, Y. Li, and R. Li.*
[paper](https://doi.org/10.1609/aaai.v39i18.34125)

1. **DiffCLIP: Few-shot Language-driven Multimodal Classifier,** in AAAI, 2025.
*J. Zhang, M. Cao, X. Yang, K. Jiang, and Y. Li.*
[paper](https://doi.org/10.1609/aaai.v39i21.34401)

1. **Text and Image Are Mutually Beneficial: Enhancing Training-Free Few-Shot Classification with CLIP,** in AAAI, 2025.
*Y. Li, J. Guo, L. Qi, W. Li, and Y. Shi.*
[paper](https://doi.org/10.1609/aaai.v39i5.32534)

1. **FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation,** in AAAI, 2025.
*Y. Bo, Y. Zhu, L. Li, and H. Zhang.*
[paper](https://doi.org/10.1609/aaai.v39i2.32184)

1. **Taylor Series-Inspired Local Structure Fitting Network for Few-shot Point Cloud Semantic Segmentation,** in AAAI, 2025.
*C. Wang, S. He, X. Fang, M. Wu, S.-K. Lam, and P. Tiwari.*
[paper](https://doi.org/10.1609/aaai.v39i7.32810)



### Robotics

1. **Towards One Shot Learning by Imitation for Humanoid Robots,** in ICRA, 2010.
*Y. Wu and Y. Demiris.*
[paper](https://spiral.imperial.ac.uk/bitstream/10044/1/12669/4/icra2010.pdf)

1. **Learning Manipulation Actions From a Few Demonstrations,** in ICRA, 2013.
*N. Abdo, H. Kretzschmar, L. Spinello, and C. Stachniss.*
[paper](https://ieeexplore.ieee.org/document/6630734)

1. **Learning Assistive Strategies From a Few User-Robot Interactions: Model-Based Reinforcement Learning Approach,** in ICRA, 2016. 
*M. Hamaya, T. Matsubara, T. Noda, T. Teramae, and J. Morimoto.*
[paper](https://ieeexplore.ieee.org/document/7487509)

1. **One-Shot Imitation Learning,** in NeurIPS, 2017.
*Y. Duan, M. Andrychowicz, B. Stadie, J. Ho, J. Schneider, I. Sutskever, P. Abbeel, and W. Zaremba.*
[paper](https://papers.nips.cc/paper/6709-one-shot-imitation-learning.pdf)

1. **Meta-Learning Language-Guided Policy Learning,** in ICLR, 2019.
*J. D. Co-Reyes, A. Gupta, S. Sanjeev, N. Altieri, J. DeNero, P. Abbeel, and S. Levine.*
[paper](https://openreview.net/forum?id=HkgSEnA5KQ)

1. **Meta Reinforcement Learning With Autonomous Inference of Subtask Dependencies,** in ICLR, 2020.
*S. Sohn, H. Woo, J. Choi, and H. Lee.*
[paper](https://openreview.net/pdf?id=HkgsWxrtPB)

1. **Watch, Try, Learn: Meta-Learning From Demonstrations and Rewards,** in ICLR, 2020.
*A. Zhou, E. Jang, D. Kappler, A. Herzog, M. Khansari, P. Wohlhart, Y. Bai, M. Kalakrishnan, S. Levine, and C. Finn.*
[paper](https://openreview.net/pdf?id=SJg5J6NtDr)

1. **Few-Shot Bayesian Imitation Learning With Logical Program Policies,** in AAAI, 2020.
*T. Silver, K. R. Allen, A. K. Lew, L. P. Kaelbling, and J. Tenenbaum.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6587)

1. **One Solution Is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL,** in NeurIPS, 2020.
*S. Kumar, A. Kumar, S. Levine, and C. Finn.*
[paper](https://proceedings.neurips.cc/paper/2020/file/5d151d1059a6281335a10732fc49620e-Paper.pdf)

1. **Bowtie Networks: Generative Modeling for Joint Few-Shot Recognition and Novel-View Synthesis,** in ICLR, 2021.
*Z. Bao, Y. Wang, and M. Hebert.*
[paper](https://openreview.net/pdf?id=ESG-DMKQKsD)

1. **Demonstration-Conditioned Reinforcement Learning for Few-Shot Imitation,** in ICML, 2021.
*C. R. Dance, J. Perez, and T. Cachet.*
[paper](http://proceedings.mlr.press/v139/dance21a/dance21a.pdf)

1. **Hierarchical Few-Shot Imitation With Skill Transition Models,** in ICLR, 2022.
*K. Hakhamaneshi, R. Zhao, A. Zhan, P. Abbeel, and M. Laskin.*
[paper](https://openreview.net/pdf?id=xKZ4K0lTj_)

1. **Prompting Decision Transformer for Few-Shot Policy Generalization,** in ICML, 2022.
*M. Xu, Y. Shen, S. Zhang, Y. Lu, D. Zhao, J. B. Tenenbaum, and C. Gan.*
[paper](https://proceedings.mlr.press/v162/xu22g/xu22g.pdf)
[code](https://github.com/mxu34/prompt-dt)

1. **Stage Conscious Attention Network (SCAN): A Demonstration-Conditioned Policy for Few-Shot Imitation,** in AAAI, 2022.
*J.-F. Yeh, C.-M. Chung, H.-T. Su, Y.-T. Chen, and W. H. Hsu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20868/20627)

1. **Online Prototype Alignment for Few-Shot Policy Transfer,** in ICML, 2023.
*Q. Yi, R. Zhang, S. Peng, J. Guo, Y. Gao, K. Yuan, R. Chen, S. Lan, X. Hu, Z. Du, X. Zhang, Q. Guo, and Y. Chen.*
[paper](https://proceedings.mlr.press/v202/yi23b/yi23b.pdf)
[code](https://github.com/albertcity/OP)

1. **LLM-planner: Few-Shot Grounded Planning for Embodied Agents With Large Language Models,** in ICCV, 2023.
*C. H. Song, J. Wu, C. Washington, B. M. Sadler, W. Chao, and Y. Su.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Song_LLM-Planner_Few-Shot_Grounded_Planning_for_Embodied_Agents_with_Large_Language_ICCV_2023_paper.pdf)
[code](https://osu-nlp-group.github.io/LLM-Planner/)

1. **Where2Explore: Few-Shot Affordance Learning for Unseen Novel Categories of Articulated Objects,** in NeurIPS, 2023.
*C. Ning, R. Wu, H. Lu, K. Mo, and H. Dong.*
[paper](https://openreview.net/attachment?id=QLllDwizVd&name=pdf)

1. **Skill Machines: Temporal Logic Skill Composition in Reinforcement Learning,** in ICLR, 2024.
*G. N. Tasse, D. Jarvis, S. James, and B. Rosman.*
[paper](https://openreview.net/attachment?id=qiduMcw3CU&name=pdf)

1. **A Conservative Approach for Few-Shot Transfer in Off-Dynamics Reinforcement Learning,** in IJCAI, 2024.
*P. Daoudi, C. Prieur, B. Robu, M. Barlier, and L. D. Santos.*
[paper](https://www.ijcai.org/proceedings/2024/430)


1. **Meta-Controller: Few-Shot Imitation of Unseen Embodiments and Tasks in Continuous Control.,** in NeurIPS, 2024.
*S. Cho, D. Kim, J. Lee, and S. Hong.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/f249db9ab5975586f36df46f8958c008-Paper-Conference.pdf)

1. **AED: Adaptable Error Detection for Few-shot Imitation Policy.,** in NeurIPS, 2024.
*J.-F. Yeh, K.-H. Hung, P.-C. Lo, C.-M. Chung, T.-H. Wu, H.-T. Su, Y.-T. Chen, and W. H. Hsu.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/f74054328beeb0c21a9b8e99da557f5a-Paper-Conference.pdf)

1. **Premier-TACO is a Few-Shot Policy Learner: Pretraining Multitask Representation via Temporal Action-Driven Contrastive Loss.,** in ICML, 2024.
*R. Zheng, Y. Liang, X. Wang, S. Ma, H. D. III, H. Xu, J. Langford, P. Palanisamy, K. S. Basu, and F. Huang.*
[paper](https://openreview.net/forum?id=KSNl7VgeVr)

1. **Predicate Hierarchies Improve Few-Shot State Classification.,** in ICLR, 2025.
*E. Jin, J. Hsu, and J. Wu.*
[paper](https://openreview.net/forum?id=lxu8Vz6cLs)

### Natural Language Processing

1. **High-Risk Learning: Acquiring New Word Vectors From Tiny Data,** in EMNLP, 2017.
*A. Herbelot and M. Baroni.*
[paper](https://www.aclweb.org/anthology/D17-1030.pdf)

1. **Few-Shot Representation Learning for Out-of-Vocabulary Words,** in ACL, 2019.
*Z. Hu, T. Chen, K.-W. Chang, and Y. Sun.*
[paper](https://www.aclweb.org/anthology/P19-1402.pdf)

1. **Learning to Customize Model Structures for Few-Shot Dialogue Generation Tasks,** in ACL, 2020.
*Y. Song, Z. Liu, W. Bi, R. Yan, and M. Zhang.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.517.pdf)

1. **Few-Shot Slot Tagging With Collapsed Dependency Transfer and Label-Enhanced Task-Adaptive Projection Network,** in ACL, 2020.
*Y. Hou, W. Che, Y. Lai, Z. Zhou, Y. Liu, H. Liu, and T. Liu.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.128.pdf)

1. **Meta-Reinforced Multi-Domain State Generator for Dialogue Systems,** in ACL, 2020.
*Y. Huang, J. Feng, M. Hu, X. Wu, X. Du, and S. Ma.*
[paper](https://www.aclweb.org/anthology/2020.acl-main.636.pdf)

1. **Universal Natural Language Processing With Limited Annotations: Try Few-Shot Textual Entailment as a Start,** in EMNLP, 2020.
*W. Yin, N. F. Rajani, D. Radev, R. Socher, and C. Xiong.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.660.pdf)
[code](https://github.com/salesforce/UniversalFewShotNLP)

1. **Simple and Effective Few-Shot Named Entity Recognition With Structured Nearest Neighbor Learning,** in EMNLP, 2020.
*Y. Yang, and A. Katiyar.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.516.pdf)
[code](https://github.com/asappresearch/structshot)

1. **Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference,** in EMNLP, 2020.
*J. Zhang, K. Hashimoto, W. Liu, C. Wu, Y. Wan, P. Yu, R. Socher, and C. Xiong.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.411.pdf)
[code](https://github.com/salesforce/DNNC-few-shot-intent)

1. **Few-Shot Learning for Opinion Summarization,** in EMNLP, 2020.
*A. Bražinskas, M. Lapata, and I. Titov.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.337.pdf)
[code](https://github.com/abrazinskas/FewSum)

1. **Few-Shot Complex Knowledge Base Question Answering via Meta Reinforcement Learning,** in EMNLP, 2020.
*Y. Hua, Y. Li, G. Haffari, G. Qi, and T. Wu.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.469.pdf)
[code](https://github.com/DevinJake/MRL-CQA)

1. **Self-Supervised Meta-Learning for Few-Shot Natural Language Classification Tasks,** in EMNLP, 2020.
*T. Bansal, R. Jha, T. Munkhdalai, and A. McCallum.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.38.pdf)
[code](https://github.com/iesl/metanlp)

1. **Uncertainty-Aware Self-Training for Few-Shot Text Classification,** in NeurIPS, 2020.
*S. Mukherjee, and A. Awadallah.*
[paper](https://proceedings.neurips.cc/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf)
[code](https://github.com/microsoft/UST)

1. **Learning to Extrapolate Knowledge: Transductive Few-Shot Out-of-Graph Link Prediction,** in NeurIPS, 2020:.
*J. Baek, D. B. Lee, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2020/file/0663a4ddceacb40b095eda264a85f15c-Paper.pdf)
[code](https://github.com/JinheonBaek/GEN)

1. **MetaNER: Named Entity Recognition With Meta-Learning,** in WWW, 2020.
*J. Li, S. Shang, and L. Shao.*
[paper](https://dl.acm.org/doi/10.1145/3366423.3380127)

1. **Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data,** in ICLR, 2021.
*J. Pilault, A. E. hattami, and C. Pal.*
[paper](https://openreview.net/pdf?id=de11dbHzAMF)
[code](https://github.com/CAMTL/CA-MTL)

1. **Revisiting Few-Sample BERT Fine-Tuning,** in ICLR, 2021.
*T. Zhang, F. Wu, A. Katiyar, K. Q. Weinberger, and Y. Artzi.*
[paper](https://openreview.net/pdf?id=cO1IH43yUF)
[code](https://pytorch.org/docs/1.4.0/_modules/torch/optim/adamw.html)

1. **Few-Shot Conversational Dense Retrieval,** in SIGIR, 2021.
*S. Yu, Z. Liu, C. Xiong, T. Feng, and Z. Liu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462856)
[code](https://github.com/thunlp/ConvDR)

1. **Few-Shot Language Coordination by Modeling Theory of Mind,** in ICML, 2021.
*H. Zhu, G. Neubig, and Y. Bisk.*
[paper](http://proceedings.mlr.press/v139/zhu21d/zhu21d.pdf)
[code](https://github.com/CLAW-Lab/ToM)

1. **KEML: A Knowledge-Enriched Meta-Learning Framework for Lexical Relation Classification,** in AAAI, 2021.
*C. Wang, M. Qiu, J. Huang, and X. He.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17640/17447)

1. **Few-Shot Learning for Multi-Label Intent Detection,** in AAAI, 2021.
*Y. Hou, Y. Lai, Y. Wu, W. Che, and T. Liu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17541/17348)
[code](https://github.com/AtmaHou/FewShotMultiLabel)

1. **SALNet: Semi-Supervised Few-Shot Text Classification With Attention-Based Lexicon Construction,** in AAAI, 2021.
*J.-H. Lee, S.-K. Ko, and Y.-S. Han.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17558/17365)

1. **Learning From My Friends: Few-Shot Personalized Conversation Systems via Social Networks,** in AAAI, 2021.
*Z. Tian, W. Bi, Z. Zhang, D. Lee, Y. Song, and N. L. Zhang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17638/17445)
[code](https://github.com/tianzhiliang/FewShotPersonaConvData)

1. **Relative and Absolute Location Embedding for Few-Shot Node Classification on Graph,** in AAAI, 2021.
*Z. Liu, Y. Fang, C. Liu, and S. C.H. Hoi.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16551/16358)

1. **Few-Shot Question Answering by Pretraining Span Selection,** in ACL-IJCNLP, 2021.
*O. Ram, Y. Kirstain, J. Berant, A. Globerson, and O. Levy.*
[paper](https://aclanthology.org/2021.acl-long.239.pdf)
[code](https://github.com/oriram/splinter)

1. **A Closer Look at Few-Shot Crosslingual Transfer: The Choice of Shots Matters,** in ACL-IJCNLP, 2021.
*M. Zhao, Y. Zhu, E. Shareghi, I. Vulic, R. Reichart, A. Korhonen, and H. Schütze.*
[paper](https://aclanthology.org/2021.acl-long.447.pdf)
[code](https://github.com/fsxlt)

1. **Learning From Miscellaneous Other-Classwords for Few-Shot Named Entity Recognition,** in ACL-IJCNLP, 2021.
*M. Tong, S. Wang, B. Xu, Y. Cao, M. Liu, L. Hou, and J. Li.*
[paper](https://aclanthology.org/2021.acl-long.487.pdf)
[code](https://github.com/shuaiwa16/OtherClassNER.git)

1. **Distinct Label Representations for Few-Shot Text Classification,** in ACL-IJCNLP, 2021.
*S. Ohashi, J. Takayama, T. Kajiwara, and Y. Arase.*
[paper](https://aclanthology.org/2021.acl-short.105.pdf)
[code](https://github.com/21335732529sky/difference_extractor)

1. **Entity Concept-Enhanced Few-Shot Relation Extraction,** in ACL-IJCNLP, 2021.
*S. Yang, Y. Zhang, G. Niu, Q. Zhao, and S. Pu.*
[paper](https://aclanthology.org/2021.acl-short.124.pdf)
[code](https://github.com/LittleGuoKe/ConceptFERE)

1. **On Training Instance Selection for Few-Shot Neural Text Generation,** in ACL-IJCNLP, 2021.
*E. Chang, X. Shen, H.-S. Yeh, and V. Demberg.*
[paper](https://aclanthology.org/2021.acl-short.2.pdf)
[code](https://gitlab.com/erniecyc/few-selector)

1. **Unsupervised Neural Machine Translation for Low-Resource Domains via Meta-Learning,** in ACL-IJCNLP, 2021.
*C. Park, Y. Tae, T. Kim, S. Yang, M. A. Khan, L. Park, and J. Choo.*
[paper](https://aclanthology.org/2021.acl-long.225.pdf)
[code](https://github.com/papago-lab/MetaGUMT)

1. **Meta-Learning With Variational Semantic Memory for Word Sense Disambiguation,** in ACL-IJCNLP, 2021.
*Y. Du, N. Holla, X. Zhen, C. Snoek, and E. Shutova.*
[paper](https://aclanthology.org/2021.acl-long.409.pdf)
[code](https://github.com/YDU-uva/VSM_WSD)

1. **Multi-Label Few-Shot Learning for Aspect Category Detection,** in ACL-IJCNLP, 2021.
*M. Hu, S. Z. H. Guo, C. Xue, H. Gao, T. Gao, R. Cheng, and Z. Su.*
[paper](https://aclanthology.org/2021.acl-long.495.pdf)

1. **TextSETTR: Few-Shot Text Style Extraction and Tunable Targeted Restyling,** in ACL-IJCNLP, 2021.
*P. Rileya, N. Constantb, M. Guob, G. Kumarc, D. Uthusb, and Z. Parekh.*
[paper](https://aclanthology.org/2021.acl-long.293.pdf)

1. **Few-Shot Text Ranking With Meta Adapted Synthetic Weak Supervision,** in ACL-IJCNLP, 2021.
*S. Sun, Y. Qian, Z. Liu, C. Xiong, K. Zhang, J. Bao, Z. Liu, and P. Bennett.*
[paper](https://aclanthology.org/2021.acl-long.390.pdf)
[code](https://github.com/thunlp/MetaAdaptRank)

1. **PROTAUGMENT: Intent Detection Meta-Learning Through Unsupervised Diverse Paraphrasing,** in ACL-IJCNLP, 2021.
*T. Dopierre, C. Gravier, and W. Logerais.*
[paper](https://aclanthology.org/2021.acl-long.191.pdf)
[code](https://github.com/tdopierre/ProtAugment)

1. **AUGNLG: Few-Shot Natural Language Generation Using Self-Trained Data Augmentation,** in ACL-IJCNLP, 2021.
*X. Xu, G. Wang, Y.-B. Kim, and S. Lee.*
[paper](https://aclanthology.org/2021.acl-long.95.pdf)
[code](https://github.com/XinnuoXu/AugNLG)

1. **Meta Self-Training for Few-Shot Neural Sequence Labeling,** in KDD, 2021.
*Y. Wang, S. Mukherjee, H. Chu, Y. Tu, M. Wu, J. Gao, and A. H. Awadallah.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467235)
[code](https://github.com/microsoft/MetaST)

1. **Knowledge-Enhanced Domain Adaptation in Few-Shot Relation Classification,** in KDD, 2021.
*J. Zhang, J. Zhu, Y. Yang, W. Shi, C. Zhang, and H. Wang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467438)
[code](https://github.com/imJiawen/KEFDA)

1. **Few-Shot Text Classification With Triplet Networks, Data Augmentation, and Curriculum Learning,** in NAACL-HLT, 2021.
*J. Wei, C. Huang, S. Vosoughi, Y. Cheng, and S. Xu.*
[paper](https://aclanthology.org/2021.naacl-main.434.pdf)
[code](https://github.com/jasonwei20/triplet-loss)

1. **Few-Shot Intent Classification and Slot Filling With Retrieved Examples,** in NAACL-HLT, 2021.
*D. Yu, L. He, Y. Zhang, X. Du, P. Pasupat, and Q. Li.*
[paper](https://aclanthology.org/2021.naacl-main.59.pdf)

1. **Non-Parametric Few-Shot Learning for Word Sense Disambiguation,** in NAACL-HLT, 2021.
*H. Chen, M. Xia, and D. Chen.*
[paper](https://aclanthology.org/2021.naacl-main.142.pdf)
[code](https://github.com/princeton-nlp/metric-wsd)

1. **Towards Few-Shot Fact-Checking via Perplexity,** in NAACL-HLT, 2021.
*N. Lee, Y. Bang, A. Madotto, and P. Fung.*
[paper](https://aclanthology.org/2021.naacl-main.158.pdf)

1. **ConVEx: Data-Efficient and Few-Shot Slot Labeling,** in NAACL-HLT, 2021.
*M. Henderson, and I. Vulic.*
[paper](https://aclanthology.org/2021.naacl-main.264.pdf)

1. **Few-Shot Text Generation With Natural Language Instructions,** in EMNLP, 2021.
*T. Schick, and H. Schütze.*
[paper](https://aclanthology.org/2021.emnlp-main.32.pdf)

1. **Towards Realistic Few-Shot Relation Extraction,** in EMNLP, 2021.
*S. Brody, S. Wu, and A. Benton.*
[paper](https://aclanthology.org/2021.emnlp-main.433.pdf)
[code](https://github.com/bloomberg/emnlp21_fewrel)

1. **Few-Shot Emotion Recognition in Conversation With Sequential Prototypical Networks,** in EMNLP, 2021.
*G. Guibon, M. Labeau, H. Flamein, L. Lefeuvre, and C. Clavel.*
[paper](https://aclanthology.org/2021.emnlp-main.549.pdf)
[code](https://github.com/gguibon/protoseq)

1. **Learning Prototype Representations Across Few-Shot Tasks for Event Detection,** in EMNLP, 2021.
*V. Lai, F. Dernoncourt, and T. H. Nguyen.*
[paper](https://aclanthology.org/2021.emnlp-main.427.pdf)

1. **Exploring Task Difficulty for Few-Shot Relation Extraction,** in EMNLP, 2021.
*J. Han, B. Cheng, and W. Lu.*
[paper](https://aclanthology.org/2021.emnlp-main.204.pdf)
[code](https://github.com/hanjiale/hcrp)

1. **Honey or Poison? Solving the Trigger Curse in Few-Shot Event Detection via Causal Intervention,** in EMNLP, 2021.
*J. Chen, H. Lin, X. Han, and L. Sun.*
[paper](https://aclanthology.org/2021.emnlp-main.637.pdf)
[code](https://github.com/chen700564/causalfsed)

1. **Nearest Neighbour Few-Shot Learning for Cross-Lingual Classification,** in EMNLP, 2021.
*M. S. Bari, B. Haider, and S. Mansour.*
[paper](https://aclanthology.org/2021.emnlp-main.131.pdf)

1. **Knowledge-Aware Meta-Learning for Low-Resource Text Classification,** in EMNLP, 2021.
*H. Yao, Y. Wu, M. Al-Shedivat, and E. P. Xing.*
[paper](https://aclanthology.org/2021.emnlp-main.136.pdf)
[code](https://github.com/huaxiuyao/KGML)

1. **Few-Shot Named Entity Recognition: An Empirical Baseline Study,** in EMNLP, 2021.
*J. Huang, C. Li, K. Subudhi, D. Jose, S. Balakrishnan, W. Chen, B. Peng, J. Gao, and J. Han.*
[paper](https://aclanthology.org/2021.emnlp-main.813.pdf)

1. **MetaTS: Meta Teacher-Student Network for Multilingual Sequence Labeling With Minimal Supervision,** in EMNLP, 2021.
*Z. Li, D. Zhang, T. Cao, Y. Wei, Y. Song, and B. Yin.*
[paper](https://aclanthology.org/2021.emnlp-main.255.pdf)

1. **Meta-Lmtc: Meta-Learning for Large-Scale Multi-Label Text Classification,** in EMNLP, 2021.
*R. Wang, X. Su, S. Long, X. Dai, S. Huang, and J. Chen.*
[paper](https://aclanthology.org/2021.emnlp-main.679.pdf)

1. **Ontology-Enhanced Prompt-Tuning for Few-Shot Learning,** in WWW, 2022.
*H. Ye, N. Zhang, S. Deng, X. Chen, H. Chen, F. Xiong, X. Chen, and H. Chen.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511921)

1. **EICO: Improving Few-Shot Text Classification via Explicit and Implicit Consistency Regularization,** in Findings of ACL, 2022.
*L. Zhao, and C. Yao.*
[paper](https://aclanthology.org/2022.findings-acl.283.pdf)

1. **Dialogue Summaries as Dialogue States (DS2), Template-Guided Summarization for Few-Shot Dialogue State Tracking,** in Findings of ACL, 2022.
*J. Shin, H. Yu, H. Moon, A. Madotto, and J. Park.*
[paper](https://aclanthology.org/2022.findings-acl.302.pdf)
[code](https://github.com/jshin49/ds2)

1. **A Few-Shot Semantic Parser for Wizard-of-Oz Dialogues With the Precise Thingtalk Representation,** in Findings of ACL, 2022.
*G. Campagna, S. J. Semnani, R. Kearns, L. J. K. Sato, S. Xu, and M. S. Lam.*
[paper](https://aclanthology.org/2022.findings-acl.317.pdf)

1. **Multi-Stage Prompting for Knowledgeable Dialogue Generation,** in Findings of ACL, 2022.
*Z. Liu, M. Patwary, R. Prenger, S. Prabhumoye, W. Ping, M. Shoeybi, and B. Catanzaro.*
[paper](https://aclanthology.org/2022.findings-acl.104.pdf)
[code](https://github.com/NVIDIA/Megatron-LM)

1. **Few-Shot Named Entity Recognition With Self-Describing Networks,** in ACL, 2022.
*J. Chen, Q. Liu, H. Lin, X. Han, and L. Sun.*
[paper](https://aclanthology.org/2022.acl-long.392.pdf)
[code](https://github.com/chen700564/sdnet)

1. **CLIP Models Are Few-Shot Learners: Empirical Studies on VQA and Visual Entailment,** in ACL, 2022.
*H. Song, L. Dong, W. Zhang, T. Liu, and F. Wei.*
[paper](https://aclanthology.org/2022.acl-long.421.pdf)

1. **CONTaiNER: Few-Shot Named Entity Recognition via Contrastive Learning,** in ACL, 2022.
*S. S. S. Das, A. Katiyar, R. J. Passonneau, and R. Zhang.*
[paper](https://aclanthology.org/2022.acl-long.439.pdf)
[code](https://github.com/psunlpgroup/container)

1. **Few-Shot Controllable Style Transfer for Low-Resource Multilingual Settings,** in ACL, 2022.
*K. Krishna, D. Nathani, X. Garcia, B. Samanta, and P. Talukdar.*
[paper](https://aclanthology.org/2022.acl-long.514.pdf)

1. **Label Semantic Aware Pre-Training for Few-Shot Text Classification,** in ACL, 2022.
*A. Mueller, J. Krone, S. Romeo, S. Mansour, E. Mansimov, Y. Zhang, and D. Roth.*
[paper](https://aclanthology.org/2022.acl-long.570.pdf)

1. **Inverse Is Better! Fast and Accurate Prompt for Few-Shot Slot Tagging,** in Findings of ACL, 2022.
*Y. Hou, C. Chen, X. Luo, B. Li, and W. Che.*
[paper](https://aclanthology.org/2022.findings-acl.53.pdf)

1. **Label Semantics for Few Shot Named Entity Recognition,** in Findings of ACL, 2022.
*J. Ma, M. Ballesteros, S. Doss, R. Anubhai, S. Mallya, Y. Al-Onaizan, and D. Roth.*
[paper](https://aclanthology.org/2022.findings-acl.155.pdf)

1. **Hierarchical Recurrent Aggregative Generation for Few-Shot NLG,** in Findings of ACL, 2022.
*G. Zhou, G. Lampouras, and I. Iacobacci.*
[paper](https://aclanthology.org/2022.findings-acl.170.pdf)

1. **Towards Few-Shot Entity Recognition in Document Images: A Label-Aware Sequence-to-Sequence Framework,** in Findings of ACL, 2022.
*Z. Wang, and J. Shang.*
[paper](https://aclanthology.org/2022.findings-acl.329.pdf)

1. **A Good Prompt Is Worth Millions of Parameters: Low-Resource Prompt-Based Learning for Vision-Language Models,** in ACL, 2022.
*W. Jin, Y. Cheng, Y. Shen, W. Chen, and X. Ren.*
[paper](https://aclanthology.org/2022.acl-long.197.pdf)
[code](https://github.com/woojeongjin/fewvlm)

1. **Generated Knowledge Prompting for Commonsense Reasoning,** in ACL, 2022.
*J. Liu, A. Liu, X. Lu, S. Welleck, P. West, R. L. Bras, Y. Choi, and H. Hajishirzi.*
[paper](https://aclanthology.org/2022.acl-long.225.pdf)
[code](https://github.com/liujch1998/gkp)

1. **End-to-End Modeling via Information Tree for One-Shot Natural Language Spatial Video Grounding,** in ACL, 2022.
*M. Li, T. Wang, H. Zhang, S. Zhang, Z. Zhao, J. Miao, W. Zhang, W. Tan, J. Wang, P. Wang, S. Pu, and F. Wu.*
[paper](https://aclanthology.org/2022.acl-long.596.pdf)

1. **Leveraging Task Transferability to Meta-Learning for Clinical Section Classification With Limited Data,** in ACL, 2022.
*Z. Chen, J. Kim, R. Bhakta, and M. Y. Sir.*
[paper](https://aclanthology.org/2022.acl-long.461.pdf)

1. **Improving Meta-Learning for Low-Resource Text Classification and Generation via Memory Imitation,** in ACL, 2022.
*Y. Zhao, Z. Tian, H. Yao, Y. Zheng, D. Lee, Y. Song, J. Sun, and N. L. Zhang.*
[paper](https://aclanthology.org/2022.acl-long.44.pdf)

1. **A Simple Yet Effective Relation Information Guided Approach for Few-Shot Relation Extraction,** in Findings of ACL, 2022.
*Y. Liu, J. Hu, X. Wan, and T. Chang.*
[paper](https://aclanthology.org/2022.findings-acl.62.pdf)
[code](https://github.com/lylylylylyly/simplefsre)

1. **Decomposed Meta-Learning for Few-Shot Named Entity Recognition,** in Findings of ACL, 2022.
*T. Ma, H. Jiang, Q. Wu, T. Zhao, and C. Lin.*
[paper](https://aclanthology.org/2022.findings-acl.124.pdf)
[code](https://github.com/microsoft/vert-papers)

1. **Meta-Learning for Fast Cross-Lingual Adaptation in Dependency Parsing,** in ACL, 2022.
*A. Langedijk, V. Dankers, P. Lippe, S. Bos, B. C. Guevara, H. Yannakoudakis, and E. Shutova.*
[paper](https://aclanthology.org/2022.acl-long.582.pdf)
[code](https://github.com/annaproxy/udify-metalearning)

1. **Enhancing Cross-Lingual Natural Language Inference by Prompt-Learning From Cross-Lingual Templates,** in ACL, 2022.
*K. Qi, H. Wan, J. Du, and H. Chen.*
[paper](https://aclanthology.org/2022.acl-long.134.pdf)
[code](https://github.com/qikunxun/pct)

1. **Few-Shot Stance Detection via Target-Aware Prompt Distillation,** in SIGIR, 2022.
*Y. Jiang, J. Gao, H. Shen, and X. Cheng.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531979)
[code](https://github.com/jyjulyseven/TAPD)

1. **Relation-Guided Few-Shot Relational Triple Extraction,** in SIGIR, 2022.
*X. Cong, J. Sheng, S. Cui, B. Yu, T. Liu, and B. Wang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531831)

1. **Curriculum Contrastive Context Denoising for Few-Shot Conversational Dense Retrieval,** in SIGIR, 2022.
*K. Mao, Z. Dou, and H. Qian.*
[paper](https://dl.acm.org/doi/10.1145/3477495.3531961)
[code](https://github.com/kyriemao/COTED)

1. **Few-Shot Subgoal Planning With Language Models,** in NAACL, 2022.
*L. Logeswaran, Y. Fu, M. Lee, and H. Lee.*
[paper](https://aclanthology.org/2022.naacl-main.402.pdf)

1. **Template-Free Prompt Tuning for Few-Shot NER,** in NAACL, 2022.
*R. Ma, X. Zhou, T. Gui, Y. Tan, L. Li, Q. Zhang, and X. Huang.*
[paper](https://aclanthology.org/2022.naacl-main.420.pdf)
[code](https://github.com/rtmaww/EntLM/)

1. **Few-Shot Document-Level Relation Extraction,** in NAACL, 2022.
*N. Popovic, and M. Färber.*
[paper](https://aclanthology.org/2022.naacl-main.421.pdf)
[code](https://github.com/nicpopovic/FREDo)

1. **An Enhanced Span-Based Decomposition Method for Few-Shot Sequence Labeling,** in NAACL, 2022.
*P. Wang, R. Xu, T. Liu, Q. Zhou, Y. Cao, B. Chang, and Z. Sui.*
[paper](https://aclanthology.org/2022.naacl-main.369.pdf)
[code](https://github.com/Wangpeiyi9979/ESD)

1. **Automatic Multi-Label Prompting: Simple and Interpretable Few-Shot Classification,** in NAACL, 2022.
*H. Wang, C. Xu, and J. McAuley.*
[paper](https://aclanthology.org/2022.naacl-main.401.pdf)
[code](https://github.com/HanNight/AMuLaP)

1. **On the Effect of Pretraining Corpora on in-Context Few-Shot Learning by a Large-Scale Language Model,** in NAACL, 2022.
*S. Shin, S.-W. Lee, H. Ahn, S. Kim, H. Kim, B. Kim, K. Cho, G. Lee, W. Park, J.-W. Ha, and N. Sung.*
[paper](https://aclanthology.org/2022.naacl-main.380.pdf)

1. **MGIMN: Multi-Grained Interactive Matching Network for Few-Shot Text Classification,** in NAACL, 2022.
*J. Zhang, M. Maimaiti, G. Xing, Y. Zheng, and J. Zhang.*
[paper](https://aclanthology.org/2022.naacl-main.141.pdf)

1. **On the Economics of Multilingual Few-Shot Learning: Modeling the Cost-Performance Trade-Offs of Machine Translated and Manual Data,** in NAACL, 2022.
*K. Ahuja, M. Choudhury, and S. Dandapat.*
[paper](https://aclanthology.org/2022.naacl-main.98.pdf)
[code](https://github.com/kabirahuja2431/PerformanceFunctionAnalysis)

1. **OmniTab: Pretraining With Natural and Synthetic Data for Few-Shot Table-Based Question Answering,** in NAACL, 2022.
*Z. Jiang, Y. Mao, P. He, G. Neubig, and W. Chen.*
[paper](https://aclanthology.org/2022.naacl-main.68.pdf)
[code](https://github.com/jzbjyb/OmniTab)

1. **Fine-Tuning Pre-Trained Language Models for Few-Shot Intent Detection: Supervised Pre-Training and Isotropization,** in NAACL, 2022.
*H. Zhang, H. Liang, Y. Zhang, L.-M. Zhan, X.-M. Wu, X. Lu, and A. Y. Lam.*
[paper](https://aclanthology.org/2022.naacl-main.39.pdf)
[code](https://github.com/fanolabs/isoIntentBert-main)

1. **Embedding Hallucination for Few-Shot Language Fine-Tuning,** in NAACL, 2022.
*Y. Jian, C. Gao, and S. Vosoughi.*
[paper](https://aclanthology.org/2022.naacl-main.404.pdf)
[code](https://github.com/yiren-jian/EmbedHalluc)

1. **Few-Shot Semantic Parsing With Language Models Trained on Code,** in NAACL, 2022.
*R. Shin, and B. V. Durme.*
[paper](https://aclanthology.org/2022.naacl-main.396.pdf)

1. **LEA: Meta Knowledge-Driven Self-Attentive Document Embedding for Few-Shot Text Classification,** in NAACL, 2022.
*S. Hong, and T. Y. Jang.*
[paper](https://aclanthology.org/2022.naacl-main.7.pdf)

1. **Contrastive Learning for Prompt-Based Few-Shot Language Learners,** in NAACL, 2022.
*Y. Jian, C. Gao, and S. Vosoughi.*
[paper](https://aclanthology.org/2022.naacl-main.408.pdf)
[code](https://github.com/yiren-jian/LM-SupCon)

1. **Learn From Relation Information: Towards Prototype Representation Rectification for Few-Shot Relation Extraction,** in NAACL, 2022.
*Y. Liu, J. Hu, X. Wan, and T.-H. Chang.*
[paper](https://aclanthology.org/2022.findings-naacl.139.pdf)
[code](https://github.com/lylylylylyly/PRM-FSRE)

1. **Efficient Few-Shot Fine-Tuning for Opinion Summarization,** in NAACL, 2022.
*A. Brazinskas, R. Nallapati, M. Bansal, and M. Dreyer.*
[paper](https://aclanthology.org/2022.findings-naacl.113.pdf)
[code](https://github.com/amazon-research/adasum)

1. **Improving Few-Shot Image Classification Using Machine- And User-Generated Natural Language Descriptions,** in NAACL, 2022.
*K. Nishida, K. Nishida, and S. Nishioka.*
[paper](https://aclanthology.org/2022.findings-naacl.106.pdf)

1. **RGL: A Simple Yet Effective Relation Graph Augmented Prompt-Based Tuning Approach for Few-Shot Learning,** in NAACL, 2022.
*Y. Wang, X. Tian, H. Xiong, Y. Li, Z. Chen, S. Guo, and D. Dou.*
[paper](https://aclanthology.org/2022.findings-naacl.81.pdf)
[code](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot/RGL)

1. **“Diversity and Uncertainty in Moderation” Are the Key to Data Selection for Multilingual Few-Shot Transfer,** in NAACL, 2022.
*S. Kumar, S. Dandapat, and M. Choudhury.*
[paper](https://aclanthology.org/2022.findings-naacl.78.pdf)

1. **A Generative Language Model for Few-Shot Aspect-Based Sentiment Analysis,** in NAACL, 2022.
*E. Hosseini-Asl, W. Liu, and C. Xiong.*
[paper](https://aclanthology.org/2022.findings-naacl.58.pdf)
[code](https://github.com/salesforce/fewshot_absa)

1. **Improving Few-Shot Relation Classiﬁcation by Prototypical Representation Learning With Deﬁnition Text,** in NAACL, 2022.
*L. Zhenzhen, Y. Zhang, J.-Y. Nie, and D. Li.*
[paper](https://aclanthology.org/2022.findings-naacl.34.pdf)

1. **Few-Shot Self-Rationalization With Natural Language Prompts,** in NAACL, 2022.
*A. Marasovic, I. Beltagy, D. Downey, and M. E. Peters.*
[paper](https://aclanthology.org/2022.findings-naacl.31.pdf)
[code](https://github.com/allenai/feb)

1. **How to Translate Your Samples and Choose Your Shots? Analyzing Translate-Train & Few-Shot Cross-Lingual Transfer,** in NAACL, 2022.
*I. Jundi, and G. Lapesa.*
[paper](https://aclanthology.org/2022.findings-naacl.11.pdf)
[code](https://github.com/imanjundi/cross-lingual-transfer)

1. **LMTurk: Few-Shot Learners as Crowdsourcing Workers in a Language-Model-as-a-Service Framework,** in NAACL, 2022.
*M. Zhao, F. Mi, Y. Wang, M. Li, X. Jiang, Q. Liu, and H. Schuetze.*
[paper](https://aclanthology.org/2022.findings-naacl.51.pdf)
[code](https://github.com/lmturk)

1. **LiST: Lite Prompted Self-Training Makes Efficient Few-Shot Learners,** in NAACL, 2022.
*Y. Wang, S. Mukherjee, X. Liu, J. Gao, A. H. Awadallah, and J. Gao.*
[paper](https://aclanthology.org/2022.findings-naacl.174.pdf)
[code](https://github.com/microsoft/LiST)

1. **Improving in-Context Few-Shot Learning via Self-Supervised Training,** in NAACL, 2022.
*M. Chen, J. Du, R. Pasunuru, T. Mihaylov, S. Iyer, V. Stoyanov, and Z. Kozareva.*
[paper](https://aclanthology.org/2022.naacl-main.260.pdf)

1. **Por Qué Não Utiliser Alla Språk? Mixed Training With Gradient Optimization in Few-Shot Cross-Lingual Transfer,** in NAACL, 2022.
*H. Xu, and K. Murray.*
[paper](https://aclanthology.org/2022.findings-naacl.157.pdf)
[code](https://github.com/fe1ixxu/Mixed-Gradient-Few-Shot)

1. **On the Effectiveness of Sentence Encoding for Intent Detection Meta-Learning,** in NAACL, 2022.
*T. Ma, Q. Wu, Z. Yu, T. Zhao, and C.-Y. Lin.*
[paper](https://aclanthology.org/2022.naacl-main.279.pdf)
[code](https://github.com/microsoft/KC/tree/main/papers/IDML)

1. **Few-Shot Fine-Grained Entity Typing With Automatic Label Interpretation and Instance Generation,** in KDD, 2022.
*J. Huang, Y. Meng, and J. Han.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539443)
[code](https://github.com/teapot123/Fine-Grained-Entity-Typing)

1. **Label-Enhanced Prototypical Network With Contrastive Learning for Multi-Label Few-Shot Aspect Category Detection,** in KDD, 2022.
*H. Liu, F. Zhang, X. Zhang, S. Zhao, J. Sun, H. Yu, and X. Zhang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539340)

1. **Task-Adaptive Few-Shot Node Classification,** in KDD, 2022.
*S. Wang, K. Ding, C. Zhang, C. Chen, and J. Li.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539265)
[code](https://github.com/SongW-SW/TENT)

1. **Diversity Features Enhanced Prototypical Network for Few-Shot Intent Detection,** in IJCAI, 2022.
*F. Yang, X. Zhou, Y. Wang, A. Atawulla, and R. Bi.*
[paper](https://www.ijcai.org/proceedings/2022/0617.pdf)

1. **Function-Words Adaptively Enhanced Attention Networks for Few-Shot Inverse Relation Classification,** in IJCAI, 2022.
*C. Dou, S. Wu, X. Zhang, Z. Feng, and K. Wang.*
[paper](https://www.ijcai.org/proceedings/2022/0407.pdf)
[code](https://github.com/DOU123321/FAEA-FSRC)

1. **Curriculum-Based Self-Training Makes Better Few-Shot Learners for Data-to-Text Generation,** in IJCAI, 2022.
*P. Ke, H. Ji, Z. Yang, Y. Huang, J. Feng, X. Zhu, and M. Huang.*
[paper](https://www.ijcai.org/proceedings/2022/0580.pdf)
[code](https://github.com/kepei1106/CBST)

1. **Graph-Based Model Generation for Few-Shot Relation Extraction,** in EMNLP, 2022.
*W. Li, and T. Qian.*
[paper](https://aclanthology.org/2022.emnlp-main.5.pdf)
[code](https://github.com/NLPWM-WHU/GM_GEN)

1. **Prompt-Based Meta-Learning for Few-Shot Text Classification,** in EMNLP, 2022.
*H. Zhang, X. Zhang, H. Huang, and L. Yu.*
[paper](https://aclanthology.org/2022.emnlp-main.87.pdf)
[code](https://github.com/MGHZHANG/PBML)

1. **Language Models of Code Are Few-Shot Commonsense Learners,** in EMNLP, 2022.
*A. Madaan, S. Zhou, U. Alon, Y. Yang, and G. Neubig.*
[paper](https://aclanthology.org/2022.emnlp-main.90.pdf)
[code](https://github.com/reasoning-machines/CoCoGen)

1. **Large Language Models Are Few-Shot Clinical Information Extractors,** in EMNLP, 2022.
*M. Agrawal, S. Hegselmann, H. Lang, Y. Kim, and D. Sontag.*
[paper](https://aclanthology.org/2022.emnlp-main.130.pdf)

1. **ToKen: Task Decomposition and Knowledge Infusion for Few-Shot Hate Speech Detection,** in EMNLP, 2022.
*B. AlKhamissi, F. Ladhak, S. Iyer, V. Stoyanov, Z. Kozareva, X. Li, P. Fung, L. Mathias, A. Celikyilmaz, and M. Diab.*
[paper](https://aclanthology.org/2022.emnlp-main.136.pdf)

1. **Exploiting Domain-Slot Related Keywords Description for Few-Shot Cross-Domain Dialogue State Tracking,** in EMNLP, 2022.
*Q. Gao, G. Dong, Y. Mou, L. Wang, C. Zeng, D. Guo, M. Sun, and W. Xu.*
[paper](https://aclanthology.org/2022.emnlp-main.157.pdf)

1. **KECP: Knowledge Enhanced Contrastive Prompting for Few-Shot Extractive Question Answering,** in EMNLP, 2022.
*J. Wang, C. Wang, M. Qiu, Q. Shi, H. Wang, J. huang, and M. Gao.*
[paper](https://aclanthology.org/2022.emnlp-main.206.pdf)
[code](https://github.com/alibaba/EasyNLP)

1. **SpanProto: A Two-Stage Span-Based Prototypical Network for Few-Shot Named Entity Recognition,** in EMNLP, 2022.
*J. Wang, C. Wang, C. Tan, M. Qiu, S. Huang, J. huang, and M. Gao.*
[paper](https://aclanthology.org/2022.emnlp-main.227.pdf)
[code](https://github.com/alibaba/EasyNLP)

1. **Few-Shot Query-Focused Summarization With Prefix-Merging,** in EMNLP, 2022.
*R. Yuan, Z. Wang, Z. Cao, and W. Li.*
[paper](https://aclanthology.org/2022.emnlp-main.243.pdf)

1. **Incorporating Relevance Feedback for Information-Seeking Retrieval Using Few-Shot Document Re-Ranking,** in EMNLP, 2022.
*T. Baumgartner, L. F. R. Ribeiro, N. Reimers, and I. Gurevych.*
[paper](https://aclanthology.org/2022.emnlp-main.614.pdf)
[code](https://github.com/UKPLab/incorporating-relevance)

1. **Few-Shot Learning With Multilingual Generative Language Models,** in EMNLP, 2022.
*X. V. Lin, T. Mihaylov, M. Artetxe, T. Wang, S. Chen, D. Simig, M. Ott, N. Goyal, S. Bhosale, J. Du, R. Pasunuru, S. Shleifer, P. S. Koura, V. Chaudhary, B. O'Horo, J. Wang, L. Zettlemoyer, Z. Kozareva, M. Diab, V. Stoyanov, and X. Li.*
[paper](https://aclanthology.org/2022.emnlp-main.616.pdf)
[code](https://github.com/facebookresearch/fairseq/tree/main/examples/xglm)

1. **Don't Stop Fine-Tuning: On Training Regimes for Few-Shot Cross-Lingual Transfer With Multilingual Language Models,** in EMNLP, 2022.
*F. D. Schmidt, I. Vulic, and G. Glavas.*
[paper](https://aclanthology.org/2022.emnlp-main.736.pdf)
[code](https://github.com/fdschmidt93/fsxlt)

1. **Better Few-Shot Relation Extraction With Label Prompt Dropout,** in EMNLP, 2022.
*P. Zhang, and W. Lu.*
[paper](https://aclanthology.org/2022.emnlp-main.471.pdf)
[code](https://github.com/jzhang38/LPD)

1. **A Dual Prompt Learning Framework for Few-Shot Dialogue State Tracking,** in WWW, 2023.
*Y. Yang, W. Lei, P. Huang, J. Cao, J. Li, and T.-S. Chua.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583238)
[code](https://github.com/YANG-Yuting/DPL)

1. **MetaTroll: Few-Shot Detection of State-Sponsored Trolls With Transformer Adapters,** in WWW, 2023.
*L. Tian, X. Zhang, and J. H. Lau.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583417)
[code](https://github.com/ltian678/metatroll-code.git)

1. **ContrastNet: A Contrastive Learning Framework for Few-Shot Text Classification,** in AAAI, 2022.
*J. Chen, R. Zhang, Y. Mao, and J. Xu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21292)

1. **Few-Shot Cross-Lingual Stance Detection With Sentiment-Based Pre-Training,** in AAAI, 2022.
*M. Hardalov, A. Arora, P. Nakov, and I. Augenstein.*
[paper](https://github.com/checkstep/senti-stance)

1. **ALP: Data Augmentation Using Lexicalized PCFGs for Few-Shot Text Classification,** in AAAI, 2022.
*H. H. Kim, D. Woo, S. J. Oh, J.-W. Cha, and Y.-S. Han.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21336/21085)

1. **CINS: Comprehensive Instruction for Few-Shot Learning in Task-Oriented Dialog Systems,** in AAAI, 2022.
*F. Mi, Y. Wang, and Y. Li.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21356/21105)

1. **An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA,** in AAAI, 2022.
*Z. Yang, Z. Gan, J. Wang, X. Hu, Y. Lu, Z. Liu, and L. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20215/19974)

1. **PROMPTAGATOR: Few-Shot Dense Retrieval From 8 Examples,** in ICLR, 2023.
*Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu, K. Hall, and M.-W. Chang.*
[paper](https://openreview.net/pdf?id=gmL46YMpu2J)

1. **QAID: Question Answering Inspired Few-Shot Intent Detection,** in ICLR, 2023.
*A. Yehudai, M. Vetzler, Y. Mass, K. Lazar, D. Cohen, and B. Carmeli.*
[paper](https://openreview.net/pdf?id=gNI4_85Cyve)

1. **CLUR: Uncertainty Estimation for Few-Shot Text Classification With Contrastive Learning,** in KDD, 2023.
*J. He, X. Zhang, S. Lei, A. Alhamadani, F. Chen, B. Xiao, and C.-T. Lu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599276)
[code](https://github.com/he159ok/CLUR_UncertaintyEst_FewShot_TextCls.)

1. **Learning Few-Shot Sample-Set Operations for Noisy Multi-Label Aspect Category Detection,** in IJCAI, 2023.
*S. Zhao, W. Chen, and T. Wang.*
[paper](https://www.ijcai.org/proceedings/2023/0589.pd)

1. **Few-Shot Document-Level Event Argument Extraction,** in ACL, 2023.
*X. Yang, Y. Lu, and L. R. Petzold.*
[paper](https://aclanthology.org/2023.acl-long.446.pdf)
[code](https://github.com/Xianjun-Yang/FewDocAE)

1. **FLamE: Few-Shot Learning From Natural Language Explanations,** in ACL, 2023.
*Y. Zhou, Y. Zhang, and C. Tan.*
[paper](https://aclanthology.org/2023.acl-long.372.pdf)

1. **MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning,** in ACL, 2023.
*Z. Yue, H. Zeng, Y. Zhang, L. Shang, and D. Wang.*
[paper](https://aclanthology.org/2023.acl-long.286.pdf)
[code](https://github.com/Yueeeeeeee/MetaAdapt)

1. **Code4Struct: Code Generation for Few-Shot Event Structure Prediction,** in ACL, 2023.
*X. Wang, S. Li, and H. Ji.*
[paper](https://aclanthology.org/2023.acl-long.202.pdf)
[code](https://github.com/xingyaoww/code4struct.)

1. **MANNER: A Variational Memory-Augmented Model for Cross Domain Few-Shot Named Entity Recognition,** in ACL, 2023.
*J. Fang, X. Wang, Z. Meng, P. Xie, F. Huang, and Y. Jiang.*
[paper](https://aclanthology.org/2023.acl-long.234.pdf)
[code](https://github.com/Alibaba-NLP/MANNER)

1. **Dual Class Knowledge Propagation Network for Multi-Label Few-Shot Intent Detection,** in ACL, 2023.
*F. Zhang, W. Chen, F. Ding, and T. Wang.*
[paper](https://aclanthology.org/2023.acl-long.480.pdf)

1. **Few-Shot Event Detection: An Empirical Study and a Unified View,** in ACL, 2023.
*Y. Ma, Z. Wang, Y. Cao, and A. Sun.*
[paper](https://aclanthology.org/2023.acl-long.628.pdf)
[code](https://github.com/mayubo2333/fewshot_ED)

1. **CodeIE: Large Code Generation Models Are Better Few-Shot Information Extractors,** in ACL, 2023.
*P. Li, T. Sun, Q. Tang, H. Yan, Y. Wu, X. Huang, and X. Qiu.*
[paper](https://aclanthology.org/2023.acl-long.855.pdf)
[code](https://github.com/artpli/CodeIE)

1. **Few-Shot Data-to-Text Generation via Unified Representation and Multi-Source Learning,** in ACL, 2023.
*A. H. Li, M. Shang, E. Spiliopoulou, J. Ma, P. Ng, Z. Wang, B. Min, W. Y. Wang, K. R. McKeown, V. Castelli, D. Roth, and B. Xiang.*
[paper](https://aclanthology.org/2023.acl-long.894.pdf)

1. **Few-Shot in-Context Learning on Knowledge Base Question Answering,** in ACL, 2023.
*T. Li, X. Ma, A. Zhuang, Y. Gu, Y. Su, and W. Chen.*
[paper](https://aclanthology.org/2023.acl-long.385.pdf)
[code](https://github.com/ltl3A87/KB-BINDER)

1. **Linguistic Representations for Fewer-Shot Relation Extraction Across Domains,** in ACL, 2023.
*S. Gururaja, R. Dutt, T. Liao, and C. P. Rosé.*
[paper](https://aclanthology.org/2023.acl-long.414.pdf)
[code](https://github.com/ShoRit/flow_graphs)

1. **Few-Shot Reranking for Multi-Hop QA via Language Model Prompting,** in ACL, 2023.
*M. Khalifa, L. Logeswaran, M. Lee, H. Lee, and L. Wang.*
[paper](https://aclanthology.org/2023.acl-long.885.pdf)
[code](https://github.com/mukhal/PromptRank)

1. **A Domain-Transfer Meta Task Design Paradigm for Few-Shot Slot Tagging,** in AAAI, 2023.
*F. Yang, X. Zhou, Y. Yang, B. Ma, R. Dong, and A. Atawulla.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26626/26398)

1. **Revisiting Sparse Retrieval for Few-Shot Entity Linking,** in EMNLP, 2023.
*Y. Chen, Z. Xu, B. Hu, and M. Zhang.*
[paper](https://aclanthology.org/2023.emnlp-main.789.pdf)
[code](https://github.com/HITsz-TMG/Sparse-Retrieval-Fewshot-EL)

1. **Vicinal Risk Minimization for Few-Shot Cross-Lingual Transfer in Abusive Language Detection,** in EMNLP, 2023.
*G. D. l. P. Sarracén, P. Rosso, R. Litschko, G. Glavaš, and S. Ponzetto.*
[paper](https://aclanthology.org/2023.emnlp-main.248.pdf)

1. **Hypernetwork-Based Decoupling to Improve Model Generalization for Few-Shot Relation Extraction,** in EMNLP, 2023.
*L. Zhang, C. Zhou, F. Meng, J. Su, Y. Chen, and J. Zhou.*
[paper](https://aclanthology.org/2023.emnlp-main.381.pdf)
[code](https://github.com/DeepLearnXMU/FSRE-HDN)

1. **Towards Low-Resource Automatic Program Repair With Meta-Learning and Pretrained Language Models,** in EMNLP, 2023.
*W. Wang, Y. Wang, S. Hoi, and S. Joty.*
[paper](https://aclanthology.org/2023.emnlp-main.430.pdf)
[code](https://github.com/wang-weishi/Meta-APR)

1. **Few-Shot Detection of Machine-Generated Text Using Style Representations,** in ICLR, 2024.
*R. A. R. Soto, K. Koch, A. Khan, B. Y. Chen, M. Bishop, and N. Andrews.*
[paper](https://openreview.net/attachment?id=cWiEN1plhJ&name=pdf)

1. **Improving In-Context Learning via Sequentially Selection and Preference Alignment for Few-Shot Aspect-Based Sentiment Analysis,** in SIGIR, 2024.
*Q. Wang, K. Ding, X. Luo, and R. Xu.*
[paper](https://doi.org/10.1145/3626772.3657932)

1. **FPT: Feature Prompt Tuning for Few-shot Readability Assessment,** in NAACL, 2024.
*Z. Wang, S. Lee, H.-Y. Huang, and Y. Wu.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.16)

1. **BUFFET: Benchmarking Large Language Models for Few-shot Cross-lingual Transfer,** in NAACL, 2024.
*A. Asai, S. Kudugunta, X. Yu, T. Blevins, H. Gonen, M. Reid, Y. Tsvetkov, S. Ruder, and H. Hajishirzi.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.100)

1. **SumTra: A Differentiable Pipeline for Few-Shot Cross-Lingual Summarization,** in NAACL, 2024.
*J. Parnell, I. J. Unanue, and M. Piccardi.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.133)

1. **Enhancing AI Assisted Writing with One-Shot Implicit Negative Feedback,** in EMNLP, 2024.
*B. Towle, and K. Zhou.*
[paper](https://aclanthology.org/2024.emnlp-main.705)

1. **OneNet: A Fine-Tuning Free Framework for Few-Shot Entity Linking via Large Language Model Prompting,** in EMNLP, 2024.
*X. Liu, Y. Liu, K. Zhang, K. Wang, Q. Liu, and E. Chen.*
[paper](https://aclanthology.org/2024.emnlp-main.756)

1. **Preserving Generalization of Language models in Few-shot Continual Relation Extraction,** in EMNLP, 2024.
*Q. Tran, N. X. Thanh, N. H. Anh, N. L. Hai, T. Le, L. V. Ngo, and T. H. Nguyen.*
[paper](https://aclanthology.org/2024.emnlp-main.763)

1. **MinPrompt: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering,** in ACL, 2024.
*X. Chen, J.-Y. Jiang, W.-C. Chang, C.-J. Hsieh, H.-F. Yu, and W. Wang.*
[paper](https://aclanthology.org/2024.acl-long.16)

1. **SparseFit: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations,** in ACL, 2024.
*J. Solano, M. Sanni, O.-M. Camburu, and P. Minervini.*
[paper](https://aclanthology.org/2024.acl-long.113)

1. **Order-Agnostic Data Augmentation for Few-Shot Named Entity Recognition,** in ACL, 2024.
*H. Wang, L. Cheng, W. Zhang, D. W. Soh, and L. Bing.*
[paper](https://aclanthology.org/2024.acl-long.421)

1. **BvSP: Broad-view Soft Prompting for Few-Shot Aspect Sentiment Quad Prediction,** in ACL, 2024.
*Y. Bai, Y. Xie, X. Liu, Y. Zhao, Z. Han, M. Hu, H. Gao, and R. Cheng.*
[paper](https://aclanthology.org/2024.acl-long.460)

1. **Argument Mining in Data Scarce Settings: Cross-lingual Transfer and Few-shot Techniques,** in ACL, 2024.
*A. Yeginbergen, M. Oronoz, and R. Agerri.*
[paper](https://aclanthology.org/2024.acl-long.628)

1. **Domain-Hierarchy Adaptation via Chain of Iterative Reasoning for Few-shot Hierarchical Text Classification,** in IJCAI, 2024.
*K. Ji, P. Wang, W. Ke, G. Li, J. Liu, J. Gao, and Z. Shang.*
[paper](https://www.ijcai.org/proceedings/2024/698)

1. **Meta In-Context Learning Makes Large Language Models Better Zero and Few-Shot Relation Extractors,** in IJCAI, 2024.
*G. Li, P. Wang, J. Liu, Y. Guo, K. Ji, Z. Shang, and Z. Xu.*
[paper](https://www.ijcai.org/proceedings/2024/702)

1. **Variational Hybrid-Attention Framework for Multi-Label Few-Shot Aspect Category Detection,** in AAAI, 2024.
*C. Peng, K. Chen, L. Shou, and G. Chen.*
[paper](https://doi.org/10.1609/aaai.v38i13.29375)

1. **Dialogue for Prompting: A Policy-Gradient-Based Discrete Prompt Generation for Few-Shot Learning,** in AAAI, 2024.
*C. Li, X. Liu, Y. Wang, D. Li, Y. Lan, and C. Shen.*
[paper](https://doi.org/10.1609/aaai.v38i16.29809)

1. **Decoupling Representation and Knowledge for Few-Shot Intent Classification and Slot Filling,** in AAAI, 2024.
*J. Han, Y. Zou, H. Wang, J. Wang, W. Liu, Y. Wu, T. Zhang, and R. Li.*
[paper](https://doi.org/10.1609/aaai.v38i16.29775)

1. **PMRC: Prompt-Based Machine Reading Comprehension for Few-Shot Named Entity Recognition,** in AAAI, 2024.
*J. Huang, D. Yan, and Y. Cai.*
[paper](https://doi.org/10.1609/aaai.v38i16.29791)

1. **Synergistic Anchored Contrastive Pre-training for Few-Shot Relation Extraction,** in AAAI, 2024.
*D. Luo, Y. Gan, R. Hou, R. Lin, Q. Liu, Y. Cai, and W. Gao.*
[paper](https://doi.org/10.1609/aaai.v38i17.29838)

1. **Robust Few-Shot Named Entity Recognition with Boundary Discrimination and Correlation Purification,** in AAAI, 2024.
*X. Xue, C. Zhang, T. Xu, and Z. Niu.*
[paper](https://doi.org/10.1609/aaai.v38i17.29904)

1. **Where It Really Matters: Few-Shot Environmental Conservation Media Monitoring for Low-Resource Languages,** in AAAI, 2024.
*S. Jain, S. S. Keh, S. Chettri, K. Dewan, P. Izquierdo, J. Prussman, P. Shrestha, C. Suárez, Z. R. Shi, L. Li, and F. Fang.*
[paper](https://doi.org/10.1609/aaai.v38i20.30218)

1. **A Closer Look at the CLS Token for Cross-Domain Few-Shot Learning.,** in NeurIPS, 2024.
*Y. Zou, S. Yi, Y. Li, and R. Li.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/9b77f07301b1ef1fe810aae96c12cb7b-Paper-Conference.pdf)

1. **SAHSD: Enhancing Hate Speech Detection in LLM-Powered Web Applications via Sentiment Analysis and Few-Shot Learning,** in WWW, 2025.
*Y. Wang, H. Li, and N. Wei.*
[paper](https://doi.org/10.1145/3696410.3714644)

1. **Active Few-Shot Learning for Text Classification,** in NAACL, 2025.
*S. Ahmadnia, A. Y. Jordehi, M. H. K. Heyran, S. A. Mirroshandel, O. Rambow, and C. Caragea.*
[paper](https://doi.org/10.18653/v1/2025.naacl-long.340)

1. **PromptRefine: Enhancing Few-Shot Performance on Low-Resource Indic Languages with Example Selection from related Example Banks,** in NAACL, 2025.
*S. S. Ghosal, S. Pal, K. Mukherjee, and D. Manocha.*
[paper](https://doi.org/10.18653/v1/2025.naacl-long.17)

1. **Enhancing Discriminative Representation in Similar Relation Clusters for Few-Shot Continual Relation Extraction,** in NAACL, 2025.
*A. D. Le, N. L. Hai, T. X. Nguyen, L. N. Van, N. T. N. Diep, S. Dinh, and T. H. Nguyen.*
[paper](https://doi.org/10.18653/v1/2025.naacl-long.123)

1. **Mutual-pairing Data Augmentation for Fewshot Continual Relation Extraction,** in NAACL, 2025.
*N. H. Anh, Q. Tran, T. X. Nguyen, N. T. N. Diep, L. N. Van, T. H. Nguyen, and T. Le.*
[paper](https://doi.org/10.18653/v1/2025.naacl-long.205)

1. **SynthDetoxM: Modern LLMs are Few-Shot Parallel Detoxification Data Annotators,** in NAACL, 2025.
*D. Moskovskiy, N. Sushko, S. Pletenev, E. Tutubalina, and A. Panchenko.*
[paper](https://doi.org/10.18653/v1/2025.naacl-long.294)

1. **Adversity-aware Few-shot Named Entity Recognition via Augmentation Learning,** in AAAI, 2025.
*L. Huang, H. Liu, Q. Gao, J. Yu, G. Liu, and X. Chen.*
[paper](https://doi.org/10.1609/aaai.v39i23.34588)

1. **Language Models of Code Are Few-Shot Planners and Reasoners for Multi-Document Summarization with Attribution,** in AAAI, 2025.
*A. Nandy, and S. Bandyopadhyay.*
[paper](https://doi.org/10.1609/aaai.v39i23.34676)

1. **Envisioning Class Entity Reasoning by Large Language Models for Few-shot Learning,** in AAAI, 2025.
*M. Liu, F. Wu, B. Li, Z. Lu, Y. Yu, and X. Li.*
[paper](https://doi.org/10.1609/aaai.v39i18.34081)

1. **Adaptive Few-shot Prompting for Machine Translation with Pre-trained Language Models,** in AAAI, 2025.
*L. Tang, J. Qin, W. Ye, H. Tan, and Z. Yang.*
[paper](https://doi.org/10.1609/aaai.v39i24.34712)

1. **Few-Shot, No Problem: Descriptive Continual Relation Extraction,** in AAAI, 2025.
*N. X. Thanh, A. D. Le, Q. Tran, T.-T. Le, L. N. Van, and T. H. Nguyen.*
[paper](https://doi.org/10.1609/aaai.v39i24.34715)

1. **VERO: Verification and Zero-Shot Feedback Acquisition for Few-Shot Multimodal Aspect-Level Sentiment Classification,** in AAAI, 2025.
*K. Sun, H. Wu, B. Shi, S. Mensah, P. Liu, and B. Dong.*
[paper](https://doi.org/10.1609/aaai.v39i24.34707)



### Acoustic Signal Processing

1. **One-Shot Learning of Generative Speech Concepts,** in CogSci, 2014. 
*B. Lake, C.-Y. Lee, J. Glass, and J. Tenenbaum.*
[paper](https://groups.csail.mit.edu/sls/publications/2014/lake-cogsci14.pdf)

1. **Machine Speech Chain With One-Shot Speaker Adaptation,** in Interspeech, 2018.
*A. Tjandra, S. Sakti, and S. Nakamura.* 
[paper](https://ahcweb01.naist.jp/papers/conference/2018/201809_Interspeech_andros-tj/201809_Interspeech_andros-tj_1.paper.pdf)

1. **Investigation of Using Disentangled and Interpretable Representations for One-Shot Cross-Lingual Voice Conversion,** in Interspeech, 2018.
*S. H. Mohammadi and T. Kim.*
[paper](https://isca-speech.org/archive/Interspeech_2018/pdfs/2525.pdf)

1. **Few-Shot Audio Classification With Attentional Graph Neural Networks,** in Interspeech, 2019.
*S. Zhang, Y. Qin, K. Sun, and Y. Lin.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1532.pdf)

1. **One-Shot Voice Conversion With Disentangled Representations by Leveraging Phonetic Posteriorgrams,** in Interspeech, 2019.
*S. H. Mohammadi, and T. Kim.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1798.pdf)

1. **One-Shot Voice Conversion With Global Speaker Embeddings,** in Interspeech, 2019.
*H. Lu, Z. Wu, D. Dai, R. Li, S. Kang, J. Jia, and H. Meng.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2365.pdf)

1. **One-Shot Voice Conversion by Separating Speaker and Content Representations With Instance Normalization,** in Interspeech, 2019.
*J.-C. Chou, and H.-Y. Lee.*
[paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2663.pdf)

1. **Audio2Head: Audio-Driven One-Shot Talking-Head Generation With Natural Head Motion,** in IJCAI, 2021.
*S. Wang, L. Li, Y. Ding, C. Fan, and X. Yu.*
[paper](https://www.ijcai.org/proceedings/2021/0152.pdf)

1. **Few-Shot Low-Resource Knowledge Graph Completion With Multi-View Task Representation Generation,** in KDD, 2023.
*S. Pei, Z. Kou, Q. Zhang, and X. Zhang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599350)
[code](https://github.com/scpei/FLow-MV)

1. **Normalizing Flow-Based Neural Process for Few-Shot Knowledge Graph Completion,** in SIGIR, 2023.
*L. Luo, Y.-F. Li, G. Haffari, and S. Pan.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591743)
[code](https://github.com/RManLuo/NP-FKGC)

1. **PALM: Few-Shot Prompt Learning for Audio Language Models,** in EMNLP, 2024.
*A. Hanif, M. T. Agro, M. A. Qazi, and H. Aldarmaki.*
[paper](https://aclanthology.org/2024.emnlp-main.1030)

1. **Phoneme Hallucinator: One-Shot Voice Conversion via Set Expansion,** in AAAI, 2024.
*S. Shan, Y. Li, A. Banerjee, and J. B. Oliva.*
[paper](https://doi.org/10.1609/aaai.v38i13.29411)

1. **UniAudio 1.5: Large Language Model-Driven Audio Codec is A Few-Shot Audio Task Learner.,** in NeurIPS, 2024.
*D. Yang, H. Guo, Y. Wang, R. Huang, X. Li, X. Tan, X. Wu, and H. Meng.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/6801fa3fd290229efc490ee0cf1c5687-Paper-Conference.pdf)

1. **Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities.,** in ICML, 2024.
*Z. Kong, A. Goel, R. Badlani, W. Ping, R. Valle, and B. Catanzaro.*
[paper](https://openreview.net/forum?id=WYi3WKZjYe)

### Graph Learning

1. **MetaEXP: Interactive Explanation and Exploration of Large Knowledge Graphs,** in WWW, 2018.
*F. Behrens, S. Bischoff, P. Ladenburger, J. Rückin, L. Seidel, F. Stolp, M. Vaichenker, A. Ziegler, D. Mottin, F. Aghaei, E. Müller, M. Preusse, N. Müller, and M. Hunger.*
[paper](https://meta-exp.github.io/resources/paper.pdf)
[code](https://hpi.de/en/mueller/metaex)

1. **Meta Relational Learning for Few-Shot Link Prediction in Knowledge Graphs,** in EMNLP-IJCNLP, 2019.
*M. Chen, W. Zhang, W. Zhang, Q. Chen, and H. Chen.*
[paper](https://www.aclweb.org/anthology/D19-1431.pdf)

1. **Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning Over Few-Shot Relations,** in EMNLP-IJCNLP, 2019.
*X. Lv, Y. Gu, X. Han, L. Hou, J. Li, and Z. Liu.*
[paper](https://www.aclweb.org/anthology/D19-1334.pdf)

1. **Knowledge Graph Transfer Network for Few-Shot Recognition,** in AAAI, 2020.
*R. Chen, T. Chen, X. Hui, H. Wu, G. Li, and L. Lin.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6630)

1. **Few-Shot Knowledge Graph Completion,** in AAAI, 2020.
*C. Zhang, H. Yao, C. Huang, M. Jiang, Z. Li, and N. V. Chawla.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5698)

1. **Adaptive Attentional Network for Few-Shot Knowledge Graph Completion,** in EMNLP, 2020.
*J. Sheng, S. Guo, Z. Chen, J. Yue, L. Wang, T. Liu, and H. Xu.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.131.pdf)
[code](https://github.com/JiaweiSheng/FAAN)

1. **Relational Learning With Gated and Attentive Neighbor Aggregator for Few-Shot Knowledge Graph Completion,** in SIGIR, 2021.
*G. Niu, Y. Li, C. Tang, R. Geng, J. Dai, Q. Liu, H. Wang, J. Sun, F. Huang, and L. Si.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3462925)

1. **Few-Shot Knowledge Validation Using Rules,** in WWW, 2021.
*M. Loster, D. Mottin, P. Papotti, J. Ehmüller, B. Feldmann, and F. Naumann.*
[paper](https://doi.org/10.1145/3442381.3450040)

1. **Graph Learning Regularization and Transfer Learning for Few-Shot Event Detection,** in SIGIR, 2021.
*V. D. Lai, M. V. Nguyen, T. H. Nguyen, and F. Dernoncourt.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463054)
[code](https://github.com/laiviet/ed-fsl)

1. **Graph-Evolving Meta-Learning for Low-Resource Medical Dialogue Generation,** in AAAI, 2021.
*S. Lin, P. Zhou, X. Liang, J. Tang, R. Zhao, Z. Chen, and L. Lin.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17577/17384)

1. **Learning Inter-Entity-Interaction for Few-Shot Knowledge Graph Completion,** in EMNLP, 2022.
*Y. Li, K. Yu, X. Huang, and Y. Zhang.*
[paper](https://aclanthology.org/2022.emnlp-main.524.pdf)
[code](https://github.com/cjlyl/FKGC-CIAN)

1. **Learning to Sample and Aggregate: Few-Shot Reasoning Over Temporal Knowledge Graphs,** in NeurIPS, 2022.
*R. Wang, z. li, D. Sun, S. Liu, J. Li, B. Yin, and T. Abdelzaher.*
[paper](https://openreview.net/pdf?id=1LmgISIDZJ)

1. **Few-Shot Relational Reasoning via Connection Subgraph Pretraining,** in NeurIPS, 2022.
*Q. Huang, H. Ren, and J. Leskovec.*
[paper](https://openreview.net/pdf?id=LvW71lgly25)
[code](https://github.com/snap-stanford/csr)

1. **Spatio-Temporal Graph Few-Shot Learning With Cross-City Knowledge Transfer,** in KDD, 2022.
*B. Lu, X. Gan, W. Zhang, H. Yao, L. Fu, and X. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539281)
[code](https://github.com/RobinLu1209/ST-GFSL)

1. **Graph Few-Shot Learning With Task-Specific Structures,** in NeurIPS, 2022.
*S. Wang, C. Chen, and J. Li.*
[paper](https://openreview.net/pdf?id=3yO3MiSOkH4)
[code](https://github.com/SongW-SW/GLITTER)

1. **Cross-Domain Few-Shot Graph Classification,** in AAAI, 2022.
*and K. Hassani.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25615)

1. **Meta Propagation Networks for Graph Few-Shot Semi-Supervised Learning,** in AAAI, 2022.
*K. Ding, J. Wang, J. Caverlee, and H. Liu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20605/20364)
[code](https://github.com/kaize0409/Meta-PN)

1. **Cross-Domain Few-Shot Graph Classification With a Reinforced Task Coordinator,** in AAAI, 2023.
*Q. Zhang, S. Pei, Q. Yang, C. Zhang, N. V. Chawla, and X. Zhang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25615/25387)

1. **Robust Graph Meta-Learning via Manifold Calibration With Proxy Subgraphs,** in AAAI, 2023.
*Z. Wang, L. Cao, W. Lin, M. Jiang, and K. C. Tan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26776/26548)

1. **Meta-Learning Based Knowledge Extrapolation for Temporal Knowledge Graph,** in WWW, 2023.
*Z. Chen, C. Xu, F. Su, Z. Huang, and Y. Dou.*
[paper](hhttps://dl.acm.org/doi/abs/10.1145/3543507.3583279)

1. **Hierarchical Relational Learning for Few-Shot Knowledge Graph Completion,** in ICLR, 2023.
*H. Wu, J. Yin, B. Rajaratnam, and J. Guo.*
[paper](https://openreview.net/pdf?id=zlwBI2gQL3K)
[code](https://github.com/alexhw15/HiRe)

1. **The Unreasonable Effectiveness of Few-Shot Learning for Machine Translation,** in ICML, 2023.
*X. Garcia, Y. Bansal, C. Cherry, G. F. Foster, M. Krikun, M. Johnson, and O. Firat.*
[paper](https://proceedings.mlr.press/v202/garcia23a/garcia23a.pdf)

1. **Leveraging Transferable Knowledge Concept Graph Embedding for Cold-Start Cognitive Diagnosis,** in SIGIR, 2023.
*W. Gao, H. Wang, Q. Liu, F. Wang, X. Lin, L. Yue, Z. Zhang, R. Lv, and S. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591774)
[code](https://github.com/bigdata-ustc/TechCD)

1. **Virtual Node Tuning for Few-Shot Node Classification,** in KDD, 2023.
*Z. Tan, R. Guo, K. Ding, and H. Liu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599541)

1. **Contrastive Meta-Learning for Few-Shot Node Classification,** in KDD, 2023.
*S. Wang, Z. Tan, H. Liu, and J. Li.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599288)
[code](https://github.com/SongW-SW/COSMIC)

1. **Task-Equivariant Graph Few-Shot Learning,** in KDD, 2023.
*S. Kim, J. Lee, N. Lee, W. Kim, S. Choi, and C. Park.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599515)
[code](https://github.com/sung-won-kim/TEG)

1. **Prompting Large Language Models With Chain-of-Thought for Few-Shot Knowledge Base Question Generation,** in EMNLP, 2023.
*Y. Liang, J. Wang, H. Zhu, L. Wang, W. Qian, and Y. Lan.*
[paper](https://aclanthology.org/2023.emnlp-main.263.pdf)

1. **A Simple but Effective Approach for Unsupervised Few-Shot Graph Classification,** in WWW, 2024.
*Y. Liu, L. Huang, B. Cao, X. Li, F. Giunchiglia, X. Feng, and R. Guan.*
[paper](https://doi.org/10.1145/3589334.3645587)

1. **MetaHKG: Meta Hyperbolic Learning for Few-shot Temporal Reasoning,** in SIGIR, 2024.
*R. Wang, Y. Zhang, J. Li, S. Liu, D. Sun, T. Wang, T. Wang, Y. Chen, D. Kara, and T. F. Abdelzaher.*
[paper](https://doi.org/10.1145/3626772.3657711)

1. **Few-shot Knowledge Graph Relational Reasoning via Subgraph Adaptation,** in NAACL, 2024.
*H. Liu, S. Wang, C. Chen, and J. Li.*
[paper](https://doi.org/10.18653/v1/2024.naacl-long.183)

1. **Few-shot Transfer Learning for Knowledge Base Question Answering: Fusing Supervised Models with In-Context Learning,** in ACL, 2024.
*M. Patidar, R. Sawhney, A. K. Singh, B. Chatterjee, Mausam, and I. Bhattacharya.*
[paper](https://aclanthology.org/2024.acl-long.495)

1. **SymKGQA: Few-Shot Knowledge Graph Question Answering via Symbolic Program Generation and Execution,** in ACL, 2024.
*P. Agarwal, N. Kumar, and S. Bedathur.*
[paper](https://aclanthology.org/2024.acl-long.545)

1. **Context-Aware Adapter Tuning for Few-Shot Relation Learning in Knowledge Graphs,** in EMNLP, 2024.
*L. Ran, Z. Liu, X. Li, and Y. Fang.*
[paper](https://aclanthology.org/2024.emnlp-main.970)

1. **LLM-based Multi-Level Knowledge Generation for Few-shot Knowledge Graph Completion,** in IJCAI, 2024.
*Q. Li, Z. Chen, C. Ji, S. Jiang, and J. Li.*
[paper](https://www.ijcai.org/proceedings/2024/236)

1. **FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering,** in AAAI, 2024.
*Z. Li, S. Fan, Y. Gu, X. Li, Z. Duan, B. Dong, N. Liu, and J. Wang.*
[paper](https://doi.org/10.1609/aaai.v38i17.29823)

1. **Self-Training Based Few-Shot Node Classification by Knowledge Distillation,** in AAAI, 2024.
*Z. Wu, Y. Mo, P. Zhou, S. Yuan, and X. Zhu.*
[paper](https://doi.org/10.1609/aaai.v38i14.29530)

1. **Fast Graph Sharpness-Aware Minimization for Enhancing and Accelerating Few-Shot Node Classification.,** in NeurIPS, 2024.
*Y. Luo, Y. Chen, S. Qiu, Y. Wang, C. Zhang, Y. Zhou, X. Cao, and J. Tang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/eeab2e00835c71d64458ad1821e05664-Paper-Conference.pdf)

1. **Pre-Training and Prompting for Few-Shot Node Classification on Text-Attributed Graphs,** in KDD, 2024.
*H. Zhao, B. Yang, Y. Cen, J. Ren, C. Zhang, Y. Dong, E. Kharlamov, S. Zhao, and J. Tang.*
[paper](https://doi.org/10.1145/3637528.3671952)

1. **Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks,** in WWW, 2024.
*H. Liu, J. Feng, L. Kong, D. Tao, Y. Chen, and M. Zhang.*
[paper](https://doi.org/10.1145/3589334.3645367)

1. **SMUG: Sand Mixing for Unobserved Class Detection in Graph Few-Shot Learning,** in WWW, 2024.
*C. Wang, X. Nie, J. Chen, P. Wang, J. Zhao, and X. Guan.*
[paper](https://doi.org/10.1145/3589334.3645466)

1. **HGPrompt: Bridging Homogeneous and Heterogeneous Graphs for Few-Shot Prompt Learning,** in AAAI, 2024.
*X. Yu, Y. Fang, Z. Liu, and X. Zhang.*
[paper](https://doi.org/10.1609/aaai.v38i15.29596)

1. **Enhancing Unsupervised Graph Few-shot Learning via Set Functions and Optimal Transport,** in KDD, 2025.
*Y. Liu, F. Giunchiglia, X. Li, L. Huang, X. Feng, and R. Guan.*
[paper](https://doi.org/10.1145/3690624.3709208)

1. **Grasp the Key Takeaways from Source Domain for Few Shot Graph Domain Adaptation,** in WWW, 2025.
*X. Lv, J. Chen, M. Li, Y. Sui, Z. Liu, and B. Liao.*
[paper](https://doi.org/10.1145/3696410.3714743)

1. **Dual-level Mixup for Graph Few-shot Learning with Fewer Tasks,** in WWW, 2025.
*Y. Liu, M. Li, F. Giunchiglia, L. Huang, X. Li, X. Feng, and R. Guan.*
[paper](https://doi.org/10.1145/3696410.3714905)

1. **Beyond Single Tabs: A Transformative Few-Shot Approach to Multi-Tab Website Fingerprinting Attacks,** in WWW, 2025.
*W. Meng, C. Ma, M. Ding, C. Ge, Y. Qian, and T. Xiang.*
[paper](https://doi.org/10.1145/3696410.3714811)

1. **Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs,** in AAAI, 2025.
*J. Yu, Y. Ren, C. Gong, J. Tan, X. Li, and X. Zhang.*
[paper](https://doi.org/10.1609/aaai.v39i12.33428)

1. **Normalize Then Propagate: Efficient Homophilous Regularization for Few-Shot Semi-Supervised Node Classification,** in AAAI, 2025.
*B. Zhang, M. Chen, J. Song, S. Li, J. Zhang, and C. Wang.*
[paper](https://doi.org/10.1609/aaai.v39i12.33437)

1. **Unlocking the Potential of Black-box Pre-trained GNNs for Graph Few-shot Learning,** in AAAI, 2025.
*Q. Zhang, S. Pei, Y. Fang, and X. Zhang.*
[paper](https://doi.org/10.1609/aaai.v39i21.34407)

1. **ReCDAP: Relation-based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion,** in SIGIR, 2025.
*J. Kim, C. Heo, and J. Jung.*
[paper](https://doi.org/10.1145/3726302.3730241)

### Recommendation

1. **A Meta-Learning Perspective on Cold-Start Recommendations for Items,** in NeurIPS, 2017.
*M. Vartak, A. Thiagarajan, C. Miranda, J. Bratman, and H. Larochelle.*
[paper](https://papers.nips.cc/paper/7266-a-meta-learning-perspective-on-cold-start-recommendations-for-items.pdf)

1. **MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation,** in KDD, 2019.
*H. Lee, J. Im, S. Jang, H. Cho, and S. Chung.*
[paper](https://arxiv.org/pdf/1908.00413.pdf)
[code](https://github.com/hoyeoplee/MeLU)

1. **Sequential Scenario-Specific Meta Learner for Online Recommendation,** in KDD, 2019.
*Z. Du, X. Wang, H. Yang, J. Zhou, and J. Tang.*
[paper](https://arxiv.org/pdf/1906.00391.pdf)
[code](https://github.com/THUDM/ScenarioMeta)

1. **Few-Shot Learning for New User Recommendation in Location-Based Social Networks,** in WWW, 2020.
*R. Li, X. Wu, X. Chen, and W. Wang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3366423.3379994)

1. **MAMO: Memory-Augmented Meta-Optimization for Cold-Start Recommendation,** in KDD, 2020.
*M. Dong, F. Yuan, L. Yao, X. Xu, and L. Zhu.*
[paper](https://arxiv.org/pdf/2007.03183.pdf)
[code](https://github.com/dongmanqing/Code-for-MAMO)

1. **Meta-Learning on Heterogeneous Information Networks for Cold-Start Recommendation,** in KDD, 2020.
*Y. Lu, Y. Fang, and C. Shi.*
[paper](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6158&context=sis_research)
[code](https://github.com/rootlu/MetaHIN)

1. **MetaSelector: Meta-Learning for Recommendation With User-Level Adaptive Model Selection,** in WWW, 2020.
*M. Luo, F. Chen, P. Cheng, Z. Dong, X. He, J. Feng, and Z. Li.*
[paper](https://arxiv.org/pdf/2001.10378v1.pdf)

1. **Fast Adaptation for Cold-Start Collaborative Filtering With Meta-Learning,** in ICDM, 2020.
*T. Wei, Z. Wu, R. Li, Z. Hu, F. Feng, X. H. Sun, and W. Wang.*
[paper](https://ieeexplore.ieee.org/document/9338389)

1. **Meta-Learning for Query Conceptualization at Web Scale,** in KDD, 2020.
*F. X. Han, D. Niu, H. Chen, W. Guo, S. Yan, and B. Long.*
[paper](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/fhan-KDD20.pdf)

1. **Meta-Learning for Query Conceptualization at Web Scale,** in KDD, 2020.
*F. X. Han, D. Niu, H. Chen, W. Guo, S. Yan, and B. Long.*
[paper](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/fhan-KDD20.pdf)

1. **Preference-Adaptive Meta-Learning for Cold-Start Recommendation,** in IJCAI, 2021.
*L. Wang, B. Jin, Z. Huang, H. Zhao, D. Lian, Q. Liu, and E. Chen.*
[paper](https://www.ijcai.org/proceedings/2021/0222.pdf)

1. **A Dynamic Meta-Learning Model for Time-Sensitive Cold-Start Recommendations,** in AAAI, 2022.
*K. P. Neupane, E. Zheng, Y. Kong, and Q. Yu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20756/20515)

1. **SMINet: State-Aware Multi-Aspect Interests Representation Network for Cold-Start Users Recommendation,** in AAAI, 2022.
*W. Tao, Y. Li, L. Li, Z. Chen, H. Wen, P. Chen, T. Liang, and Q. Lu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20824)
[code](https://github.com/wanjietao/Fliggy-SMINet-AAAI2022)

1. **Recognizing Medical Search Query Intent by Few-Shot Learning,** in SIGIR, 2022.
*Y. Wang, S. Wang, L. Yanyan, and D. Dou.*
[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531789)
[code](https://github.com/tata1661/MEDIC-SIGIR22)

1. **Meta-Learning Helps Personalized Product Search,** in WWW, 2022.
*B. Wu, Z. Meng, Q. Zhang, and S. Liang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512036)

1. **Alleviating Cold-Start Problem in CTR Prediction With a Variational Embedding Learning Framework,** in WWW, 2022.
*X. Xu, C. Yang, Q. Yu, Z. Fang, J. Wang, C. Fan, Y. He, C. Peng, Z. Lin, and J. Shao.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512048)

1. **PNMTA: A Pretrained Network Modulation and Task Adaptation Approach for User Cold-Start Recommendation,** in WWW, 2022.
*H. Pang, F. Giunchiglia, X. Li, R. Guan, and X. Feng.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511963)

1. **Few-Shot News Recommendation via Cross-Lingual Transfer,** in WWW, 2023.
*T. Guo, L. Yu, B. Shihada, and X. Zhang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583383)
[code](https://github.com/taichengguo/Few-shot-NewsRec)

1. **ColdNAS: Search to Modulate for User Cold-Start Recommendation,** in WWW, 2023.
*S. Wu, Y. Wang, Q. Jing, D. Dong, D. Dou, and Q. Yao.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3543507.3583344)
[code](https://github.com/LARS-research/ColdNAS)

1. **Contrastive Collaborative Filtering for Cold-Start Item Recommendation,** in WWW, 2023.
*Z. Zhou, L. Zhang, and N. Yang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583383)
[code](https://github.com/zzhin/CCFCRec)

1. **Bootstrapping Contrastive Learning Enhanced Music Cold-Start Matching,** in WWW, 2023.
*X. Zhao, Y. Zhang, Q. Xiao, Y. Ren, and Y. Yang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3543873.3584626)

1. **M2EU: Meta Learning for Cold-Start Recommendation via Enhancing User Preference Estimation,** in SIGIR, 2023.
*Z. Wu, and X. Zhou.*
[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591719)
[code](https://github.com/zhenchaowu/M2EU)

1. **TAML: Time-Aware Meta Learning for Cold-Start Problem in News Recommendation,** in SIGIR, 2023.
*J. Li, Y. Zhang, X. Lin, X. Yang, G. Zhou, L. Li, H. Chen, and J. Zhou.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3592068)

1. **Uncertainty-Aware Consistency Learning for Cold-Start Item Recommendation,** in SIGIR, 2023.
*T. Liu, C. Gao, Z. Wang, D. Li, J. Hao, D. Jin, and Y. Li.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3592078)

1. **DCBT: A Simple but Effective Way for Unified Warm and Cold Recommendation,** in SIGIR, 2023.
*J. Yang, L. Zhang, Y. He, K. Ding, Z. Huan, X. Zhang, and L. Mo.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591856)

1. **A Preference Learning Decoupling Framework for User Cold-Start Recommendation,** in SIGIR, 2023.
*C. Wang, Y. Zhu, A. Sun, Z. Wang, and K. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591627)

1. **Aligning Distillation for Cold-Start Item Recommendation,** in SIGIR, 2023.
*F. Huang, Z. Wang, X. Huang, Y. Qian, Z. Li, and H. Chen.*
[paper](https://dl.acm.org/doi/10.1145/3539618.3591732)
[code](https://github.com/zfnWong/ALDI)

1. **Content-based Graph Reconstruction for Cold-start Item Recommendation,** in SIGIR, 2024.
*J. Kim, E. Kim, K. Yeo, Y. Jeon, C. Kim, S. Lee, and J. Lee.*
[paper](https://doi.org/10.1145/3626772.3657801)

1. **CMCLRec: Cross-modal Contrastive Learning for User Cold-start Sequential Recommendation,** in SIGIR, 2024.
*X. Xu, H. Dong, L. Qi, X. Zhang, H. Xiang, X. Xia, Y. Xu, and W. Dou.*
[paper](https://doi.org/10.1145/3626772.3657839)

1. **Label Hierarchical Structure-Aware Multi-Label Few-Shot Intent Detection via Prompt Tuning,** in SIGIR, 2024.
*X. Zhang, X. Li, H. Liu, X. Liu, and X. Zhang.*
[paper](https://doi.org/10.1145/3626772.3657947)

1. **Cold Start or Hot Start? Robust Slow Start in Congestion Control with A Priori Knowledge for Mobile Web Services,** in WWW, 2024.
*J. Zhang, H. Tong, E. Dong, X. Qian, M. Xu, X. Li, and Z. Meng.*
[paper](https://doi.org/10.1145/3589334.3645393)

1. **Cold-start Bundle Recommendation via Popularity-based Coalescence and Curriculum Heating,** in WWW, 2024.
*H. Jeon, J.-e. Lee, J. Yun, and U. Kang.*
[paper](https://doi.org/10.1145/3589334.3645377)

1. **Could Small Language Models Serve as Recommenders? Towards Data-centric Cold-start Recommendation,** in WWW, 2024.
*X. Wu, H. Zhou, Y. Shi, W. Yao, X. Huang, and N. Liu.*
[paper](https://doi.org/10.1145/3589334.3645494)

1. **When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User Interactions,** in WWW, 2024.
*C. Zhang, G. Long, T. Zhou, Z. Zhang, P. Yan, and B. Yang.*
[paper](https://doi.org/10.1145/3589334.3645525)

1. **LiMAML: Personalization of Deep Recommender Models via Meta Learning,** in KDD, 2024.
*R. Wang, P. Prabhakar, G. Srivastava, T. Wang, Z. S. Jalali, V. Bharill, Y. Ouyang, A. Nigam, D. Venugopalan, A. Gupta, F. Borisyuk, S. S. Keerthi, and A. Muralidharan.*
[paper](https://doi.org/10.1145/3637528.3671599)

1. **Inductive Modeling for Realtime Cold Start Recommendations,** in KDD, 2024.
*C. Zuo, J. Castaldo, H. Zhu, H. Zhang, J. Liu, Y. Ou, and X. Kong.*
[paper](https://doi.org/10.1145/3637528.3671588)

1. **LARP: Language Audio Relational Pre-training for Cold-Start Playlist Continuation,** in KDD, 2024.
*R. Salganik, X. Liu, Y. Ma, J. Kang, and T.-S. Chua.*
[paper](https://doi.org/10.1145/3637528.3671772)

1. **Warming Up Cold-Start CTR Prediction by Learning Item-Specific Feature Interactions,** in KDD, 2024.
*Y. Wang, H. Piao, D. Dong, Q. Yao, and J. Zhou.*
[paper](https://doi.org/10.1145/3637528.3671784)

1. **Temporally and Distributionally Robust Optimization for Cold-Start Recommendation,** in AAAI, 2024.
*X. Lin, W. Wang, J. Zhao, Y. Li, F. Feng, and T.-S. Chua.*
[paper](https://doi.org/10.1609/aaai.v38i8.28721)

1. **Preference Aware Dual Contrastive Learning for Item Cold-Start Recommendation,** in AAAI, 2024.
*W. Wang, B. Liu, L. Shan, C. Sun, B. Chen, and J. Guan.*
[paper](https://doi.org/10.1609/aaai.v38i8.28763)

1. **Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning,** in KDD, 2025.
*Y. Luo, Y. Jiang, Y. Jiang, G. Chen, J. Wang, K. Bian, P. Li, and Q. Zhang.*
[paper](https://doi.org/10.1145/3690624.3709336)

1. **Diffusion-Inspired Cold Start with Sufficient Prior in Computerized Adaptive Testing,** in KDD, 2025.
*H. Ma, A. Xia, C. Wang, H. Wang, and X. Zhang.*
[paper](https://doi.org/10.1145/3690624.3709317)

1. **Personalized Federated Recommendation for Cold-Start Users via Adaptive Knowledge Fusion,** in WWW, 2025.
*Y. Li, Y. Shan, Y. Liu, H. Wang, W. Wang, Y. Wang, and R. Li.*
[paper](https://doi.org/10.1145/3696410.3714635)

1. **BayesCNS: A Unified Bayesian Approach to Address Cold Start and Non-Stationarity in Search Systems at Scale,** in AAAI, 2025.
*R. Ardywibowo, R. Sunki, S. T. L. Kuo, and S. Nayak.*
[paper](https://doi.org/10.1609/aaai.v39i1.31975)

1. **Feature-Structure Adaptive Completion Graph Neural Network for Cold-start Recommendation,** in AAAI, 2025.
*S. Lei, X. Chang, Z. Yu, D. He, C. Huo, J. Wang, and D. Jin.*
[paper](https://doi.org/10.1609/aaai.v39i11.33309)

1. **DisCo: Graph-Based Disentangled Contrastive Learning for Cold-Start Cross-Domain Recommendation,** in AAAI, 2025.
*H. Li, Y. Wang, Z. Xiao, J. Yang, C. Zhou, M. Zhang, and W. Ju.*
[paper](https://doi.org/10.1609/aaai.v39i11.33312)

1. **Domain-Level Disentanglement Framework Based on Information Enhancement for Cross-Domain Cold-Start Recommendation,** in AAAI, 2025.
*N. Rong, F. Xiong, S. Pan, G. Luo, J. Wu, and L. Wang.*
[paper](https://doi.org/10.1609/aaai.v39i12.33361)

1. **Counterfactual Task-augmented Meta-learning for Cold-start Sequential Recommendation,** in AAAI, 2025.
*Z. Wang, J. Pan, X. Zhao, J. Liang, C. Feng, and K. Yao.*
[paper](https://doi.org/10.1609/aaai.v39i12.33396)

1. **Addressing Cold-Start Problem in Click-Through Rate Prediction via Supervised Diffusion Modeling,** in AAAI, 2025.
*W. Zhu, L. Wang, and J. Wu.*
[paper](https://doi.org/10.1609/aaai.v39i12.33469)

1. **GRAIN: Group-Reinforced Adaptive Interaction Network for Cold-Start CTR Prediction in E-commerce Search,** in SIGIR, 2025.
*W. Bao, H. Chen, B. Lin, T. Zhang, and C. Huo.*
[paper](https://doi.org/10.1145/3726302.3731947)

1. **Reinforcement Learning for Effective Few-Shot Ranking,** in SIGIR, 2025.
*S. Soleimany, S. Ebrahimi, S. Seyedsalehi, F. Zarrinkalam, and E. Bagheri.*
[paper](https://doi.org/10.1145/3726302.3730243)

### Anomaly Detection

1. **Few-Shot Scene-Adaptive Anomaly Detection,** in ECCV, 2020.
*Y. Lu, F. Yu, M. K. K. Reddy, and Y. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500120.pdf)
[code](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection)

1. **Few-Shot Network Anomaly Detection via Cross-Network Meta-Learning,** in WWW, 2021.
*K. Ding, Q. Zhou, H. Tong, and H. Liu.*
[paper](https://doi.org/10.1145/3442381.3449922)

1. **A Hierarchical Transformation-Discriminating Generative Model for Few Shot Anomaly Detection,** in ICCV, 2021.
*S. Sheynin, S. Benaim, and L. Wolf.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sheynin_A_Hierarchical_Transformation-Discriminating_Generative_Model_for_Few_Shot_Anomaly_Detection_ICCV_2021_paper.pdf)

1. **FastRecon: Few-Shot Industrial Anomaly Detection via Fast Feature Reconstruction,** in ICCV, 2023.
*Z. Fang, X. Wang, H. Li, J. Liu, Q. Hu, and J. Xiao.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)
[code](https://github.com/FzJun26th/FastRecon)

1. **Pushing the Limits of Few-Shot Anomaly Detection in Industry Vision: GraphCore,** in ICLR, 2023.
*G. Xie, J. Wang, J. Liu, F. Zheng, and Y. Jin.*
[paper](https://openreview.net/pdf?id=xzmqxHdZAwO)

1. **PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection,** in CVPR, 2024.
*X. Li, Z. Zhang, X. Tan, C. Chen, Y. Qu, Y. Xie, and L. Ma.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01594)

1. **Toward Generalist Anomaly Detection via In-Context Residual Learning with Few-Shot Sample Prompts,** in CVPR, 2024.
*J. Zhu, and G. Pang.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01688)

1. **Graph Anomaly Detection with Few Labels: A Data-Centric Approach,** in KDD, 2024.
*X. Ma, R. Li, F. Liu, K. Ding, J. Yang, and J. Wu.*
[paper](https://doi.org/10.1145/3637528.3671929)

1. **One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection.,** in NeurIPS, 2024.
*Y. Li, S. Zhang, K. Li, and Q. Lao.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/8f4477b086a9c97e30d1a0621ea6b2f5-Paper-Conference.pdf)

1. **CLIP-FSAC: Boosting CLIP for Few-Shot Anomaly Classification with Synthetic Anomalies,** in IJCAI, 2024.
*Z. Zuo, Y. Wu, B. Li, J. Dong, Y. Zhou, L. Zhou, Y. Qu, and Z. Wu.*
[paper](https://www.ijcai.org/proceedings/2024/203)

1. **One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning.,** in ICLR, 2025.
*W. Lv, Q. Su, and W. Xu.*
[paper](https://openreview.net/forum?id=Zzs3JwknAY)

1. **Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection,** in AAAI, 2025.
*F. Tao, G.-S. Xie, F. Zhao, and X. Shu.*
[paper](https://doi.org/10.1609/aaai.v39i7.32790)

### AI for Science

1. **Low Data Drug Discovery With One-Shot Learning,** in ACS Central Science, 2017.
*H. Altae-Tran, B. Ramsundar, A. S. Pappu, and V. Pande.* 
[paper](https://arxiv.org/abs/1611.03199)

1. **Few-Shot Graph Learning for Molecular Property Prediction,** in WWW, 2021.
*Z. Guo, C. Zhang, W. Yu, J. Herr, O. Wiest, M. Jiang, and N. V. Chawla.*
[paper](https://doi.org/10.1145/3442381.3450112)
[code](https://github.com/zhichunguo/Meta-MGNN)

1. **Property-Aware Relation Networks for Few-Shot Molecular Property Prediction,** in NeurIPS, 2021.
*Y. Wang, A. Abuduweili, Q. Yao, and D. Dou.*
[paper](https://proceedings.neurips.cc/paper/2021/file/91bc333f6967019ac47b49ca0f2fa757-Paper.pdf)
[code](https://github.com/tata1661/PAR-NeurIPS21)

1. **Context-Enriched Molecule Representations Improve Few-Shot Drug Discovery,** in ICLR, 2023.
*J. Schimunek, P. Seidl, L. Friedrich, D. Kuhn, F. Rippmann, S. Hochreiter, and G. Klambauer.*
[paper](https://openreview.net/pdf?id=XrMWUuEevr)
[code](https://github.com/ml-jku/MHNfs)

1. **PACIA: Parameter-Efficient Adapter for Few-Shot Molecular Property Prediction,** in IJCAI, 2024.
*S. Wu, Y. Wang, and Q. Yao.*
[paper](https://www.ijcai.org/proceedings/2024/576)

1. **Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction.,** in NeurIPS, 2024.
*Q. Liu, S. Liu, X. Sun, S. Wu, and L. Wang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/82a9fb94035dad3ec007de4ad13c6748-Paper-Conference.pdf)

1. **Cross-Gate MLP with Protein Complex Invariant Embedding Is a One-Shot Antibody Designer,** in AAAI, 2024.
*C. Tan, Z. Gao, L. Wu, J. Xia, J. Zheng, X. Yang, Y. Liu, B. Hu, and S. Z. Li.*
[paper](https://doi.org/10.1609/aaai.v38i14.29445)

1. **Cross-Modal Few-Shot Learning with Second-Order Neural Ordinary Differential Equations,** in AAAI, 2025.
*Y. Zhang, C.-W. Cheng, J. He, Z. He, C.-B. Schönlieb, Y. Chen, and A. I. Avilés-Rivero.*
[paper](https://doi.org/10.1609/aaai.v39i10.33118)

1. **UniMatch: Universal Matching from Atom to Task for Few-Shot Drug Discovery.,** in ICLR, 2025.
*R. Li, M. Li, W. Liu, Y. Zhou, X. Zhou, Y. Yao, Q. Zhang, and H. Chen.*
[paper](https://openreview.net/forum?id=v9EjwMM55Y)


### AI for Healthcare

1. **MetaPred: Meta-Learning for Clinical Risk Prediction With Limited Patient Electronic Health Records,** in KDD, 2019.
*X. S. Zhang, F. Tang, H. H. Dodge, J. Zhou, and F. Wang.*
[paper](https://arxiv.org/pdf/1905.03218.pdf)
[code](https://github.com/sheryl-ai/MetaPred)

1. **AffnityNet: Semi-Supervised Few-Shot Learning for Disease Type Prediction,** in AAAI, 2019.
*T. Ma, and A. Zhang.*
[paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3898/3776)

1. **Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation,** in CVPR, 2019.
*A. Zhao, G. Balakrishnan, F. Durand, J. V. Guttag, and A. V. Dalca.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Data_Augmentation_Using_Learned_Transformations_for_One-Shot_Medical_Image_Segmentation_CVPR_2019_paper.pdf)

1. **LT-Net: Label Transfer by Learning Reversible Voxel-Wise Correspondence for One-Shot Medical Image Segmentation,** in CVPR, 2020.
*S. Wang, S. Cao, D. Wei, R. Wang, K. Ma, L. Wang, D. Meng, and Y. Zheng.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_LT-Net_Label_Transfer_by_Learning_Reversible_Voxel-Wise_Correspondence_for_One-Shot_CVPR_2020_paper.pdf)

1. **Few-Shot Pill Recognition,** in CVPR, 2020.
*S. Ling, A. Pastor, J. Li, Z. Che, J. Wang, J. Kim, and P. L. Callet.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ling_Few-Shot_Pill_Recognition_CVPR_2020_paper.pdf)

1. **Self-Supervision With Superpixels: Training Few-Shot Medical Image Segmentation Without Annotation,** in ECCV, 2020.
*C. Ouyang, C. Biffi, C. Chen, T. Kart, H. Qiu, and D. Rueckert.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740749.pdf)
[code](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)

1. **Deep Complementary Joint Model for Complex Scene Registration and Few-Shot Segmentation on Medical Images,** in ECCV, 2020.
*Y. He, T. Li, G. Yang, Y. Kong, Y. Chen, H. Shu, J. Coatrieux, J. Dillenseger, and S. Li.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630749.pdf)

1. **Recurrent Mask Refinement for Few-Shot Medical Image Segmentation,** in ICCV, 2021.
*H. Tang, X. Liu, S. Sun, X. Yan, and X. Xie.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_Recurrent_Mask_Refinement_for_Few-Shot_Medical_Image_Segmentation_ICCV_2021_paper.pdf)
[code](https://github.com/uci-cbcl/RP-Net)

1. **Modeling the Probabilistic Distribution of Unlabeled Data for One-Shot Medical Image Segmentation,** in AAAI, 2021.
*Y. Ding, X. Yu, and Y. Yang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16212/16019)
[code](https://github.com/dyh127/Modeling-the-Probabilistic-Distribution-of-Unlabeled-Data)

1. **Which Images to Label for Few-Shot Medical Landmark Detection?,** in CVPR, 2022.
*Q. Quan, Q. Yao, J. Li, and S. K. Zhou.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Quan_Which_Images_To_Label_for_Few-Shot_Medical_Landmark_Detection_CVPR_2022_paper.pdf)

1. **MetaCare++: Meta-Learning With Hierarchical Subtyping for Cold-Start Diagnosis Prediction in Healthcare Data,** in SIGIR, 2022.
*Y. Tan, C. Yang, X. Wei, C. Chen, W. Liu, L. Li, and J. Z. a. X. Zheng.*
[paper](https://dl.acm.org/doi/10.1145/3477495.3532020)

1. **Dual Meta-Learning With Longitudinally Consistent Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan,** in ICCV, 2023.
*Y. Sun, F. Wang, J. Shu, H. Wang, L. Wang, D. Meng, and C. Lian.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Dual_Meta-Learning_with_Longitudinally_Consistent_Regularization_for_One-Shot_Brain_Tissue_ICCV_2023_paper.pdf)
[code](https://github.com/ladderlab-xjtu/DuMeta)

1. **The Rise of AI Language Pathologists: Exploring Two-Level Prompt Learning for Few-Shot Weakly-Supervised Whole Slide Image Classification,** in NeurIPS, 2023.
*L. Qu, x. Luo, K. Fu, M. Wang, and Z. Song.*
[paper](https://openreview.net/attachment?id=mSDfBXr8Py&name=pdf)
[code](https://github.com/miccaiif/TOP)

1. **Robust One-Shot Segmentation of Brain Tissues via Image-Aligned Style Transformation,** in AAAI, 2023.
*J. Lv, X. Zeng, S. Wang, R. Duan, Z. Wang, and Q. Li.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25276/25048)
[code](https://github.com/JinxLv/One-shot-segmentation-via-IST)

1. **Rethinking Few-Shot Medical Segmentation: A Vector Quantization View,** in CVPR, 2023.
*S. Huang, T. Xu, N. Shen, F. Mu, and J. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Few-Shot_Medical_Segmentation_A_Vector_Quantization_View_CVPR_2023_paper.pdf)

1. **Enhancing Small Medical Learners With Privacy-Preserving Contextual Prompting,** in ICLR, 2024.
*X. Zhang, S. Li, X. Yang, C. Tian, Y. Qin, and L. R. Petzol.*
[paper](https://openreview.net/attachment?id=ztpy1gsUpT&name=pdf)
[code](https://github.com/XZhang97666/PrivacyBoost-SLM)

1. **EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records,** in EMNLP, 2024.
*W. Shi, R. Xu, Y. Zhuang, Y. Yu, J. Zhang, H. Wu, Y. Zhu, J. C. Ho, C. Yang, and M. D. Wang.*
[paper](https://aclanthology.org/2024.emnlp-main.1245)




### Others

1. **SMASH: One-Shot Model Architecture Search Through Hypernetworks,** in ICLR, 2018.
*A. Brock, T. Lim, J. Ritchie, and N. Weston.*
[paper](https://openreview.net/forum?id=rydeCEhs-)

1. **SPARC: Self-Paced Network Representation for Few-Shot Rare Category Characterization,** in KDD, 2018.
*D. Zhou, J. He, H. Yang, and W. Fan.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3219968)

1. **Learning From Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction,** in WWW, 2019.
*H. Yao, Y. Liu, Y. Wei, X. Tang, and Z. Li.*
[paper](https://arxiv.org/pdf/1901.08518.pdf)
[code](https://github.com/huaxiuyao/MetaST)

1. **Federated Meta-Learning for Fraudulent Credit Card Detection,** in IJCAI, 2020.
*W. Zheng, L. Yan, C. Gou, and F. Wang.*
[paper](https://www.ijcai.org/Proceedings/2020/0642.pdf)

1. **Differentially Private Meta-Learning,** in ICLR, 2020.
*J. Li, M. Khodak, S. Caldas, and A. Talwalkar.*
[paper](https://openreview.net/pdf?id=rJgqMRVYvr)

1. **Towards Fast Adaptation of Neural Architectures With Meta Learning,** in ICLR, 2020.
*D. Lian, Y. Zheng, Y. Xu, Y. Lu, L. Lin, P. Zhao, J. Huang, and S. Gao.*
[paper](https://openreview.net/pdf?id=r1eowANFvr)

1. **Using Optimal Embeddings to Learn New Intents With Few Examples: An Application in the Insurance Domain,** in KDD, 2020.
*S. Acharya, and G. Fung.*
[paper](http://ceur-ws.org/Vol-2666/KDD_Converse20_paper_10.pdf)

1. **Few-Sample and Adversarial Representation Learning for Continual Stream Mining,** in WWW, 2020.
*Z. Wang, Y. Wang, Y. Lin, E. Delord, and L. Khan.*
[paper](https://dl.acm.org/doi/10.1145/3366423.3380153)

1. **Few-Shot Data-Driven Algorithms for Low Rank Approximation,** in NeurIPS, 2021.
*P. Indyk, T. Wagner, and D. Woodruff.*
[paper](https://proceedings.neurips.cc/paper/2021/file/588da7a73a2e919a23cb9a419c4c6d44-Paper.pdf)

1. **Non-Gaussian Gaussian Processes for Few-Shot Regression,** in NeurIPS, 2021.
*M. Sendera, J. Tabor, A. Nowak, A. Bedychaj, M. Patacchiola, T. Trzcinski, P. Spurek, and M. Zieba.*
[paper](https://proceedings.neurips.cc/paper/2021/file/54f3bc04830d762a3b56a789b6ff62df-Paper.pdf)

1. **HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning,** in NeurIPS, 2021.
*H. Lee, S. Lee, S. Chong, and S. J. Hwang.*
[paper](https://proceedings.neurips.cc/paper/2021/file/e3251075554389fe91d17a794861d47b-Paper.pdf)

1. **Learning to Learn Dense Gaussian Processes for Few-Shot Learning,** in NeurIPS, 2021.
*Z. Wang, Z. Miao, X. Zhen, and Q. Qiu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/6e2713a6efee97bacb63e52c54f0ada0-Paper.pdf)

1. **Curriculum Meta-Learning for Next POI Recommendation,** in KDD, 2021.
*Y. Chen, X. Wang, M. Fan, J. Huang, S. Yang, and W. Zhu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467132)
[code](https://github.com/PaddlePaddle/Research/tree/master/ST_DM/KDD2021-CHAML)

1. **MFNP: A Meta-Optimized Model for Few-Shot Next POI Recommendation,** in IJCAI, 2021.
*H. Sun, J. Xu, K. Zheng, P. Zhao, P. Chao, and X. Zhou.*
[paper](https://www.ijcai.org/proceedings/2021/0415.pdf)

1. **Taxonomy-Aware Learning for Few-Shot Event Detection,** in WWW, 2021.
*J. Zheng, F. Cai, W. Chen, W. Lei, and H. Chen.*
[paper](https://doi.org/10.1145/3442381.344994)

1. **Learning From Graph Propagation via Ordinal Distillation for One-Shot Automated Essay Scoring,** in WWW, 2021.
*Z. Jiang, M. Liu, Y. Yin, H. Yu, Z. Cheng, and Q. Gu.*
[paper](https://doi.org/10.1145/3442381.3450017)

1. **FL-MSRE: A Few-Shot Learning Based Approach to Multimodal Social Relation Extraction,** in AAAI, 2021.
*H. Wan, M. Zhang, J. Du, Z. Huang, Y. Yang, and J. Z. Pan.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17639/17446)
[code](https://github.com/sysulic/FL-MSRE)

1. **Progressive Network Grafting for Few-Shot Knowledge Distillation,** in AAAI, 2021.
*C. Shen, X. Wang, Y. Yin, J. Song, S. Luo, and M. Song.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16356/16163)
[code](https://github.com/zju-vipa/NetGraft)

1. **Physics-Aware Spatiotemporal Modules With Auxiliary Tasks for Meta-Learning,** in IJCAI, 2021.
*S. Seo, C. Meng, S. Rambhatla, and Y. Liu.*
[paper](https://www.ijcai.org/proceedings/2021/0405.pdf)

1. **Meta-Learning Dynamics Forecasting Using Task Inference,** in NeurIPS, 2022.
*R. Wang, R. Walters, and R. Yu.*
[paper](https://openreview.net/pdf?id=BsSP7pZGFQO)
[code](https://github.com/Rose-STL-Lab/Dynamic-Adaptation-Network)

1. **Rapid Model Architecture Adaption for Meta-Learning,** in NeurIPS, 2022.
*Y. Zhao, X. Gao, I. Shumailov, N. Fusi, and R. D. Mullins.*
[paper](https://openreview.net/pdf?id=Yq6g9xluV)

1. **A Meta-Learning Based Stress Category Detection Framework on Social Media,** in WWW, 2022.
*X. Wang, L. Cao, H. Zhang, L. Feng, Y. Ding, and N. Li.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512013)

1. **Few-Shot Learning for Trajectory-Based Mobile Game Cheating Detection,** in KDD, 2022.
*Y. Su, D. Yao, X. Chu, W. Li, J. Bi, S. Zhao, R. Wu, S. Zhang, J. Tao, and H. Deng.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539157)
[code](https://github.com/super1225/cheating-detection)

1. **Improving Few-Shot Text-to-SQL With Meta Self-Training via Column Specificity,** in IJCAI, 2022.
*X. Guo, Y. Chen, G. Qi, T. Wu, and H. Xu.*
[paper](https://www.ijcai.org/proceedings/2022/0576.pdf)
[code](https://github.com/ygxw0909/MST-SQL)

1. **Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting,** in ICLR, 2023.
*X. Jiang, R. Missel, Z. Li, and L. Wang.*
[paper](https://openreview.net/pdf?id=7C9aRX2nBf2)
[code](https://github.com/john-x-jiang/meta_ss)

1. **Transfer NAS With Meta-Learned Bayesian Surrogates,** in ICLR, 2023.
*G. Shala, T. Elsken, F. Hutter, and J. Grabocka.*
[paper](https://openreview.net/pdf?id=paGvsrl4Ntr)
[code](https://github.com/TNAS-DCS/TNAS-DCS)

1. **Few-Shot Domain Adaptation for End-to-End Communication,** in ICLR, 2023.
*J. Raghuram, Y. Zeng, D. Garcia, R. Ruiz, S. Jha, J. Widmer, and S. Banerjee.*
[paper](https://openreview.net/pdf?id=4F1gvduDeL)
[code](https://github.com/jayaram-r/domain-adaptation-autoencoder)

1. **Supervised Contrastive Few-Shot Learning for High-Frequency Time Series,** in AAAI, 2023.
*X. Chen, C. Ge, M. Wang, and J. Wang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25863/25635)

1. **Few-Shot Defect Image Generation via Defect-Aware Feature Manipulation,** in AAAI, 2023.
*Y. Duan, Y. Hong, L. Niu, and L. Zhang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25132/24904)
[code](https://github.com/Ldhlwh/DFMGAN)

1. **Multi-Label Few-Shot ICD Coding as Autoregressive Generation With Prompt,** in AAAI, 2023.
*Z. Yang, S. Kwon, Z. Yao, and H. Yu.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25668/25440)
[code](https://github.com/whaleloops/KEPT)

1. **Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning.,** in ICML, 2024.
*S. Han, J. Yoon, S. Ö. Arik, and T. Pfister.*
[paper](https://openreview.net/forum?id=fRG45xL1WT)

1. **D2R2: Diffusion-based Representation with Random Distance Matching for Tabular Few-shot Learning.,** in NeurIPS, 2024.
*R. Liu, L. Fang, W. Wang, and B. Jing.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/40eff1670d6b08bb1bda48b0c5f30110-Paper-Conference.pdf)

1. **Improved Few-Shot Jailbreaking Can Circumvent Aligned Language Models and Their Defenses.,** in NeurIPS, 2024.
*X. Zheng, T. Pang, C. Du, Q. Liu, J. Jiang, and M. Lin.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/39a3aa9dfd0280ff8fbad1d330662cac-Paper-Conference.pdf)

1. **TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series,** in ICLR, 2024.
*C. Sun, Y. Li, H. Li, and S. Hong.*
[paper](https://openreview.net/attachment?id=Tuh4nZVb0g&name=pdf)

1. **Modularized Networks for Few-shot Hateful Meme Detection,** in WWW, 2024.
*R. Cao, R. K.-W. Lee, and J. Jiang.*
[paper](https://doi.org/10.1145/3589334.3648145)

1. **Contrastive Fingerprinting: A Novel Website Fingerprinting Attack over Few-shot Traces,** in WWW, 2024.
*Y. Xie, J. Feng, W. Huang, Y. Zhang, X. Sun, X. Chen, and X. Luo.*
[paper](https://doi.org/10.1145/3589334.3645575)

1. **ProCom: A Few-shot Targeted Community Detection Algorithm,** in KDD, 2024.
*X. Wu, K. Xiong, Y. Xiong, X. He, Y. Zhang, Y. Jiao, and J. Zhang.*
[paper](https://doi.org/10.1145/3637528.3671749)

1. **Wearable Sensor-Based Few-Shot Continual Learning on Hand Gestures for Motor-Impaired Individuals via Latent Embedding Exploitation,** in IJCAI, 2024.
*R. B. Rafiq, W. Shi, and M. V. Albert.*
[paper](https://www.ijcai.org/proceedings/2024/823)

1. **Hot or Cold? Adaptive Temperature Sampling for Code Generation with Large Language Models,** in AAAI, 2024.
*Y. Zhu, J. Li, G. Li, Y. Zhao, J. Li, Z. Jin, and H. Mei.*
[paper](https://doi.org/10.1609/aaai.v38i1.27798)

1. **Count What You Want: Exemplar Identification and Few-Shot Counting of Human Actions in the Wild,** in AAAI, 2024.
*Y. Huang, D. D. Nguyen, L. Nguyen, C. Pham, and M. Hoai.*
[paper](https://doi.org/10.1609/aaai.v38i9.28869)

1. **Few-Shot Natural Language to First-Order Logic Translation via Code Generation,** in NAACL, 2025.
*and J. Liu.*
[paper](https://doi.org/10.18653/v1/2025.naacl-long.547)

1. **HeGTa: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding,** in AAAI, 2025.
*R. Jin, Y. Li, G. Qi, N. Hu, Y.-F. Li, J. Chen, J. Wang, Y. Chen, D. Min, and S. Bi.*
[paper](https://doi.org/10.1609/aaai.v39i23.34606)

1. **In-Context Unlearning: Language Models as Few-Shot Unlearners.,** in ICML, 2024.
*M. Pawelczyk, S. Neel, and H. Lakkaraju.*
[paper](https://openreview.net/forum?id=GKcwle8XC9)

1. **Manhattan Self-Attention Diffusion Residual Networks with Dynamic Bias Rectification for BCI-based Few-Shot Learning,** in AAAI, 2025.
*H. Wang, L. Xu, Y. Yu, W. Ding, and Y. Xu.*
[paper](https://doi.org/10.1609/aaai.v39i13.33580)

1. **Retrieving Versus Understanding Extractive Evidence in Few-Shot Learning,** in AAAI, 2025.
*K. Elbakian, and S. Carton.*
[paper](https://doi.org/10.1609/aaai.v39i26.34936)

## [Theories](#content)

1. **Learning to Learn Around a Common Mean,** in NeurIPS, 2018.
*G. Denevi, C. Ciliberto, D. Stamos, and M. Pontil.* 
[paper](https://papers.nips.cc/paper/8220-learning-to-learn-around-a-common-mean.pdf)

1. **Meta-Learning and Universality: Deep Representations and Gradient Descent Can Approximate Any Learning Algorithm,** in ICLR, 2018.
*C. Finn and S. Levine.*
[paper](https://openreview.net/forum?id=HyjC5yWCW)

1. **A Theoretical Analysis of the Number of Shots in Few-Shot Learning,** in ICLR, 2020.
*T. Cao, M. T. Law, and S. Fidler.*
[paper](https://openreview.net/pdf?id=HkgB2TNYPS)

1. **Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML,** in ICLR, 2020.
*A. Raghu, M. Raghu, S. Bengio, and O. Vinyals.*
[paper](https://openreview.net/pdf?id=rkgMkCEtPB)

1. **Robust Meta-Learning for Mixed Linear Regression With Small Batches,** in NeurIPS, 2020.
*W. Kong, R. Somani, S. Kakade, and S. Oh.*
[paper](https://proceedings.neurips.cc/paper/2020/file/3214a6d842cc69597f9edf26df552e43-Paper.pdf)

1. **One-Shot Distributed Ridge Regression in High Dimensions,** in ICML, 2020.
*Y. Sheng, and E. Dobriban.*
[paper](http://proceedings.mlr.press/v119/sheng20a/sheng20a.pdf)

1. **Bridging the Gap Between Practice and PAC-Bayes Theory in Few-Shot Meta-Learning,** in NeurIPS, 2021.
*N. Ding, X. Chen, T. Levinboim, S. Goodman, and R. Soricut.*
[paper](https://proceedings.neurips.cc/paper/2021/file/f6b6d2a114a9644419dc8d2315f22401-Paper.pdf)

1. **Generalization Bounds for Meta-Learning: An Information-Theoretic Analysis,** in NeurIPS, 2021.
*Q. CHEN, C. Shui, and M. Marchand.*
[paper](https://proceedings.neurips.cc/paper/2021/file/d9d347f57ae11f34235b4555710547d8-Paper.pdf)

1. **Generalization Bounds for Meta-Learning via PAC-Bayes and Uniform Stability,** in NeurIPS, 2021.
*A. Farid, and A. Majumdar.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf)

1. **Unraveling Model-Agnostic Meta-Learning via the Adaptation Learning Rate,** in ICLR, 2022.
*Y. Zou, F. Liu, and Q. Li.*
[paper](https://openreview.net/pdf?id=3rULBvOJ8D2)

1. **On the Importance of Firth Bias Reduction in Few-Shot Classification,** in ICLR, 2022.
*S. Ghaffari, E. Saleh, D. Forsyth, and Y. Wang.*
[paper](https://openreview.net/pdf?id=DNRADop4ksB)
[code](https://github.com/ehsansaleh/firth_bias_reduction)

1. **Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning,** in CVPR, 2022.
*H. Wang, Y. Wang, R. Sun, and B. Li.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Global_Convergence_of_MAML_and_Theory-Inspired_Neural_Architecture_Search_for_CVPR_2022_paper.pdf)

1. **Smoothed Embeddings for Certified Few-Shot Learning,** in NeurIPS, 2022.
*M. Pautov, O. Kuznetsova, N. Tursynbek, A. Petiushko, and I. Oseledets.*
[paper](https://openreview.net/pdf?id=m2JJO3iEe_5)
[code](https://github.com/koava36/certrob-fewshot)

1. **Towards Few-Shot Adaptation of Foundation Models via Multitask Finetuning,** in ICLR, 2024.
*Z. Xu, Z. Shi, J. Wei, F. Mu, Y. Li, and Y. Liang.*
[paper](https://openreview.net/attachment?id=1jbh2e0b2K&name=pdf)
[code](https://github.com/OliverXUZY/Foudation-Model_Multitask)


## [Few-shot Learning and Zero-shot Learning](#content)

1. **Label-Embedding for Attribute-Based Classification,** in CVPR, 2013.
*Z. Akata, F. Perronnin, Z. Harchaoui, and C. Schmid.*
[paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Akata_Label-Embedding_for_Attribute-Based_2013_CVPR_paper.pdf)

1. **A Unified Semantic Embedding: Relating Taxonomies and Attributes,** in NeurIPS, 2014.
*S. J. Hwang and L. Sigal.*
[paper](https://papers.nips.cc/paper/5289-a-unified-semantic-embedding-relating-taxonomies-and-attributes.pdf)

1. **Multi-Attention Network for One Shot Learning,** in CVPR, 2017.
*P. Wang, L. Liu, C. Shen, Z. Huang, A. van den Hengel, and H. T. Shen.*
[paper](http://zpascal.net/cvpr2017/Wang_Multi-Attention_Network_for_CVPR_2017_paper.pdf)

1. **Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces,** in EMNLP, 2018.
*A. Rios and R. Kavuluru.*
[paper](https://www.aclweb.org/anthology/D18-1352.pdf)

1. **Learning Compositional Representations for Few-Shot Recognition,** in ICCV, 2019.
*P. Tokmakov, Y.-X. Wang, and M. Hebert.*
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tokmakov_Learning_Compositional_Representations_for_Few-Shot_Recognition_ICCV_2019_paper.pdf)
[code](https://sites.google.com/view/comprepr/home)

1. **Large-Scale Few-Shot Learning: Knowledge Transfer With Class Hierarchy,** in CVPR, 2019.
*A. Li, T. Luo, Z. Lu, T. Xiang, and L. Wang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Large-Scale_Few-Shot_Learning_Knowledge_Transfer_With_Class_Hierarchy_CVPR_2019_paper.pdf)

1. **Generalized Zero- And Few-Shot Learning via Aligned Variational Autoencoders,** in CVPR, 2019.
*E. Schonfeld, S. Ebrahimi, S. Sinha, T. Darrell, and Z. Akata.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schonfeld_Generalized_Zero-_and_Few-Shot_Learning_via_Aligned_Variational_Autoencoders_CVPR_2019_paper.pdf)
[code](https://github.com/edgarschnfld/CADA-VAE-PyTorch)

1. **F-Vaegan-D2: A Feature Generating Framework for Any-Shot Learning,** in CVPR, 2019.
*Y. Xian, S. Sharma, B. Schiele, and Z. Akata.*
[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xian_F-VAEGAN-D2_A_Feature_Generating_Framework_for_Any-Shot_Learning_CVPR_2019_paper.pdf)

1. **TGG: Transferable Graph Generation for Zero-Shot and Few-Shot Learning,** in ACM MM, 2019.
*C. Zhang, X. Lyu, and Z. Tang.*
[paper](https://dl.acm.org/doi/abs/10.1145/3343031.3351000)

1. **Adaptive Cross-Modal Few-Shot Learning,** in NeurIPS, 2019.
*C. Xing, N. Rostamzadeh, B. N. Oreshkin, and P. O. Pinheiro.*
[paper](https://papers.nips.cc/paper/8731-adaptive-cross-modal-few-shot-learning.pdf)

1. **Learning Meta Model for Zero- And Few-Shot Face Anti-Spoofing,** in AAAI, 2020.
*Y. Qin, C. Zhao, X. Zhu, Z. Wang, Z. Yu, T. Fu, F. Zhou, J. Shi, and Z. Lei.*
[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6866)

1. **RD-GAN: Few/Zero-Shot Chinese Character Style Transfer via Radical Decomposition and Rendering,** in ECCV, 2020.
*Y. Huang, M. He, L. Jin, and Y. Wang.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510154.pdf)

1. **An Empirical Study on Large-Scale Multi-Label Text Classification Including Few and Zero-Shot Labels,** in EMNLP, 2020.
*I. Chalkidis, M. Fergadiotis, S. Kotitsas, P. Malakasiotis, N. Aletras, and I. Androutsopoulos.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.607.pdf)

1. **Multi-Label Few/Zero-Shot Learning With Knowledge Aggregated From Multiple Label Graphs,** in EMNLP, 2020.
*J. Lu, L. Du, M. Liu, and J. Dipnall.*
[paper](https://www.aclweb.org/anthology/2020.emnlp-main.235.pdf)

1. **Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design,** in NeurIPS, 2020.
*M. Dennis, N. Jaques, E. Vinitsky, A. Bayen, S. Russell, A. Critch, and S. Levine.*
[paper](https://proceedings.neurips.cc/paper/2020/file/985e9a46e10005356bbaf194249f6856-Paper.pdf)

1. **Learning Graphs for Knowledge Transfer With Limited Labels,** in CVPR, 2021.
*P. Ghosh, N. Saini, L. S. Davis, and A. Shrivastava.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghosh_Learning_Graphs_for_Knowledge_Transfer_With_Limited_Labels_CVPR_2021_paper.pdf)

1. **Improving Zero and Few-Shot Abstractive Summarization With Intermediate Fine-Tuning and Data Augmentation,** in NAACL-HLT, 2021.
*A. R. Fabbri, S. Han, H. Li, H. Li, M. Ghazvininejad, S. R. Joty, D. R. Radev, and Y. Mehdad.*
[paper](https://aclanthology.org/2021.naacl-main.57.pdf)

1. **SEQZERO: Few-Shot Compositional Semantic Parsing With Sequential Prompts and Zero-Shot Models,** in NAACL, 2022.
*J. Yang, H. Jiang, Q. Yin, D. Zhang, B. Yin, and D. Yang.*
[paper](https://aclanthology.org/2022.findings-naacl.5.pdf)
[code](https://github.com/amzn/SeqZero)

1. **Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction,** in EMNLP, 2021.
*O. Sainz, O. L. d. Lacalle, G. Labaka, A. Barrena, and E. Agirre.*
[paper](https://aclanthology.org/2021.emnlp-main.92.pdf)
[code](https://github.com/osainz59/Ask2Transformers)

1. **An Empirical Investigation of Word Alignment Supervision for Zero-Shot Multilingual Neural Machine Translation,** in EMNLP, 2021.
*A. Raganato, R. Vázquez, M. Creutz, and J. Tiedemann.*
[paper](https://aclanthology.org/2021.emnlp-main.664.pdf)

1. **Bridge to Target Domain by Prototypical Contrastive Learning and Label Confusion: Re-Explore Zero-Shot Learning for Slot Filling,** in EMNLP, 2021.
*L. Wang, X. Li, J. Liu, K. He, Y. Yan, and W. Xu.*
[paper](https://aclanthology.org/2021.emnlp-main.746.pdf)
[code](https://github.com/w-lw/pclc)

1. **A Label-Aware BERT Attention Network for Zero-Shot Multi-Intent Detection in Spoken Language Understanding,** in EMNLP, 2021.
*T. Wu, R. Su, and B. Juang.*
[paper](https://aclanthology.org/2021.emnlp-main.399.pdf)

1. **Zero-Shot Dialogue Disentanglement by Self-Supervised Entangled Response Selection,** in EMNLP, 2021.
*T. Chi, and A. I. Rudnicky.*
[paper](https://aclanthology.org/2021.emnlp-main.400.pdf)
[code](https://github.com/chijames/zero_shot_dialogue_disentanglement)

1. **Robust Retrieval Augmented Generation for Zero-Shot Slot Filling,** in EMNLP, 2021.
*M. R. Glass, G. Rossiello, M. F. M. Chowdhury, and A. Gliozzo.*
[paper](https://aclanthology.org/2021.emnlp-main.148.pdf)
[code](https://github.com/IBM/kgi-slot-filling)

1. **Everything Is All It Takes: A Multipronged Strategy for Zero-Shot Cross-Lingual Information Extraction,** in EMNLP, 2021.
*M. Yarmohammadi, S. Wu, M. Marone, H. Xu, S. Ebner, G. Qin, Y. Chen, J. Guo, C. Harman, K. Murray, A. S. White, M. Dredze, and B. V. Durme.*
[paper](https://aclanthology.org/2021.emnlp-main.149.pdf)
[code](https://github.com/shijie-wu/crosslingual-nlp)

1. **An Empirical Study on Multiple Information Sources for Zero-Shot Fine-Grained Entity Typing,** in EMNLP, 2021.
*Y. Chen, H. Jiang, L. Liu, S. Shi, C. Fan, M. Yang, and R. Xu.*
[paper](https://aclanthology.org/2021.emnlp-main.210.pdf)

1. **Zero-Shot Dialogue State Tracking via Cross-Task Transfer,** in EMNLP, 2021.
*Z. Lin, B. Liu, A. Madotto, S. Moon, Z. Zhou, P. Crook, Z. Wang, Z. Yu, E. Cho, R. Subba, and P. Fung.*
[paper](https://aclanthology.org/2021.emnlp-main.622.pdf)
[code](https://github.com/facebookresearch/Zero-Shot-DST)

1. **Finetuned Language Models Are Zero-Shot Learners,** in ICLR, 2022.
*J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le.*
[paper](https://openreview.net/pdf?id=gEZrGCozdqR)
[code](https://github.com/google-research/flan)

1. **Zero-Shot Stance Detection via Contrastive Learning,** in WWW, 2022.
*B. Liang, Z. Chen, L. Gui, Y. He, M. Yang, and R. Xu.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511994)
[code](https://github.com/HITSZ-HLT/PT-HCL)

1. **Reframing Instructional Prompts to GPTk's Language,** in Findings of ACL, 2022.
*D. Khashabi, C. Baral, Y. Choi, and H. Hajishirzi.*
[paper](https://aclanthology.org/2022.findings-acl.50.pdf)

1. **JointCL: A Joint Contrastive Learning Framework for Zero-Shot Stance Detection,** in ACL, 2022.
*B. Liang, Q. Zhu, X. Li, M. Yang, L. Gui, Y. He, and R. Xu.*
[paper](https://aclanthology.org/2022.acl-long.7.pdf)
[code](https://github.com/hitsz-hlt/jointcl)

1. **Knowledgeable Prompt-Tuning: Incorporating Knowledge Into Prompt Verbalizer for Text Classification,** in ACL, 2022.
*S. Hu, N. Ding, H. Wang, Z. Liu, J. Wang, J. Li, W. Wu, and M. Sun.*
[paper](https://aclanthology.org/2022.acl-long.158.pdf)
[code](https://github.com/thunlp/knowledgeableprompttuning)

1. **Uni-Perceiver: Pre-Training Unified Architecture for Generic Perception for Zero-Shot and Few-Shot Tasks,** in CVPR, 2022.
*X. Zhu, J. Zhu, H. Li, X. Wu, H. Li, X. Wang, and J. Dai.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.pdf)

1. **Enhancing Zero-Shot Stance Detection via Targeted Background Knowledge,** in SIGIR, 2022.
*Q. Zhu, B. Liang, J. Sun, J. Du, and L. Z. a. X. Ruifeng.*
[paper](https://dl.acm.org/doi/10.1145/3477495.3531807)

1. **Textual Entailment for Event Argument Extraction: Zero- And Few-Shot With Multi-Source Learning,** in NAACL, 2022.
*O. Sainz, I. Gonzalez-Dios, O. L. d. Lacalle, B. Min, and E. Agirre.*
[paper](https://aclanthology.org/2022.findings-naacl.187.pdf)
[code](https://github.com/osainz59/Ask2Transformers)

1. **Extreme Zero-Shot Learning for Extreme Text Classification,** in NAACL, 2022.
*Y. Xiong, W.-C. Chang, C.-J. Hsieh, H.-F. Yu, and I. S. Dhillon.*
[paper](https://aclanthology.org/2022.naacl-main.399.pdf)
[code](https://github.com/amzn/pecos/tree/mainline/examples/MACLR)

1. **Domain-Oriented Prefix-Tuning: Towards Efficient and Generalizable Fine-Tuning for Zero-Shot Dialogue Summarization,** in NAACL, 2022.
*L. Zhao, F. Zheng, W. Zeng, K. He, W. Xu, H. Jiang, W. Wu, and Y. Wu.*
[paper](https://aclanthology.org/2022.naacl-main.357.pdf)
[code](https://github.com/Zeng-WH/DOP-Tuning)

1. **Nearest Neighbor Zero-Shot Inference,** in EMNLP, 2022.
*W. Shi, J. Michael, S. Gururangan, and L. Zettlemoyer.*
[paper](https://aclanthology.org/2022.emnlp-main.214.pdf)
[code](https://github.com/swj0419/kNN_prompt)

1. **Continued Pretraining for Better Zero- And Few-Shot Promptability,** in EMNLP, 2022.
*Z. Wu, R. L. L. IV, P. Walsh, A. Bhagia, D. Groeneveld, S. Singh, and I. Beltagy.*
[paper](https://aclanthology.org/2022.emnlp-main.300.pdf)
[code](https://github.com/allenai/better-promptability)

1. **InstructDial: Improving Zero and Few-Shot Generalization in Dialogue Through Instruction Tuning,** in EMNLP, 2022.
*P. Gupta, C. Jiao, Y.-T. Yeh, S. Mehri, M. Eskenazi, and J. P. Bigham.*
[paper](https://aclanthology.org/2022.emnlp-main.33.pdf)
[code](https://github.com/prakharguptaz/Instructdial)

1. **Prompt-and-Rerank: A Method for Zero-Shot and Few-Shot Arbitrary Textual Style Transfer With Small Language Models,** in EMNLP, 2022.
*M. Suzgun, L. Melas-Kyriazi, and D. Jurafsky.*
[paper](https://aclanthology.org/2022.emnlp-main.141.pdf)
[code](https://github.com/suzgunmirac/prompt-and-rerank)

1. **Learning Instructions With Unlabeled Data for Zero-Shot Cross-Task Generalization,** in EMNLP, 2022.
*Y. Gu, P. Ke, X. Zhu, and M. Huang.*
[paper](https://aclanthology.org/2022.emnlp-main.105.pdf)
[code](https://github.com/thu-coai/UDIT)

1. **Zero-Shot Cross-Lingual Transfer of Prompt-Based Tuning With a Unified Multilingual Prompt,** in EMNLP, 2022.
*L. Huang, S. Ma, D. Zhang, F. Wei, and H. Wang.*
[paper](https://aclanthology.org/2022.emnlp-main.790.pdf)
[code](https://github.com/mojave-pku/UniPrompt)

1. **Finetune Like You Pretrain: Improved Finetuning of Zero-Shot Vision Models,** in CVPR, 2023.
*S. Goyal, A. Kumar, S. Garg, Z. Kolter, and A. Raghunathan.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Goyal_Finetune_Like_You_Pretrain_Improved_Finetuning_of_Zero-Shot_Vision_Models_CVPR_2023_paper.pdf)
[code](https://github.com/locuslab/FLYP)

1. **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation,** in CVPR, 2023.
*J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)

1. **SemSup-XC: Semantic Supervision for Zero and Few-Shot Extreme Classification,** in ICML, 2023.
*P. Aggarwal, A. Deshpande, and K. R. Narasimhan.*
[paper](https://proceedings.mlr.press/v202/aggarwal23a/aggarwal23a.pdf)
[code](https://github.com/princeton-nlp/semsup-xc)

1. **Zero- And Few-Shot Event Detection via Prompt-Based Meta Learning,** in ACL, 2023.
*Z. Yue, H. Zeng, M. Lan, H. Ji, and D. Wang.*
[paper](https://aclanthology.org/2023.acl-long.440.pdf)
[code](https://github.com/Yueeeeeeee/MetaEvent)

1. **HINT: Hypernetwork Instruction Tuning for Efficient Zero- And Few-Shot Generalisation,** in ACL, 2023.
*H. Ivison, A. Bhagia, Y. Wang, H. Hajishirzi, and M. E. Peters.*
[paper](https://aclanthology.org/2023.acl-long.631.pdf)
[code](https://github.com/allenai/hyper-task-descriptions)

1. **What Does the Failure to Reason With "Respectively" in Zero/Few-Shot Settings Tell Us About Language Models? Acl 2023,** in ACL, 2023.
*R. Cui, S. Lee, D. Hershcovich, and A. Søgaard.*
[paper](https://aclanthology.org/2023.acl-long.489.pdf)
[code](https://github.com/ruixiangcui/WikiResNLI_NatResNLI)

1. **Pre-Training Intent-Aware Encoders for Zero- And Few-Shot Intent Classification,** in EMNLP, 2023.
*M. Sung, J. Gung, E. Mansimov, N. Pappas, R. Shu, S. Romeo, Y. Zhang, and V. Castelli.*
[paper](https://aclanthology.org/2023.emnlp-main.646.pdf)

1. **ZGUL: Zero-Shot Generalization to Unseen Languages Using Multi-Source Ensembling of Language Adapters,** in EMNLP, 2023.
*V. Rathore, R. Dhingra, P. Singla, and Mausam.*
[paper](https://aclanthology.org/2023.emnlp-main.431.pdf)
[code](https://github.com/dair-iitd/ZGUL)

1. **Adaptive End-to-End Metric Learning for Zero-Shot Cross-Domain Slot Filling,** in EMNLP, 2023.
*Y. Shi, L. Wu, and M. Shao.*
[paper](https://aclanthology.org/2023.emnlp-main.387.pdf)
[code](https://github.com/Switchsyj/AdaE2ML-XSF)

1. **Empirical Study of Zero-Shot NER With ChatGPT,** in EMNLP, 2023.
*T. Xie, Q. Li, J. Zhang, Y. Zhang, Z. Liu, and H. Wang.*
[paper](https://aclanthology.org/2023.emnlp-main.493.pdf)
[code](https://github.com/Emma1066/Zero-Shot-NER-with-ChatGPT)

1. **Learning to Describe for Predicting Zero-Shot Drug-Drug Interactions,** in EMNLP, 2023.
*F. Zhu, Y. Zhang, L. Chen, B. Qin, and R. Xu.*
[paper](https://aclanthology.org/2023.emnlp-main.918.pdf)
[code](https://github.com/zhufq00/DDIs-Prediction)

1. **The Benefits of Label-Description Training for Zero-Shot Text Classification,** in EMNLP, 2023.
*L. Gao, D. Ghosh, and K. Gimpel.*
[paper](https://aclanthology.org/2023.emnlp-main.853.pdf)
[code](https://github.com/lingyugao/LabelDescTraining)

1. **Gen-Z: Generative Zero-Shot Text Classification With Contextualized Label Descriptions,** in ICLR, 2024.
*S. Kumar, C. Y. Park, and Y. Tsvetkov.*
[paper](https://openreview.net/attachment?id=rkplYfqUr0&name=pdf)

1. **Evaluating the Zero-Shot Robustness of Instruction-Tuned Language Models,** in ICLR, 2024.
*J. Sun, C. Shaib, and B. C. Wallace.*
[paper](https://openreview.net/attachment?id=g9diuvxN6D&name=pdf)

1. **Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis,** in ICLR, 2024.
*Z. Jiang, J. Liu, Y. Ren, J. He, Z. Ye, S. Ji, Q. Yang, C. Zhang, P. Wei, C. Wang, X. Yin, Z. MA, and Z. Zhao.*
[paper](https://openreview.net/attachment?id=mvMI3N4AvD&name=pdf)

1. **Zero and Few-Shot Semantic Parsing With Ambiguous Inputs,** in ICLR, 2024.
*E. Stengel-Eskin, K. Rawlins, and B. V. Durme.*
[paper](https://openreview.net/attachment?id=qL9gogRepu&name=pdf)

1. **Uni3D: Exploring Unified 3D Representation at Scale,** in ICLR, 2024.
*J. Zhou, J. Wang, B. Ma, Y.-S. Liu, T. Huang, and X. Wang.*
[paper](https://openreview.net/attachment?id=wcaE4Dfgt8&name=pdf)

1. **Transductive Zero-Shot and Few-Shot CLIP,** in CVPR, 2024.
*S. Martin, Y. Huang, F. Shakeri, J.-C. Pesquet, and I. B. Ayed.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02722)

1. **Liberating Seen Classes: Boosting Few-Shot and Zero-Shot Text Classification via Anchor Generation and Classification Reframing,** in AAAI, 2024.
*H. Liu, S. Zhao, X. Zhang, F. Zhang, W. Wang, F. Ma, H. Chen, H. Yu, and X. Zhang.*
[paper](https://doi.org/10.1609/aaai.v38i17.29827)

1. **Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series.,** in NeurIPS, 2024.
*V. Ekambaram, A. Jati, P. Dayama, S. Mukherjee, N. Nguyen, W. M. Gifford, C. Reddy, and J. Kalagnanam.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf)

1. **Manifold Induced Biases for Zero-shot and Few-shot Detection of Generated Images.,** in ICLR, 2025.
*J. Brokman, A. Giloni, O. Hofman, R. Vainshtein, H. Kojima, and G. Gilboa.*
[paper](https://openreview.net/forum?id=7gGl6HB5Zd)

## [Variants of Few-shot Learning](#content)

1. **Incremental Few-Shot Learning for Pedestrian Attribute Recognition,** in EMNLP, 2018.
*L. Xiang, X. Jin, G. Ding, J. Han, and L. Li.*
[paper](https://www.ijcai.org/Proceedings/2019/0543.pdf)

1. **Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments,** in ICLR, 2018.
*M. Al-Shedivat, T. Bansal, Y. Burda, I. Sutskever, I. Mordatch, and P. Abbeel.*
[paper](https://openreview.net/forum?id=Sk2u1g-0-)

1. **Deep Online Learning via Meta-Learning: Continual Adaptation for Model-Based RL,** in ICLR, 2018.
*A. Nagabandi, C. Finn, and S. Levine.*
[paper](https://openreview.net/references/pdf?id=ryuIpa6S4)

1. **Incremental Few-Shot Learning With Attention Attractor Networks,** in NeurIPS, 2019.
*M. Ren, R. Liao, E. Fetaya, and R. S. Zemel.*
[paper](https://papers.nips.cc/paper/8769-incremental-few-shot-learning-with-attention-attractor-networks.pdf)
[code](https://github.com/renmengye/inc-few-shot-attractor-public)

1. **Bidirectional One-Shot Unsupervised Domain Mapping,** in ICCV, 2019.
*T. Cohen, and L. Wolf.*
[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cohen_Bidirectional_One-Shot_Unsupervised_Domain_Mapping_ICCV_2019_paper.pdf)

1. **XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning,** in ICML, 2020.
*S. W. Yoon, D. Kim, J. Seo, and J. Moon.*
[paper](http://proceedings.mlr.press/v119/yoon20b/yoon20b.pdf)
[code](https://github.com/EdwinKim3069/XtarNet)

1. **Few-Shot Class-Incremental Learning,** in CVPR, 2020.
*X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tao_Few-Shot_Class-Incremental_Learning_CVPR_2020_paper.pdf)

1. **Wandering Within a World: Online Contextualized Few-Shot Learning,** in ICLR, 2021.
*M. Ren, M. L. Iuzzolino, M. C. Mozer, and R. Zemel.*
[paper](https://openreview.net/pdf?id=oZIvHV04XgC)

1. **Repurposing Pretrained Models for Robust Out-of-Domain Few-Shot Learning,** in ICLR, 2021.
*N. Kwon, H. Na, G. Huang, and S. Lacoste-Julien.*
[paper](https://openreview.net/pdf?id=qkLMTphG5-h)
[code](https://anonymous.4open.science/r/08ef52cf-456a-4e36-a408-04e1ad0bc5a9/)

1. **Incremental Few-Shot Instance Segmentation,** in CVPR, 2021.
*D. A. Ganea, B. Boom, and R. Poppe.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ganea_Incremental_Few-Shot_Instance_Segmentation_CVPR_2021_paper.pdf)
[code](https://github.com/danganea/iMTFA)

1. **Prototypical Cross-Domain Self-Supervised Learning for Few-Shot Unsupervised Domain Adaptation,** in CVPR, 2021.
*X. Yue, Z. Zheng, S. Zhang, Y. Gao, T. Darrell, K. Keutzer, and A. S. Vincentelli.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Prototypical_Cross-Domain_Self-Supervised_Learning_for_Few-Shot_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)

1. **Self-Promoted Prototype Refinement for Few-Shot Class-Incremental Learning,** in CVPR, 2021.
*K. Zhu, Y. Cao, W. Zhai, J. Cheng, and Z. Zha.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf)

1. **Semantic-Aware Knowledge Distillation for Few-Shot Class-Incremental Learning,** in CVPR, 2021.
*A. Cheraghian, S. Rahman, P. Fang, S. K. Roy, L. Petersson, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheraghian_Semantic-Aware_Knowledge_Distillation_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf)

1. **Few-Shot Incremental Learning With Continually Evolved Classifiers,** in CVPR, 2021.
*C. Zhang, N. Song, G. Lin, Y. Zheng, P. Pan, and Y. Xu.*
[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf)

1. **Learning a Universal Template for Few-Shot Dataset Generalization,** in ICML, 2021.
*E. Triantafillou, H. Larochelle, R. Zemel, and V. Dumoulin.*
[paper](http://proceedings.mlr.press/v139/triantafillou21a/triantafillou21a.pdf)

1. **GP-Tree: A Gaussian Process Classifier for Few-Shot Incremental Learning,** in ICML, 2021.
*I. Achituve, A. Navon, Y. Yemini, G. Chechik, and E. Fetaya.*
[paper](http://proceedings.mlr.press/v139/achituve21a/achituve21a.pdf)
[code](https://github.com/IdanAchituve/GP-Tree)

1. **Addressing Catastrophic Forgetting in Few-Shot Problems,** in ICML, 2021.
*P. Yap, H. Ritter, and D. Barber.*
[paper](http://proceedings.mlr.press/v139/yap21a/yap21a.pdf)
[code](https://github.com/pauchingyap/boml)

1. **Few-Shot Conformal Prediction With Auxiliary Tasks,** in ICML, 2021.
*A. Fisch, T. Schuster, T. Jaakkola, and R. Barzilay.*
[paper](http://proceedings.mlr.press/v139/fisch21a/fisch21a.pdf)
[code](https://github.com/ajfisch/few-shot-cp)

1. **Few-Shot Lifelong Learning,** in AAAI, 2021.
*P. Mazumder, P. Singh, and P. Rai.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16334/16141)

1. **Few-Shot Class-Incremental Learning via Relation Knowledge Distillation,** in AAAI, 2021.
*S. Dong, X. Hong, X. Tao, X. Chang, X. Wei, and Y. Gong.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16213/16020)

1. **Few-Shot One-Class Classification via Meta-Learning,** in AAAI, 2021.
*A. Frikha, D. Krompass, H. Koepken, and V. Tresp.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16913/16720)
[code](https://github.com/AhmedFrikha/Few-Shot-One-Class-Classification-via-Meta-Learning)

1. **Practical One-Shot Federated Learning for Cross-Silo Setting,** in IJCAI, 2021.
*Q. Li, B. He, and D. Song.*
[paper](https://www.ijcai.org/proceedings/2021/0205.pdf)
[code](https://github.com/QinbinLi/FedK)

1. **Incremental Few-Shot Text Classification With Multi-Round New Classes: Formulation, Dataset and System,** in NAACL-HLT, 2021.
*C. Xia, W. Yin, Y. Feng, and P. S. Yu.*
[paper](https://aclanthology.org/2021.naacl-main.106.pdf)

1. **Continual Few-Shot Learning for Text Classification,** in EMNLP, 2021.
*R. Pasunuru, V. Stoyanov, and M. Bansal.*
[paper](https://aclanthology.org/2021.emnlp-main.460.pdf)
[code](https://github.com/ramakanth-pasunuru/cfl-benchmark)

1. **Self-Training With Few-Shot Rationalization,** in EMNLP, 2021.
*M. M. Bhat, A. Sordoni, and S. Mukherjee.*
[paper](https://aclanthology.org/2021.emnlp-main.836.pdf)

1. **Diverse Distributions of Self-Supervised Tasks for Meta-Learning in NLP,** in EMNLP, 2021.
*T. Bansal, K. P. Gunasekaran, T. Wang, T. Munkhdalai, and A. McCallum.*
[paper](https://aclanthology.org/2021.emnlp-main.469.pdf)

1. **Generalized and Incremental Few-Shot Learning by Explicit Learning and Calibration Without Forgetting,** in ICCV, 2021.
*A. Kukleva, H. Kuehne, and B. Schiele.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kukleva_Generalized_and_Incremental_Few-Shot_Learning_by_Explicit_Learning_and_Calibration_ICCV_2021_paper.pdf)

1. **Meta Learning on a Sequence of Imbalanced Domains With Difficulty Awareness,** in ICCV, 2021.
*Z. Wang, T. Duan, L. Fang, Q. Suo, and M. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Meta_Learning_on_a_Sequence_of_Imbalanced_Domains_With_Difficulty_ICCV_2021_paper.pdf)
[code](https://github.com/joey-wang123/imbalancemeta)

1. **Synthesized Feature Based Few-Shot Class-Incremental Learning on a Mixture of Subspaces,** in ICCV, 2021.
*A. Cheraghian, S. Rahman, S. Ramasinghe, P. Fang, C. Simon, L. Petersson, and M. Harandi.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cheraghian_Synthesized_Feature_Based_Few-Shot_Class-Incremental_Learning_on_a_Mixture_of_ICCV_2021_paper.pdf)

1. **Few-Shot and Continual Learning With Attentive Independent Mechanisms,** in ICCV, 2021.
*E. Lee, C. Huang, and C. Lee.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Few-Shot_and_Continual_Learning_With_Attentive_Independent_Mechanisms_ICCV_2021_paper.pdf)
[code](https://github.com/huang50213/AIM-Fewshot-Continual)

1. **Low-Shot Validation: Active Importance Sampling for Estimating Classifier Performance on Rare Categories,** in ICCV, 2021.
*F. Poms, V. Sarukkai, R. T. Mullapudi, N. S. Sohoni, W. R. Mark, D. Ramanan, and K. Fatahalian.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Poms_Low-Shot_Validation_Active_Importance_Sampling_for_Estimating_Classifier_Performance_on_ICCV_2021_paper.pdf)

1. **Overcoming Catastrophic Forgetting in Incremental Few-Shot Learning by Finding Flat Minima,** in NeurIPS, 2021.
*G. SHI, J. CHEN, W. Zhang, L. Zhan, and X. Wu.*
[paper](https://proceedings.neurips.cc/paper/2021/file/357cfba15668cc2e1e73111e09d54383-Paper.pdf)

1. **Variational Continual Bayesian Meta-Learning,** in NeurIPS, 2021.
*Q. Zhang, J. Fang, Z. Meng, S. Liang, and E. Yilmaz.*
[paper](https://proceedings.neurips.cc/paper/2021/file/cdd0500dc0ef6682fa6ec6d2e6b577c4-Paper.pdf)

1. **LFPT5: A Unified Framework for Lifelong Few-Shot Language Learning Based on Prompt Tuning of T5,** in ICLR, 2022.
*C. Qin, and S. Joty.*
[paper](https://openreview.net/pdf?id=HCRVf71PMF)
[code](https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning)

1. **Subspace Regularizers for Few-Shot Class Incremental Learning,** in ICLR, 2022.
*A. F. Akyürek, E. Akyürek, D. Wijaya, and J. Andreas.*
[paper](https://openreview.net/pdf?id=boJy41J-tnQ)
[code](https://github.com/feyzaakyurek/subspace-reg)

1. **Meta Discovery: Learning to Discover Novel Classes Given Very Limited Data,** in ICLR, 2022.
*H. Chi, F. Liu, W. Yang, L. Lan, T. Liu, B. Han, G. Niu, M. Zhou, and M. Sugiyama.*
[paper](https://openreview.net/pdf?id=MEpKGLsY8f)

1. **Topological Transduction for Hybrid Few-Shot Learning,** in WWW, 2022.
*J. Chen, and A. Zhang.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512033)

1. **Continual Few-Shot Relation Learning via Embedding Space Regularization and Data Augmentation,** in ACL, 2022.
*C. Qin, and S. Joty.*
[paper](https://aclanthology.org/2022.acl-long.198.pdf)
[code](https://github.com/qcwthu/continual_fewshot_relation_learning)

1. **Few-Shot Class-Incremental Learning for Named Entity Recognition,** in ACL, 2022.
*R. Wang, T. Yu, H. Zhao, S. Kim, S. Mitra, R. Zhang, and R. Henao.*
[paper](https://aclanthology.org/2022.acl-long.43.pdf)

1. **Task-Adaptive Negative Envision for Few-Shot Open-Set Recognition,** in CVPR, 2022.
*S. Huang, J. Ma, G. Han, and S. Chang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task-Adaptive_Negative_Envision_for_Few-Shot_Open-Set_Recognition_CVPR_2022_paper.pdf)
[code](https://github.com/shiyuanh/TANE)

1. **Forward Compatible Few-Shot Class-Incremental Learning,** in CVPR, 2022.
*D. Zhou, F. Wang, H. Ye, L. Ma, S. Pu, and D. Zhan.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)
[code](https://github.com/zhoudw-zdw/CVPR22-Fact)

1. **Sylph: A Hypernetwork Framework for Incremental Few-Shot Object Detection,** in CVPR, 2022.
*L. Yin, J. M. Perez-Rua, and K. J. Liang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_Sylph_A_Hypernetwork_Framework_for_Incremental_Few-Shot_Object_Detection_CVPR_2022_paper.pdf)

1. **Constrained Few-Shot Class-Incremental Learning,** in CVPR, 2022.
*M. Hersche, G. Karunaratne, G. Cherubini, L. Benini, A. Sebastian, and A. Rahimi.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf)

1. **iFS-RCNN: An Incremental Few-Shot Instance Segmenter,** in CVPR, 2022.
*K. Nguyen, and S. Todorovic.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_iFS-RCNN_An_Incremental_Few-Shot_Instance_Segmenter_CVPR_2022_paper.pdf)

1. **MetaFSCIL: A Meta-Learning Approach for Few-Shot Class Incremental Learning,** in CVPR, 2022.
*Z. Chi, L. Gu, H. Liu, Y. Wang, Y. Yu, and J. Tang.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chi_MetaFSCIL_A_Meta-Learning_Approach_for_Few-Shot_Class_Incremental_Learning_CVPR_2022_paper.pdf)

1. **Few-Shot Incremental Learning for Label-to-Image Translation,** in CVPR, 2022.
*P. Chen, Y. Zhang, Z. Li, and L. Sun.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Few-Shot_Incremental_Learning_for_Label-to-Image_Translation_CVPR_2022_paper.pdf)

1. **Revisiting Learnable Affines for Batch Norm in Few-Shot Transfer Learning,** in CVPR, 2022.
*M. Yazdanpanah, A. A. Rahman, M. Chaudhary, C. Desrosiers, M. Havaei, E. Belilovsky, and S. E. Kahou.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yazdanpanah_Revisiting_Learnable_Affines_for_Batch_Norm_in_Few-Shot_Transfer_Learning_CVPR_2022_paper.pdf)

1. **Few-Shot Learning With Noisy Labels,** in CVPR, 2022.
*K. J. Liang, S. B. Rangrej, V. Petrovic, and T. Hassner.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Few-Shot_Learning_With_Noisy_Labels_CVPR_2022_paper.pdf)

1. **Improving Adversarially Robust Few-Shot Image Classification With Generalizable Representations,** in CVPR, 2022.
*J. Dong, Y. Wang, J. Lai, and X. Xie.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Improving_Adversarially_Robust_Few-Shot_Image_Classification_With_Generalizable_Representations_CVPR_2022_paper.pdf)

1. **Geometer: Graph Few-Shot Class-Incremental Learning via Prototype Representation,** in KDD, 2022.
*B. Lu, X. Gan, L. Yang, W. Zhang, L. Fu, and X. Wang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539280)
[code](https://github.com/RobinLu1209/Geometer)

1. **Few-Shot Heterogeneous Graph Learning via Cross-Domain Knowledge Transfer,** in KDD, 2022.
*Q. Zhang, X. Wu, Q. Yang, C. Zhang, and X. Zhang.*
[paper](https://dl.acm.org/doi/10.1145/3534678.3539431)

1. **Few-Shot Adaptation of Pre-Trained Networks for Domain Shift,** in IJCAI, 2022.
*W. Zhang, L. Shen, W. Zhang, and C.-S. Foo.*
[paper](https://www.ijcai.org/proceedings/2022/0232.pdf)

1. **MemREIN: Rein the Domain Shift for Cross-Domain Few-Shot Learning,** in IJCAI, 2022.
*Y. Xu, L. Wang, Y. Wang, C. Qin, Y. Zhang, and Y. FU.*
[paper](https://www.ijcai.org/proceedings/2022/0505.pdf)

1. **Continual Few-Shot Learning With Transformer Adaptation and Knowledge Regularization,** in WWW, 2023.
*X. Wang, Y. Liu, J. Fan, W. Wen, H. Xue, and W. Zhu.*
[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583262)

1. **DENSE: Data-Free One-Shot Federated Learning,** in NeurIPS, 2022.
*J. Zhang, C. Chen, B. Li, L. Lyu, S. Wu, S. Ding, C. Shen, and C. Wu.*
[paper](https://openreview.net/pdf?id=QFQoxCFYEkA)

1. **Towards Practical Few-Shot Query Sets: Transductive Minimum Description Length Inference,** in NeurIPS, 2022.
*S. T. Martin, M. Boudiaf, E. Chouzenoux, J.-C. Pesquet, and I. B. Ayed.*
[paper](https://openreview.net/pdf?id=j9JL96S8Vl)
[code](https://github.com/SegoleneMartin/PADDLE)

1. **Task-Level Differentially Private Meta Learning,** in NeurIPS, 2022.
*X. Zhou, and R. Bassily.*
[paper](https://openreview.net/pdf?id=FhyrZ92DcI9)
[code](https://github.com/xyzhou055/MetaNSGD)

1. **FiT: Parameter Efficient Few-Shot Transfer Learning for Personalized and Federated Image Classification,** in ICLR, 2023.
*A. Shysheya, J. F. Bronskill, M. Patacchiola, S. Nowozin, and R. E. Turner.*
[paper](https://openreview.net/pdf?id=9aokcgBVIj1)
[code](https://github.com/cambridge-mlg/fit)

1. **Towards Addressing Label Skews in One-Shot Federated Learning,** in ICLR, 2023.
*Y. Diao, Q. Li, and B. He.*
[paper](https://openreview.net/pdf?id=rzrqh85f4Sc)
[code](https://github.com/Xtra-Computing/FedOV)

1. **Data-Free One-Shot Federated Learning Under Very High Statistical Heterogeneity,** in ICLR, 2023.
*C. E. Heinbaugh, E. Luz-Ricca, and H. Shao.*
[paper](https://openreview.net/pdf?id=_hb4vM3jspB)
[code](https://github.com/ceh-2000/fed_cvae)

1. **Contrastive Meta-Learning for Partially Observable Few-Shot Learning,** in ICLR, 2023.
*A. Jelley, A. Storkey, A. Antoniou, and S. Devlin.*
[paper](https://openreview.net/pdf?id=6iVJOtr2zL2)
[code](https://github.com/AdamJelley/POEM)

1. **On the Soft-Subnetwork for Few-Shot Class Incremental Learning,** in ICLR, 2023.
*H. Kang, J. Yoon, S. R. H. Madjid, S. J. Hwang, and C. D. Yoo.*
[paper](https://openreview.net/pdf?id=z57WK5lGeHd)
[code](https://github.com/ihaeyong/ SoftNet-FSCIL)

1. **Warping the Space: Weight Space Rotation for Class-Incremental Few-Shot Learning,** in ICLR, 2023.
*D.-Y. Kim, D.-J. Han, J. Seo, and J. Moon.*
[paper](https://openreview.net/pdf?id=kPLzOfPfA2l)
[code](https://github.com/EdwinKim3069/WaRP-CIFSL)

1. **Neural Collapse Inspired Feature-Classifier Alignment for Few-Shot Class-Incremental Learning,** in ICLR, 2023.
*Y. Yang, H. Yuan, X. Li, Z. Lin, P. Torr, and D. Tao.*
[paper](https://openreview.net/pdf?id=y5W8tpojhtJ)
[code](https://github.com/NeuralCollapseApplications/FSCIL)

1. **Learning With Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning,** in CVPR, 2023.
*Z. Song, Y. Zhao, Y. Shi, P. Peng, L. Yuan, and Y. Tian.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Learning_With_Fantasy_Semantic-Aware_Virtual_Contrastive_Constraint_for_Few-Shot_Class-Incremental_CVPR_2023_paper.pdf)
[code](https://github.com/zysong0113/SAVC)

1. **Few-Shot Class-Incremental Learning via Class-Aware Bilateral Distillation,** in CVPR, 2023.
*L. Zhao, J. Lu, Y. Xu, Z. Cheng, D. Guo, Y. Niu, and X. Fang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Few-Shot_Class-Incremental_Learning_via_Class-Aware_Bilateral_Distillation_CVPR_2023_paper.pdf)
[code](https://github.com/LinglanZhao/BiDistFSCIL)

1. **GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task,** in CVPR, 2023.
*H. Zhuang, Z. Weng, R. He, Z. Lin, and Z. Zeng.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.pdf)

1. **Glocal Energy-Based Learning for Few-Shot Open-Set Recognition,** in CVPR, 2023.
*H. Wang, G. Pang, P. Wang, L. Zhang, W. Wei, and Y. Zhang.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Glocal_Energy-Based_Learning_for_Few-Shot_Open-Set_Recognition_CVPR_2023_paper.pdf)
[code](https://github.com/00why00/Glocal)

1. **Open-Set Likelihood Maximization for Few-Shot Learning,** in CVPR, 2023.
*M. Boudiaf, E. Bennequin, M. Tami, A. Toubhans, P. Piantanida, C. Hudelot, and I. B. Ayed.*
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Boudiaf_Open-Set_Likelihood_Maximization_for_Few-Shot_Learning_CVPR_2023_paper.pdf)
[code](https://github.com/ebennequin/fewshot-open-set.)

1. **Federated Few-Shot Learning,** in KDD, 2023.
*S. Wang, X. Fu, K. Ding, C. Chen, H. Chen, and J. Li.*
[paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599347)
[code](https://github.com/SongW-SW/F2L)

1. **LFS-GAN: Lifelong Few-Shot Image Generation,** in ICCV, 2023.
*J. Seo, J. Kang, and G. Park.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Seo_LFS-GAN_Lifelong_Few-Shot_Image_Generation_ICCV_2023_paper.pdf)
[code](https://github.com/jjuon/lfs-gan)

1. **Domain Adaptive Few-Shot Open-Set Learning,** in ICCV, 2023.
*D. Pal, D. More, S. Bhargav, D. Tamboli, V. Aggarwal, and B. Banerjee.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Pal_Domain_Adaptive_Few-Shot_Open-Set_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/DebabrataPal7/DAFOSNET)

1. **Few-Shot Continual Infomax Learning,** in ICCV, 2023.
*Z. Gu, C. Xu, J. Yang, and Z. Cui.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_Few-shot_Continual_Infomax_Learning_ICCV_2023_paper.pdf)

1. **Prototypical Kernel Learning and Open-Set Foreground Perception for Generalized Few-Shot Semantic Segmentation,** in ICCV, 2023.
*K. Huang, F. Wang, Y. Xi, and Y. Gao.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Prototypical_Kernel_Learning_and_Open-set_Foreground_Perception_for_Generalized_Few-shot_ICCV_2023_paper.pdf)

1. **DETA: Denoised Task Adaptation for Few-Shot Learning,** in ICCV, 2023.
*J. Zhang, L. Gao, X. Luo, H. Shen, and J. Song.*
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_DETA_Denoised_Task_Adaptation_for_Few-Shot_Learning_ICCV_2023_paper.pdf)
[code](https://github.com/JimZAI/DETA)

1. **Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration,** in NeurIPS, 2023.
*Q. Wang, D. Zhou, Y. Zhang, D. Zhan, and H. Ye.*
[paper](https://openreview.net/attachment?id=8NAxGDdf7H&name=pdf)
[code](https://github.com/wangkiw/TEEN)

1. **Alignment With Human Representations Supports Robust Few-Shot Learning,** in NeurIPS, 2023.
*I. Sucholutsky, and T. L. Griffiths.*
[paper](https://openreview.net/attachment?id=HYGnmSLBCf&name=pdf)

1. **Incremental-Detr: Incremental Few-Shot Object Detection via Self-Supervised Learning,** in AAAI, 2023.
*N. Dong, Y. Zhang, M. Ding, and G. H. Lee.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25129/24901)
[code](https://github.com/dongnana777/Incremental-DETR)

1. **Bayesian Cross-Modal Alignment Learning for Few-Shot Out-of-Distribution Generalization,** in AAAI, 2023.
*L. Zhu, X. Wang, C. Zhou, and N. Ye.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26355/26127)
[code](https://github.com/LinLLLL/BayesCAL)

1. **High-Level Semantic Feature Matters Few-Shot Unsupervised Domain Adaptation,** in AAAI, 2023.
*L. Yu, W. Yang, S. Huang, L. Wang, and M. Yang.*
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26306/26078)

1. **Enhancing One-Shot Federated Learning Through Data and Ensemble Co-Boosting,** in ICLR, 2024.
*R. Dai, Y. Zhang, A. Li, T. Liu, X. Yang, and B. Han.*
[paper](https://openreview.net/attachment?id=tm8s3696Ox&name=pdf)

1. **Pre-trained Vision and Language Transformers are Few-Shot Incremental Learners,** in CVPR, 2024.
*K.-H. Park, K. Song, and G.-M. Park.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02254)

1. **OrCo: Towards Better Generalization via Orthogonality and Contrast for Few-Shot Class-Incremental Learning,** in CVPR, 2024.
*N. Ahmed, A. Kukleva, and B. Schiele.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02717)

1. **Descriptor and Word Soups Q: Overcoming the Parameter Efficiency Accuracy Tradeoff for Out-of-Distribution Few-shot Learning,** in CVPR, 2024.
*C. Liao, T. Tsiligkaridis, and B. Kulis.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02551)

1. **ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection,** in CVPR, 2024.
*Y. Bai, Z. Han, B. Cao, X. Jiang, Q. Hu, and C. Zhang.*
[paper](https://doi.org/10.1109/CVPR52733.2024.01655)

1. **Visual Prompting for Generalized Few-shot Segmentation: A Multi-scale Approach,** in CVPR, 2024.
*M. R. I. Hossain, M. Siam, L. Sigal, and J. J. Little.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02215)

1. **DeIL: Direct-and-Inverse CLIP for Open-World Few-Shot Learning,** in CVPR, 2024.
*S. Shao, Y. Bai, Y. Wang, B. Liu, and Y. Zhou.*
[paper](https://doi.org/10.1109/CVPR52733.2024.02693)

1. **Delve into Base-Novel Confusion: Redundancy Exploration for Few-Shot Class-Incremental Learning,** in IJCAI, 2024.
*H. Zhou, Y. Zou, R. Li, Y. Li, and K. Xiao.*
[paper](https://www.ijcai.org/proceedings/2024/623)

1. **FineFMPL: Fine-grained Feature Mining Prompt Learning for Few-Shot Class Incremental Learning,** in IJCAI, 2024.
*H. Sun, J. Zhou, X. He, J. Xu, and Y. Peng.*
[paper](https://www.ijcai.org/proceedings/2024/144)

1. **M2SD: Multiple Mixing Self-Distillation for Few-Shot Class-Incremental Learning,** in AAAI, 2024.
*J. Lin, Z. Wu, W. Lin, J. Huang, and R. Luo.*
[paper](https://doi.org/10.1609/aaai.v38i4.28129)

1. **Collaborative Consortium of Foundation Models for Open-World Few-Shot Learning,** in AAAI, 2024.
*S. Shao, Y. Bai, Y. Wang, B. Liu, and B. Liu.*
[paper](https://doi.org/10.1609/aaai.v38i5.28275)

1. **Scaling Few-Shot Learning for the Open World,** in AAAI, 2024.
*Z. Lin, W. Yang, H. Wang, H. Chi, L. Lan, and J. Wang.*
[paper](https://doi.org/10.1609/aaai.v38i12.29291)

1. **Does Few-Shot Learning Suffer from Backdoor Attacks?** in AAAI, 2024.
*X. Liu, X. Jia, J. Gu, Y. Xun, S. Liang, and X. Cao.*
[paper](https://doi.org/10.1609/aaai.v38i18.29965)

1. **H-ensemble: An Information Theoretic Approach to Reliable Few-Shot Multi-Source-Free Transfer,** in AAAI, 2024.
*Y. Wu, J. Wang, W. Wang, and Y. Li.*
[paper](https://doi.org/10.1609/aaai.v38i14.29528)

1. **Exploring One-Shot Semi-supervised Federated Learning with Pre-trained Diffusion Models,** in AAAI, 2024.
*M. Yang, S. Su, B. Li, and X. Xue.*
[paper](https://doi.org/10.1609/aaai.v38i15.29568)

1. **One-shot Federated Learning via Synthetic Distiller-Distillate Communication.,** in NeurIPS, 2024.
*J. Zhang, S. Liu, and X. Wang.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/ba0ad9d1e0c737800b2340b9cd68c208-Paper-Conference.pdf)

1. **FuseFL: One-Shot Federated Learning through the Lens of Causality with Progressive Model Fusion.,** in NeurIPS, 2024.
*Z. Tang, Y. Zhang, P. Dong, Y.-m. Cheung, A. C. Zhou, B. Han, and X. Chu.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/31e6e0c09325a3be16d93f84e40e0c7e-Paper-Conference.pdf)

1. **Revisiting Ensembling in One-Shot Federated Learning.,** in NeurIPS, 2024.
*Y. Allouah, A. Dhasade, R. Guerraoui, N. Gupta, A.-M. Kermarrec, R. Pinot, R. Pires, and R. Sharma.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/7ea46207ec9bda974b140fe11d8dd727-Paper-Conference.pdf)

1. **FedLPA: One-shot Federated Learning with Layer-Wise Posterior Aggregation.,** in NeurIPS, 2024.
*X. Liu, L. Liu, F. Ye, Y. Shen, X. Li, L. Jiang, and J. Li.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/9482f45fdd89aba9130bb04c44f788a9-Paper-Conference.pdf)

1. **(FL)2: Overcoming Few Labels in Federated Semi-Supervised Learning.,** in NeurIPS, 2024.
*S. Lee, T.-L. V. Le, J. Shin, and S.-J. Lee.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/4d2aa4c034745f558bfea34643c8d6a6-Paper-Conference.pdf)

1. **An Efficient Memory Module for Graph Few-Shot Class-Incremental Learning.,** in NeurIPS, 2024.
*D. Li, A. Zhang, J. Gao, and B. Qi.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/eae7fa3e1584f46253c891bcb61846b8-Paper-Conference.pdf)

1. **Byzantine Resilient and Fast Federated Few-Shot Learning.,** in ICML, 2024.
*A. P. Singh, and N. Vaswani.*
[paper](https://openreview.net/forum?id=q5q59s2WJy)

1. **Compositional Few-Shot Class-Incremental Learning.,** in ICML, 2024.
*Y. Zou, S. Zhang, H. Zhou, Y. Li, and R. Li.*
[paper](https://openreview.net/forum?id=t4908PyZxs)

1. **Meta Evidential Transformer for Few-Shot Open-Set Recognition.,** in ICML, 2024.
*H. Sapkota, K. P. Neupane, and Q. Yu.*
[paper](https://openreview.net/forum?id=CquFGSIU6w)

1. **Federated Few-Shot Class-Incremental Learning.,** in ICLR, 2025.
*M. A. Ma'sum, M. Pratama, L. Liu, Habibullah, and R. Kowalczyk.*
[paper](https://openreview.net/forum?id=ZiPoAlKf9Y)

1. **Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection.,** in ICLR, 2025.
*F. Zeng, Z. Cheng, F. Zhu, H. Wei, and X.-Y. Zhang.*
[paper](https://openreview.net/forum?id=Ew3VifXaxZ)

1. **FedTMOS: Efficient One-Shot Federated Learning with Tsetlin Machine.,** in ICLR, 2025.
*S. H. S. Qi, J. Chauhan, G. V. Merrett, and J. S. Hare.*
[paper](https://openreview.net/forum?id=44hcrfzydU)

1. **Dist Loss: Enhancing Regression in Few-Shot Region through Distribution Distance Constraint.,** in ICLR, 2025.
*G. Nie, G. Tang, and S. Hong.*
[paper](https://openreview.net/forum?id=YeSxbRrDRl)

1. **Capture Global Feature Statistics for One-Shot Federated Learning,** in AAAI, 2025.
*Z. Guan, Y. Zhou, and X. Gu.*
[paper](https://doi.org/10.1609/aaai.v39i16.33862)

1. **FedFSL-CFRD: Personalized Federated Few-Shot Learning with Collaborative Feature Representation Disentanglement,** in AAAI, 2025.
*S. Wang, J. Li, Z. Liu, Y. Zhang, and M. Gong.*
[paper](https://doi.org/10.1609/aaai.v39i20.35423)

1. **Deconfound Semantic Shift and Incompleteness in Incremental Few-shot Semantic Segmentation,** in AAAI, 2025.
*Y. Wu, Y. Xia, H. Li, L. Yuan, J. Chen, J. Liu, T. Lu, and S. Wan.*
[paper](https://doi.org/10.1609/aaai.v39i8.32915)

1. **AnchorInv: Few-Shot Class-Incremental Learning of Physiological Signals via Feature Space-Guided Inversion,** in AAAI, 2025.
*C. Li, B. Gao, G. D. Jones, T. Denison, and T. Zhu.*
[paper](https://doi.org/10.1609/aaai.v39i13.33563)

1. **Pseudo Informative Episode Construction for Few-Shot Class-Incremental Learning,** in AAAI, 2025.
*C. Chen, X. Yang, and C. Xu.*
[paper](https://doi.org/10.1609/aaai.v39i15.33729)

1. **Few-Shot Incremental Learning via Foreground Aggregation and Knowledge Transfer for Audio-Visual Semantic Segmentation,** in AAAI, 2025.
*J. Xiu, M. Li, Z. Yang, W. Ji, Y. Yin, and R. Zimmermann.*
[paper](https://doi.org/10.1609/aaai.v39i8.32950)

1. **Few-Shot Audio-Visual Class-Incremental Learning with Temporal Prompting and Regularization,** in AAAI, 2025.
*Y. Cui, L. Liu, Z. Yu, G. Huang, and X. Hong.*
[paper](https://doi.org/10.1609/aaai.v39i15.33770)

1. **Adaptive Decision Boundary for Few-Shot Class-Incremental Learning,** in AAAI, 2025.
*L. Li, Y. Tan, S. Yang, H. Cheng, Y. Dong, and L. Yang.*
[paper](https://doi.org/10.1609/aaai.v39i17.34020)



## [Datasets/Benchmarks](#content)

1. **FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset With State-of-the-Art Evaluation,** in EMNLP, 2018.
*X. Han, H. Zhu, P. Yu, Z. Wang, Y. Yao, Z. Liu, and M. Sun.*
[paper](https://www.aclweb.org/anthology/D18-1514.pdf)
[code](https://github.com/thunlp/FewRel)

1. **Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning,** arXiv preprint, 2019.
*T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine.*
[paper](https://arxiv.org/abs/1910.10897)
[code](https://meta-world.github.io/)

1. **The Omniglot Challenge: A 3-Year Progress Report,** in Current Opinion in Behavioral Sciences, 2019.
*B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum.*
[paper](https://arxiv.org/abs/1902.03477)
[code](https://github.com/brendenlake/omniglot)

1. **FewRel 2.0: Towards More Challenging Few-Shot Relation Classification,** in EMNLP-IJCNLP, 2019.
*T. Gao, X. Han, H. Zhu, Z. Liu, P. Li, M. Sun, and J. Zhou.*
[paper](https://www.aclweb.org/anthology/D19-1649.pdf)
[code](https://github.com/thunlp/FewRel)

1. **META-DATASET: A Dataset of Datasets for Learning to Learn From Few Examples,** in ICLR, 2020.
*E. Triantafillou, T. Zhu, V. Dumoulin, P. Lamblin, U. Evci, K. Xu, R. Goroshin, C. Gelada, K. Swersky, P. Manzagol, and H. Larochelle.*
[paper](https://openreview.net/pdf?id=rkgAGAVKPr)
[code](https://github.com/google-research/meta-dataset)

1. **Few-Shot Object Detection With Attention-RPN and Multi-Relation Detector,** in CVPR, 2020.
*Q. Fan, W. Zhuo, C.-K. Tang, Y.-W. Tai.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Few-Shot_Object_Detection_With_Attention-RPN_and_Multi-Relation_Detector_CVPR_2020_paper.pdf)
[code](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset)

1. **FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation,** in CVPR, 2020.
*X. Li, T. Wei, Y. P. Chen, Y.-W. Tai, and C.-K. Tang.*
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_FSS-1000_A_1000-Class_Dataset_for_Few-Shot_Segmentation_CVPR_2020_paper.pdf)
[code](https://github.com/HKUSTCV/FSS-1000)

1. **Impact of Base Dataset Design on Few-Shot Image Classification,** in ECCV, 2020.
*O. Sbai, C. Couprie, and M. Aubry.*
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610579.pdf)
[code](https://github.com/facebookresearch/fewshotDatasetDesign)

1. **A Unified Few-Shot Classification Benchmark to Compare Transfer and Meta Learning Approaches,** in NeurIPS, 2021.
*V. Dumoulin, N. Houlsby, U. Evci, X. Zhai, R. Goroshin, S. Gelly, and H. Larochelle.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/28dd2c7955ce926456240b2ff0100bde-Paper-round1.pdf)

1. **Few-Shot Learning Evaluation in Natural Language Understanding,** in NeurIPS, 2021.
*S. Mukherjee, X. Liu, G. Zheng, S. Hosseini, H. Cheng, G. Yang, C. Meek, A. H. Awadallah, and J. Gao.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/3644a684f98ea8fe223c713b77189a77-Paper-round2.pdf)
[code](https://github.com/microsoft/CLUES)

1. **FS-Mol: A Few-Shot Learning Dataset of Molecules,** in NeurIPS, 2021.
*M. Stanley, J. Bronskill, K. Maziarz, H. Misztela, J. Lanini, M. H. S. Segler, N. Schneider, and M. Brockschmidt.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/8d3bba7425e7c98c50f52ca1b52d3735-Paper-round2.pdf)
[code](https://github.com/microsoft/FS-Mol/)

1. **RAFT: A Real-World Few-Shot Text Classification Benchmark,** in NeurIPS, 2021.
*N. Alex, E. Lifland, L. Tunstall, A. Thakur, P. Maham, C. J. Riedel, E. Hine, C. Ashurst, P. Sedille, A. Carlier, M. Noetel, and A. Stuhlmüller.*
[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/ca46c1b9512a7a8315fa3c5a946e8265-Paper-round2.pdf)
[code](https://raft.elicit.org/)

1. **A Large-Scale Benchmark for Few-Shot Program Induction and Synthesis,** in ICML, 2021.
*F. Alet, J. Lopez-Contreras, J. Koppel, M. Nye, A. Solar-Lezama, T. Lozano-Perez, L. Kaelbling, and J. Tenenbaum.*
[paper](http://proceedings.mlr.press/v139/alet21a/alet21a.pdf)
[code](https://github.com/javierlc2000/progres)

1. **FEW-NERD: A Few-Shot Named Entity Recognition Dataset,** in ACL-IJCNLP, 2021.
*N. Ding, G. Xu, Y. Chen, X. Wang, X. Han, P. Xie, H. Zheng, and Z. Liu.*
[paper](https://aclanthology.org/2021.acl-long.248.pdf)
[code](https://ningding97.github.io/fewnerd/)

1. **CrossFit: A Few-Shot Learning Challenge for Cross-Task Generalization in NLP,** in EMNLP, 2021.
*Q. Ye, B. Y. Lin, and X. Ren.*
[paper](https://aclanthology.org/2021.emnlp-main.572.pdf)
[code](https://github.com/INK-USC/CrossFit)

1. **ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition,** in ICCV, 2021.
*D. Massiceti, L. Zintgraf, J. Bronskill, L. Theodorou, M. T. Harris, E. Cutrell, C. Morrison, K. Hofmann, and S. Stumpf.*
[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Massiceti_ORBIT_A_Real-World_Few-Shot_Dataset_for_Teachable_Object_Recognition_ICCV_2021_paper.pdf)
[code](https://github.com/microsoft/ORBIT-Dataset)

1. **FLEX: Unifying Evaluation for Few-Shot NLP,** in NeurIPS, 2021.
*J. Bragg, A. Cohan, K. Lo, and I. Beltagy.*
[paper](https://proceedings.neurips.cc/paper/2021/file/8493eeaccb772c0878f99d60a0bd2bb3-Paper.pdf)

1. **Two Sides of Meta-Learning Evaluation: In vs. Out of Distribution,** in NeurIPS, 2021.
*A. Setlur, O. Li, and V. Smith.*
[paper](https://proceedings.neurips.cc/paper/2021/file/1e932f24dc0aa4e7a6ac2beec387416d-Paper.pdf)

1. **Realistic Evaluation of Transductive Few-Shot Learning,** in NeurIPS, 2021.
*O. Veilleux, M. Boudiaf, P. Piantanida, and I. B. Ayed.*
[paper](https://proceedings.neurips.cc/paper/2021/file/4d7a968bb636e25818ff2a3941db08c1-Paper.pdf)

1. **Meta-Album: Multi-Domain Meta-Dataset for Few-Shot Image Classification,** in NeurIPS, 2022.
*I. Ullah, D. Carrión-Ojeda, S. Escalera, I. Guyon, M. Huisman, F. Mohr, J. N. v. Rijn, H. Sun, J. Vanschoren, and P. A. Vu.*
[paper](https://papers.nips.cc/paper_files/paper/2022/file/1585da86b5a3c4fb15520a2b3682051f-Paper-Datasets_and_Benchmarks.pdf)
[code](https://meta-album.github.io/)

1. **Geoclidean: Few-Shot Generalization in Euclidean Geometry,** in NeurIPS, 2022.
*J. Hsu, J. Wu, and N. D. Goodman.*
[paper](https://papers.nips.cc/paper_files/paper/2022/file/feb34ce77fc8b94c85d12e608b23ce67-Paper-Datasets_and_Benchmarks.pdf)
[code](https://github.com/joyhsu0504/geoclidean_framewor)

1. **FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding,** in ACL, 2022.
*Y. Zheng, J. Zhou, Y. Qian, M. Ding, C. Liao, L. Jian, R. Salakhutdinov, J. Tang, S. Ruder, and Z. Yang.*
[paper](https://aclanthology.org/2022.acl-long.38.pdf)
[code](https://github.com/THUDM/FewNLU)

1. **Bongard-Hoi: Benchmarking Few-Shot Visual Reasoning for Human-Object Interactions,** in CVPR, 2022.
*H. Jiang, X. Ma, W. Nie, Z. Yu, Y. Zhu, and A. Anandkumar.*
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Bongard-HOI_Benchmarking_Few-Shot_Visual_Reasoning_for_Human-Object_Interactions_CVPR_2022_paper.pdf)
[code](https://github.com/nvlabs/Bongard-HOI)

1. **Hard-Meta-Dataset++: Towards Understanding Few-Shot Performance on Difficult Tasks,** in ICLR, 2023.
*S. Basu, M. Stanley, J. F. Bronskill, S. Feizi, and D. Massiceti.*
[paper](https://openreview.net/pdf?id=wq0luyH3m4)

1. **MEWL: Few-Shot Multimodal Word Learning With Referential Uncertainty,** in ICML, 2023.
*G. Jiang, M. Xu, S. Xin, W. Liang, Y. Peng, C. Zhang, and Y. Zhu.*
[paper](https://proceedings.mlr.press/v202/jiang23i/jiang23i.pdf)
[code](https://github.com/jianggy/MEWL)

1. **UNISUMM and SUMMZOO: Unified Model and Diverse Benchmark for Few-Shot Summarization,** in ACL, 2023.
*Y. Chen, Y. Liu, R. Xu, Z. Yang, C. Zhu, M. Zeng, and Y. Zhang.*
[paper](https://aclanthology.org/2023.acl-long.718.pdf)
[code](https://github.com/microsoft/UniSumm)

1. **EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models,** in NeurIPS, 2023.
*M. Wornow, R. Thapa, E. Steinberg, J. A. Fries, and N. Shah.*
[paper](https://openreview.net/attachment?id=CsXC6IcdwI&name=pdf)
[code](https://github.com/som-shahlab/ehrshot-benchmark)

1. **CORE: A Few-Shot Company Relation Classification Dataset for Robust Domain Adaptation,** in EMNLP, 2023.
*P. Borchert, J. D. Weerdt, K. Coussement, A. D. Caigny, and M.-F. Moens.*
[paper](https://aclanthology.org/2023.emnlp-main.722.pdf)
[code](https://github.com/pnborchert/CORE)

1. **The COT Collection: Improving Zero-Shot and Few-Shot Learning of Language Models via Chain-of-Thought Fine-Tuning,** in EMNLP, 2023.
*S. Kim, S. Joo, D. Kim, J. Jang, S. Ye, J. Shin, and M. Seo.*
[paper](https://aclanthology.org/2023.emnlp-main.782.pdf)
[code](https://github.com/kaistAI/CoT-Collection)

1. **JASMINE: Arabic GPT Models for Few-Shot Learning,** in EMNLP, 2023.
*E. M. B. Nagoudi, M. Abdul-Mageed, A. Elmadany, A. Inciarte, and M. T. I. Khondaker.*
[paper](https://aclanthology.org/2023.emnlp-main.1040.pdf)

1. **Fine-Tuned LLMs Know More, Hallucinate Less With Few-Shot Sequence-to-Sequence Semantic Parsing Over Wikidata,** in EMNLP, 2023.
*S. Xu, S. Liu, T. Culhane, E. Pertseva, M.-H. Wu, S. Semnani, and M. Lam.*
[paper](https://aclanthology.org/2023.emnlp-main.353.pdf)
[code](https://github.com/stanford-oval/wikidata-emnlp23)

1. **MetaCoCo: A New Few-Shot Classification Benchmark With Spurious Correlation,** in ICLR, 2024.
*M. Zhang, H. Li, F. Wu, and K. Kuang.*
[paper](https://openreview.net/attachment?id=DiWRG9JTWZ&name=pdf)

1. **Bongard-OpenWorld: Few-Shot Reasoning for Free-Form Visual Concepts in the Real World,** in ICLR, 2024.
*R. Wu, X. Ma, Q. Li, Z. Zhang, W. Wang, S.-C. Zhu, and Y. Wang.*
[paper](https://openreview.net/attachment?id=hWS4MueyzC&name=pdf)

1. **Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark.,** in NeurIPS, 2024.
*B. Karpowicz, J. Ye, C. Fan, P. Tostado-Marcos, F. Rizzoglio, C. Washington, T. Scodeler, D. d. Lucena, S. R. Nason-Tomaszewski, M. Mender, X. Ma, E. M. Arneodo, L. R. Hochberg, C. A. Chestek, J. M. Henderson, T. Gentner, V. Gilja, L. E. Miller, A. Rouse, R. Gaunt, J. L. Collinger, and C. Pandarinath.*
[paper](http://papers.nips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)

1. **tinyBenchmarks: evaluating LLMs with fewer examples.,** in ICML, 2024.
*F. M. Polo, L. Weber, L. Choshen, Y. Sun, G. Xu, and M. Yurochkin.*
[paper](https://openreview.net/forum?id=qAml3FpfhG)

## [Software Library](#content)

1. **PaddleFSL,** a library for few-shot learning written in *PaddlePaddle*.
[link](https://github.com/tata1661/FSL-Mate/tree/master/PaddleFSL)

1. **Torchmeta,** a library for few-shot learning & meta-learning written in *PyTorch*.
[link](https://github.com/tristandeleu/pytorch-meta#torchmeta)

1. **learn2learn,** a library for meta-learning written in *PyTorch*.
[link](https://github.com/learnables/learn2learn)

1. **keras-fsl,** a library for few-shot learning written in *Tensorflow*.
[link](https://github.com/few-shot-learning/Keras-FewShotLearning)





