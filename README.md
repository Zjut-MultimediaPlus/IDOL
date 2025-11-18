# IDOL
The code of "IDOL: Meeting Diverse Distribution Shifts with Prior Physics for Tropical Cyclone Multi-Task Estimation".
## Introductionaccept
![image](https://github.com/Zjut-MultimediaPlus/IDOL/blob/main/fig-IDOL.png)

Contribution:
1. To address concept shift in multi-task learning, we propose a Task Dependency Flow learning module. By incorporating the prior wind field model, the conditional probabilities of multiple specific tasks are decoupled to model the dependencies among tasks, thereby facilitating the learning of distinct TC attribute identities.
2. To address covariate and label shifts, we design a Correlation-Aware Information Bridge to model the latent distribution aligned with both ends of the model. The resulting task-shared identity serves as an information bridge between the input and output by capturing their invariant physical correlations.
3. Extensive experiments are conducted on multiple TC estimation and prediction tasks to evaluate the effectiveness of the proposed IDOL. The results demonstrate the efficacy of IDOL in handling diverse distribution shifts through feature space constraints informed by prior physical knowledge.

## Requirements 
* python 3.8.8
* Pytorch 1.1.0
* CUDA 11.7

## Citation
If you find this work useful, please consider citing us!   
```bibtex
@inproceedings{chen2025unirestore,
  title={UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior},
  author={Chen, I and Chen, Wei-Ting and Liu, Yu-Wei and Chiang, Yuan-Chun and Kuo, Sy-Yen and Yang, Mi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17969--17979},
  year={2025}
}
