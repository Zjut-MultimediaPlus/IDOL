# IDOL
The code of "IDOL: Meeting Diverse Distribution Shifts with Prior Physics for Tropical Cyclone Multi-Task Estimation" was accepted to NIPS2025.

## Introductionaccept
![image](https://github.com/yht1214/IDOL/blob/main/figs/fig-IODL.png)

Contribution:
1. To address concept shift in multi-task learning, we propose a Task Dependency Flow learning module. By incorporating the prior wind field model, the conditional probabilities of multiple specific tasks are decoupled to model the dependencies among tasks, thereby facilitating the learning of distinct TC attribute identities.
2. To address covariate and label shifts, we design a Correlation-Aware Information Bridge to model the latent distribution aligned with both ends of the model. The resulting task-shared identity serves as an information bridge between the input and output by capturing their invariant physical correlations.
3. Extensive experiments are conducted on multiple TC estimation and prediction tasks to evaluate the effectiveness of the proposed IDOL. The results demonstrate the efficacy of IDOL in handling diverse distribution shifts through feature space constraints informed by prior physical knowledge.

## Requirements 
* python 3.8.8
* Pytorch 1.1.0
* CUDA 11.7
## Dataset
Physical Dynamic TC datasets (PDTC) will be open-sourced upon acceptance of the paper.
