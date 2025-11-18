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

## Datatset
This dataset contains multi-channel infrared satellite images centered on tropical cyclones.  
The dataset is provided in compressed formats to reduce storage size.  
Files shared via cloud storage:  
Link: https://pan.baidu.com/s/1_Y4gwEM1NRE9zMpUUWK0kQ  Code: 4jtp  
- Dataset Description
Inside the extracted directory:  
Each file corresponds to a satellite infrared cloud image centered on a tropical cyclone.  
Files are stored in NumPy .npy format.  
Each .npy file contains 4 infrared (IR) channels, the general shape of each data file is: (4, 156, 156)  
Please follow the instructions below to decompress and use the data.
```bash
tar -xzvf 45_270du_k89_4ch1_data.tar.gz
---

## Citation
If you find this work useful, please consider citing us!   
```bibtex
@article{IDOL,
    title={IDOL: Meeting Diverse Distribution Shifts with Prior
    Physics for Tropical Cyclone Multi-Task Estimation},
    author={Hanting Yan and Pan Mu, Shiqi Zhang and Yuchao Zhu and Jinglin Zhang, and Cong Bai},
    journal={Advances in neural information processing systems},
    year={2025}
}
