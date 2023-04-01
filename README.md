# MANAS
## This is an implementation of Multi-scale Attentive Image De-raining Networks via Neural Architecture Search  
[[Paper Link]](https://ieeexplore.ieee.org/document/9894375 "悬停显示")(TCSVT 2023)
## Abstract
Multi-scale architectures and attention modules
have shown effectiveness in many deep learning-based image
de-raining methods. However, manually designing and integrating
these two components into a neural network requires
a bulk of labor and extensive expertise. In this article, a
high-performance multi-scale attentive neural architecture search
(MANAS) framework is technically developed for image deraining.
The proposed method formulates a new multi-scale
attention search space with multiple flexible modules that are
favorite to the image de-raining task. Under the search space,
multi-scale attentive cells are built, which are further used to construct
a powerful image de-raining network. The internal multiscale
attentive architecture of the de-raining network is searched
automatically through a gradient-based search algorithm, which
avoids the daunting procedure of the manual design to some
extent. Moreover, in order to obtain a robust image de-raining
model, a practical and effective multi-to-one training strategy is
also presented to allow the de-raining network to get sufficient
background information from multiple rainy images with the
same background scene, and meanwhile, multiple loss functions
including external loss, internal loss, architecture regularization
loss, and model complexity loss are jointly optimized to achieve
robust de-raining performance and controllable model complexity.
Extensive experimental results on both synthetic and realistic
rainy images, as well as the down-stream vision applications (i.e.,
objection detection and segmentation) consistently demonstrate
the superiority of our proposed method.
## Requirements
Python 3.6  
PyTorch 1.10.0
## Quick Start
### Test
1. Pre-trained model (Trained on DID-MDN dataset) had been uploaded in ./eval-EXP/

2. Test the proposed MANAS:

   ```
   python test_view.py
   ```

### Train (have two stages: architecture search and model training)
1. Architecture search stage
    
   ```
   python train_search_batch.py
   ```
   
2. Model training stage

   ```
   python train_model_batch.py
   ```

## Citation
#### Please consider cite our work if you find it helpful.

```
@article{cai2023multi,  
    title={Multi-scale Attentive Image De-raining Networks via Neural Architecture Search},  
    author={Cai, Lei and Fu, Yuli and Huo, Wanliang and Xiang, Youjun and Zhu, Tao and Zhang, Ying and Zeng Huanqiang and Zeng Delu},  
    journal={IEEE Transactions on Circuits and Systems for Video Technology},  
    volume={33},  
    number={2},  
    pages={618-633},  
    year={2023},  
    doi={10.1109/TCSVT.2022.3207516}    	  
}
```
## Acknowledgments
Authors of this work are affiliated with School of Electronic and Information Engineering, South China University of Technology. This work has been supported by the supported in part by the National Key Research and Development Program of China under the grant 2021YFE0205400, in part by the Natural Science Foundation of Guangdong Province under the Grant 2019A1515010861, in part by Guangzhou Technical Project under the Grant 201902020008, in part by NSFC under the Grant 61471174, and in part by Fundamental Research Program of Guangdong under the Grant 2020B1515310023.

#### Parts of this code repository is based on the following works:
* https://github.com/XLearning-SCU/2020-NeurIPS-CLEARER
