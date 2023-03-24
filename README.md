# MANAS
## This is an implementation of Multi-scale Attentive Image De-raining Networks via Neural Architecture Search  
Our paper can be found in [[Paper Link]](https://ieeexplore.ieee.org/document/9894375 "悬停显示")(TCSVT 2023)
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
### Dataset
You can refer to the following link [[IEEE]](https://ieeexplore.ieee.org/document/9894375 "悬停显示") to download Rain1200 dataset, and 
