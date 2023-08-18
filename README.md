## Notice

This code repository is used to store all the code required by Mr. Zhining Hu to complete his master's thesis in automotive computer science, **Utilizing Generative Adversarial Networks for Image Data Augmentation of Semiconductor Wafer Dicing Induced Faults for Defect Classification**, at the Chemnitz University of Technology.

All contents of this code repository are only shared with members of Jun.-Prof. Dr. Danny Kowerko's team at Chemnitz University of Technology for review or use. It may not be downloaded, copied, modified, or used commercially by other uninvolved persons.

## Recommended Configuration and Environment
* **Operating System** : Windows, ubantu
* **GPU** :
  - NVIDIA T4/4 (GAN)
  - NVIDIA GeForce RTX 2080 Ti (ResNet)
* **CUDA** : 11.4
* **Special Requirements Involving StyleGan 3** :GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* **Python libraries** : see [environment.yaml](./environment.yaml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
  - `conda env create -f environment.yaml`


## Getting started
* Gan Model Training
  - **DCGAN** : `./DCGan/train.py`
  - **CycleGan** : `./CycleGan/train.py`
  - **StyleGan** : `./StyleGan/stylegan3_train.py`

* Image Data Generation
  - **DCGAN** : `./DCGan/test-from_dir.py`
  - **CycleGan** :
    - `./CycleGan/test-from_dirA2B.py` (Conversion of faulty samples into flawless samples)
    - `./CycleGan/test-from_dirB2A.py` (Conversion of flawless samples into faulty samples)
  - **StyleGan** : `./StyleGan/stylegan3_test.py`

* Experimentation and Verification
  - **ResNet** : `./ResNet152V2.py`

* Parameters and Tuning
  Please look for this specially labeled field in the file above, where all adjustable parameters for training, generation, and testing are located. The meaning of each parameter is commented on in detail.
  ```
  '''---------------------------The following is the definition of the parameter section---------------------------------'''
  #the tuning parameter only needs to be changed here, but not anywhere else
  

  '''---------------------------Above is the definition parameter section------------------------------------------------'''
  ```
  **Please do not modify anything other than this field.**

* Organization of data sets
  - **Dataset preprocessing and hybrid dataset synthesis** : `./synthesize.ipynb`

## References:
* **DCGAN**
  - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), Radford et al. 2015
  - [PyTorch Implementation of DCGAN trained on the CelebA dataset](https://github.com/Natsu6767/DCGAN-PyTorch) by [Natsu6767](https://github.com/Natsu6767) (GitHub)
  - [PyTorch implementations of Generative Adversarial Networks](https://github.com/eriklindernoren/PyTorch-GAN) by [eriklindernoren](https://github.com/eriklindernoren) (GitHub)
  - [labml.ai Annotated PyTorch Paper Implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) by [labmlai](https://github.com/labmlai) (GitHub)
* **CycleGan**
  - [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), Zhu et al. 2017
  - [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by [junyanz](https://github.com/junyanz) (GitHub)
  - [A clean and readable Pytorch implementation of CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN) by [aitorzip](https://github.com/aitorzip) (GitHub)
* **StyleGan**
  - [Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/1703.10593), Karras et al. 2021
  - [Official PyTorch implementation of StyleGAN3](https://github.com/NVlabs/stylegan3) by [NVlabs](https://github.com/NVlabs) (GitHub)
* **ResNet**
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1), He et al. 2015
  - [torchvision.models.resnet152](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet152.html) by PyTorch
