## Notice

This code repository is used to store all the code required by Mr. Zhining Hu to complete his master's thesis in automotive computer science, **Utilizing Generative Adversarial Networks for Image Data Augmentation of Semiconductor Wafer Dicing Induced Faults for Defect Classification**, at the Chemnitz University of Technology.

All contents of this code repository are only shared with members of Jun.-Prof. Dr. Danny Kowerko's team at Chemnitz University of Technology for review or use. It may not be downloaded, copied, modified, or used commercially by other uninvolved persons.

## Recommended Configuration and Environment
* **Operating System** : Windows 10, ubantu
* **GPU** :
* **CUDA** :
* **Special Requirements Involving StyleGan 3** :GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).


## Getting started
* Gan Model Training
  - **DCGAN** : `./DCGan/train.py`
  - **CycleGan** : `./CycleGan/train.py`
  - **StyleGan** : `./StyleGan/stylegan3_train.py`

* Image Data Generation
  - **DCGAN** : `./DCGan/test-from_dir.py`
  - **CycleGan** :  `./CycleGan/test-from_dirA2B.py` (Conversion of faulty samples into flawless samples)
                    `./CycleGan/test-from_dirB2A.py` (Conversion of flawless samples into faulty samples)
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


