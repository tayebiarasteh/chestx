# Flexible Federated Learning (FFL)


[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)


Overview
------

* Code DOI: https://doi.org/10.5281/zenodo.7513925
* This is the official repository of the paper [**Collaborative Training of Medical Artificial Intelligence Models with non-uniform Labels**](https://arxiv.org/abs/2211.13606).
* Pre-print version: [https://arxiv.org/abs/2211.13606](https://arxiv.org/abs/2211.13606)


Introduction
------
Artificial intelligence (AI) methods are revolutionizing medical image analysis. However, robust AI models require large multi-site datasets for training. While multiple stakeholders have provided publicly available datasets, the ways in which these data are labeled differ widely. For example, one dataset of chest radiographs might contain labels denoting the presence of metastases in the lung, while another dataset of chest radiograph might focus on the presence of pneumonia. With conventional approaches, these data cannot be used together to train a single AI model. We propose a new framework that we call flexible federated learning (FFL) for collaborative training on such data. Using publicly available data of 695,000 chest radiographs from five institutions - each with differing labels - we demonstrate that large and heterogeneously labeled datasets can be used to train one big AI model with this framework. We find that models trained with FFL are superior to models that are trained on matching annotations only. This may pave the way for training of truly large-scale AI models that make efficient use of all existing data.

![](./intro.png)


### Prerequisites

The software is developed in **Python 3.8**. For the deep learning, the **PyTorch 1.4** framework is used. The secure federated learning process was developed using **PySyft 0.2.9**.



Main Python modules required for the software can be installed from ./requirements in three stages:

1. Create a Python3 environment by installing the conda `environment.yml` file:

```
$ conda env create -f environment.yml
$ source activate FFL
$ pip install syft==0.2.9
```


2. Install the remaining dependencies from `requirements.txt`.


**Note:** These might take a few minutes.


Code structure
---

Our source code for flexible federated learning as well as training and evaluation of the deep neural networks, image analysis and preprocessing, and data augmentation are available here.

1. Everything can be run from *./main_2D_chestx.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, augmentation, and loading files.
* *./Train_Valid_chestx.py* contains the training and validation processes for central training.
* *./Train_Valid_chestx_federated.py* contains the training and validation processes for FFL process.
* *./single_head_Train_Valid_chestx.py* contains the training and validation processes for further training of individual classification heads.
* *./Prediction_chestx.py* all the prediction and testing processes.

------
### In case you use this repository, please cite the original paper:

S. Tayebi Arasteh, P. Isfort, M. Saehn, G. Mueller-Franzes, F. Khader, J.N. Kather, C. Kuhl, S. Nebelung, D. Truhn. "*Collaborative Training of Medical Artificial Intelligence Models with non-uniform Labels*". ArXiv, 2211.13606, https://arxiv.org/abs/2211.13606, 2022.

### BibTex

    @article {FFL2022,
      author = {Tayebi Arasteh, Soroosh and Isfort, Peter and Saehn, Marwin and Mueller-Franzes, Gustav and Khader, Firas and Kather, Jakob Nikolas and Kuhl, Christiane and Nebelung, Sven and Truhn, Daniel},
      title = {Collaborative Training of Medical Artificial Intelligence Models with non-uniform Labels},
      year = {2022},
      doi = {10.48550/ARXIV.2211.13606},
      publisher = {arXiv},
      URL = {https://arxiv.org/abs/2211.13606},
      journal = {arXiv}
    }
