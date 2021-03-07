# 3CNets

This repository contains the code release for: **Continuous Convolutional Contrastive 3D Scene Representations for ViewPredictive Self-supervised Visual Feature Learning (CVPR 2021)**.

## pytorch_disco
This repository contains the code for all our models. 

### Installation
First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use anaconda.
`conda env create -f environment.yaml`
`conda activate pdisco`

### Dataset ###
#### Use pre rendered dataset ####
We strongly recommend to use the pre-rendered dataset. This dataset can be donwloaded from [here](https://drive.google.com/file/d/14danQIUYmZ-R3Gy3rRe9xiuAVDbgoEqD/view?usp=sharing). It is around 20GBs. Download and extract it in ``pytorch_disco`` repository. 

### Multiview Training ###
Checkpoints will be saved in ``pytorch_disco/checkpoints`` directory. 
Any checkpoint model that you want to be loaded should be placed at the end in pretrained_nets_carla.py file.

``python main.py cm --en trainer_implicit --rn <run_name>``

## Evaluation on Tracking task ###

``python main.py cz --en tester_pointfeat_dense --rn <run_name>``