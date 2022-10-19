# CoCoNets: Continuous Contrastive 3D Scene Representations

** (Code in progress) **
This repository contains the code that accompanies our CVPR 2021 paper [CoCoNets: Continuous Contrastive 3D Scene Representations](https://mihirp1998.github.io/project_pages/coconets/)

<img src="vis.gif">

### Installation
First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use pip install.
`pip install -r requirements.txt`


### Dataset and checkpoints ###

You can find the CARLA and Kitti dataset, along with the pretrained checkpoints  [here](https://drive.google.com/drive/u/1/folders/1mLk837YmNAF0rfiUyDSrzInfNXL4kN6n). 

For instance download, the CARLA dataset zip file from [here](https://drive.google.com/file/d/1aySBNPNmDZ0mG6bYUj_SKUxFKeyyzpeD/view?usp=sharing)

After extracting the dataset edit the ``dataset_location`` variable [here](https://github.com/shamitlal/CoCoNets/blob/581b616b5a89dae05233c8cf036e77ee5b88fd97/pytorch_disco_recovery/exp_base.py#L21) 

<!-- It is around 20GBs. Download and extract it in ``pytorch_disco`` repository.  -->



### Multiview Training ###
Checkpoints will be saved in ``pytorch_disco/checkpoints`` directory. 
Any checkpoint model that you want to be loaded should be placed at the end in pretrained_nets_carla.py file.

``python main.py cm --en trainer_implicit --rn <run_name>``

## Evaluation on Tracking task ###

``python main.py cz --en tester_pointfeat_dense --rn <run_name>``
