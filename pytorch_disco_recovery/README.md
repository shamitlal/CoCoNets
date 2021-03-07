# First-time setup

Install torch

`conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`

Install cmake

`conda install -c anaconda cmake`

Install cv2 (opencv):

`conda install -c conda-forge opencv`

Install cuda corr3d

```
cd cuda_ops/corr3d
python setup.py install
cd ..
```

Install tensorboardX from `https://github.com/zuoym15/tensorboardX`

Install these pip things:

`pip install moviepy scikit-image cupy-cuda100 fire`

Run

`./carla_sta_go.sh`

# Tensorboard

With some cpu resources, on say node `0-16`, run tensorboard with a command like `./tb.sh 3456`.

On the head node, open a tunnel to your tensorboard node, with a command like `./tunnel.sh 3456 0-16`.

# Development

If multiple people are developing in a mode, it is useful for each contributer to run things in `CUSTOM` mode. Run `./custom_go.sh` to do this. If you do not have an `exp_custom.py` file, you should create one. You can copy from any other experiments file. For example, you may want to start with `cp exp_carla_sta.py exp_custom.py`.

Note that `exp_custom.py` is in the `.gitignore` for this repo! This is because custom experiments are private experiments -- and do not usually last long. Once you get a solid result that you would like others to be able to reproduce, add it to one of the main `exp_whatever.py` files and push it to the repo.
