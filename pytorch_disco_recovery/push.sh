rsync -avtu --exclude="log*" --exclude="offline_cluster" --exclude="cuda_ops*" --exclude="dump" --exclude="cuda_ops*"  --exclude="checkpoints" --exclude="__py**" --exclude="offline_obj_cluster" ./*  cmu:~/projects/pytorch_disco/

