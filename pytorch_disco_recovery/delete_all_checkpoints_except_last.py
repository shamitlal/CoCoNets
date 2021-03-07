import os
import re
import ipdb
st = ipdb.set_trace
import glob
checkpoints = glob.glob("checkpoints/*")
total = len(checkpoints)

for ind, checkpoint_dir in enumerate(checkpoints):
        print(ind,total)
        digit_regex = re.compile('\d+')
        old_checkpoints = sorted([int(digit_regex.findall(old_model)[0]) for old_model in os.listdir(checkpoint_dir) if 'model' in old_model])
        for old_checkpoint in old_checkpoints[:-1]:
                os.remove(os.path.join(checkpoint_dir, "model-%d.pth"%(old_checkpoint)))
