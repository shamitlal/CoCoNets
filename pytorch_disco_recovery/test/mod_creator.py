import os 

import pdb 
st = pdb.set_trace

path = "/home/mprabhud/dataset/carla/processed/npzs"
mod = "new_complete_ad_s10_i1"
# st()
fi = open(f"{path}/{mod}v.txt", "w")
files = os.listdir(os.path.join(path, mod))

for fname in files:
    fi.write(f"{mod}/{fname}\n")
fi.close()