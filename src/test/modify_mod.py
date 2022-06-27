import os
dire = "/projects/katefgroup/datasets/carla/processed/npzs/surveil_traj_ah_s50_i1"
filename = "/projects/katefgroup/datasets/carla/processed/npzs/surveil_traj_ah_s50_i1.txt"
newfilename = "/projects/katefgroup/datasets/carla/processed/npzs/surveil_traj_ah_s50_i1_new.txt"
fi = open(filename, "w")
finew = open(newfilename, "w")

# for line in fi.readlines():
#     print("line is: ", line)
#     finew.write("surveil_traj_ah_s50_i1/"+line[2:])

files = [npy for npy in os.listdir(dire) if npy.endswith(".npz")]
for filename in files:
    fi.write(f"./{filename}\n")
    finew.write(f"./surveil_traj_ah_s50_i1/{filename}\n")

finew.close()
fi.close()