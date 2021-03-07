import os
import glob
import random

mod = "ae_day_single"
mod = "af_day_empty"
mod = 'ae_day_single_SC_1'
mod = "multi_obj_480_a_200"
mod = 'multi_obj_480_a_200'
mod = 'single_obj_large_480_g_SC_1_16'
mod = 'ae_day_single_All_1'
mod = 'ae_day_multi'
mod = 'single_obj_large_480_g'
mod = 'multi_obj_480_a_200'
mod ='multi_obj_480_a'

mod = 'single_obj_large_480_g_SC_1_48'
mod = 'singleVehicle_farCam_SC_1_52'
mod = 'twoVehicle_farCam_200'
mod = 'twoVehicle_farCam'

mod = 'multiVehicle_farCam_200'
mod = 'allVehicle'
mod = 'single_obj_large_480_a'
mod = 'single_obj_large_480_g_SC_1_100'
mod = 'singleVehicle_farCam_cor_SC_1_200'
mod = 'multi_obj_480_a_selected'
mod ='stahs50i1'
mod ='complete_small'

all_set = []
import ipdb

def prebasename(val):
	return "/".join(val.split("/")[-2:])

st = ipdb.set_trace
import socket
hostname = socket.gethostname()
if "Alien" in hostname:
	out_dir_base = '/media/mihir/dataset/shamit_carla/npys/'
	out_dir_base = "/media/mihir/dataset/clevr_lang/npys"
	out_dir_base = "/media/mihir/dataset/clevr_vqa/raw/npys"
	out_dir_base = '/media/mihir/DATA/dataset/carla/npzs'
	dataset = ["twoVehicle_farCam"]
	dataset = ["singleVehicle_farCam"]	
	dataset = ["single_obj_large_480_a"]
	dataset = ['surveil_traj_ah_s50_i1']
	dataset = ['complete_ac_s10_i2']
elif "compute" in hostname:
	out_dir_base = "/home/mprabhud/dataset/clevr_veggies/npys"
	out_dir_base = "/home/mprabhud/dataset/carla/npy"
	out_dir_base = '/home/mprabhud/dataset/clevr_lang/npys'
	out_dir_base = '/home/shamitl/datasets/clevr_lang/npys/'
	out_dir_base = "/projects/katefgroup/datasets/shamit_carla_correct/npys"	
	out_dir_base = "/projects/katefgroup/datasets/clevr_vqa/raw/npys"
	dataset = ["aa_day","ab_day","ac_day","ad_day","ad_day_multi","aa_day_multi"]
	dataset = ["aa_day","ab_day","ac_day","ad_day"]
	dataset = ["aa_day_empty","ab_day_empty","ac_day_empty","ad_day_empty","ae_day_empty"]
	dataset = ["multiVehicle_farCam",'singleVehicle_farCam_cor','multiVehicle_farCam_b','twoVehicle_farCam']
	dataset = ["multiVehicle_farCam",'multiVehicle_farCam_b']
	dataset = ["twoVehicle_farCam"]
	dataset = ['singleVehicle_farCam_cor','singleVehicle_farCam_cor_b']
	dataset = ['multi_obj_480_a']
else:
	out_dir_base = "/projects/katefgroup/datasets/carla/npy"
	dataset = ["bb","tv_updated"]	

lengths= []

for i in dataset:
	current  = glob.glob("%s/%s/*"%(out_dir_base,i))
	all_set =  all_set + current
	lengths.append(len(current))
print(lengths,"lengths")	
split = int(len(all_set)*0.9)

if len(all_set)<200:
	all_set =  all_set + all_set

print(len(all_set),"total length")
split = 200
print(split)

print(len(all_set),split)
random.shuffle(all_set)
print(mod)
# st()
with open(out_dir_base + '/%st.txt' % mod, 'w') as f:
	for item in all_set[:]:
		if "*" not in item:
			f.write("%s\n" % prebasename(item))
if "Alien" in hostname:
	with open(out_dir_base + '/%sv.txt' % mod, 'w') as f:
		for item in all_set[:split+1]:
			if "*" not in item:
				f.write("%s\n" % prebasename(item))	
else:
	with open(out_dir_base + '/%sv.txt' % mod, 'w') as f:
		for item in all_set[split:]:
			if "*" not in item:
				f.write("%s\n" % prebasename(item))
