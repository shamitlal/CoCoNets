import numpy as np 
import matplotlib.pyplot as plt 
import ipdb
import os 
st = ipdb.set_trace
fname = "/Users/shamitlal/Desktop/temp/tsdf/085471.npy"
a = np.load(fname)
# plt.imshow(a)
# plt.show(block=True)

# import pickle 
# a = pickle.load(open('/Users/shamitlal/Desktop/temp/tsdf/15885536913012376.p','rb'))
# rgb = a['rgb_camXs_raw']
# plt.imshow(rgb[0])
# plt.show(block=True)
# aa=1
# 0, 12, cam10, cam20, CameraRGB2
path = "/Users/shamitlal/Desktop/temp/tsdf/CameraRGB2"
# path = "/hdd/carla97/PythonAPI/examples/movingdata/drone_aq/Town05_episode_0000_vehicles_050/CameraRGB10"
npys = os.listdir(path)
npys = [a.split('.')[0] for a in npys if 'c2w' not in a]
npys.sort(key = int)
for npy in npys:
    npy = npy + '.npy'
    if 'c2w' in npy:
        continue
    a = np.load(os.path.join(path, npy))
    plt.imshow(a)
    plt.show(block=True)


'''
Good ones:
/hdd/carla97/PythonAPI/examples/movingta/drone_ak/Town01_episode_0000_vehicles_050/CameraRGB5
/hdd/carla97/PythonAPI/examples/movingdata/drone_an/Town03_episode_0000_vehicles_050/CameraRGB25
'''