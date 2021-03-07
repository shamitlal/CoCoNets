import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from scipy.misc import imsave, imresize
from imageio import imsave

fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f26_k5_taps100i2vce_ns_pip26_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f26_k5_taps100i2vce_ns_pip27_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f26_k5_taps100i2vce_ns_pip28_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip29_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip30_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip31_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip32_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip34_output_dict.npy"
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip33_output_dict.npy"

fn1 = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip35_output_dict.npy" #
fn2 = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip36_output_dict.npy" #

fns = []
fns.append("01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip41_output_dict.npy")
fns.append("01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip38_output_dict.npy")
# fns.append("01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip43_output_dict.npy")
# fns.append("01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip46_output_dict.npy")
# fns.append("01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip47_output_dict.npy")
# fns.append("01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip52_output_dict.npy")
fns.append("01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip54_output_dict.npy")

names = ['matcher',
         'matcher+forecaster',
         'matcher+forecaster+top3_templates',
]
colors = ['indianred',
          'seagreen',
          'darkblue',
]
linestyles = ['o-',
              'o-',
              'o-',
]
fs1 = 18 # axis headers
fs2 = 18 # legend, ticks
fam = 'Times New Roman'


os = []

# o = np.load(fn1, allow_pickle=True)
# o = o.item()
# os.append(o)
# o = np.load(fn2, allow_pickle=True)
# o = o.item()
# os.append(o)

for ind, fn in enumerate(fns):
    o = np.load(fn, allow_pickle=True)
    o = o.item()
    
    ious = o['all_ious']
    # these are N x S

    # print('overall mean iou', np.mean(ious))
    # print('%s; end mean iou' % fn, np.mean(ious[:,-1]))
    print('%s; end mean iou' % names[ind], np.mean(ious[:,-1]))

    ious = np.reshape(ious, (-1))
    # print('ious', ious.shape)


    ious = o['all_ious']
    # these are N x S
    N, S = list(ious.shape)
    # print('N, S', N, S)
    time = list(range(S))

    if False:
        for n in list(range(N)):
            # o_line, = plt.plot(time, ious[n], '.', linewidth=2, color=colors[0])
            o_line, = plt.plot(time, ious[n], '-', linewidth=2, color=colors[ind])
            if n==0:
                o_line.set_label(names[ind])
        # o_line, = plt.plot(time, np.mean(ious, axis=0), '--', linewidth=5, color='red')#, color=colors[0])
    else:
        o_line,_,_ = plt.errorbar(time, np.mean(ious, axis=0), np.std(ious, axis=0), barsabove=True, color=colors[ind])#, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False,
        o_line.set_label(names[ind])
        o_line, = plt.plot(time, np.mean(ious, axis=0), linewidth=3, color=colors[ind])#, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False,
        # o_line, = plt.plot(time, ious[n], '-', linewidth=2, color=colors[ind])
        

    
    
plt.xlabel('Time', fontsize=fs1, family=fam)
plt.ylabel('IOU', fontsize=fs1, family=fam)
# plt.ylim([0.0, 1.0])
# plt.xlim([0.0, 1.0])
plt.grid(True,which="minor",ls="-", color='0.9')
plt.grid(True,which="major",ls="-", color='0.75')
# plt.legend(prop={'family':fam, 'size':fs2}, loc='upper right')
plt.legend(prop={'family':fam, 'size':fs2}, loc='lower left')
fn = 'iou_over_time.png' 
plt.savefig(fn, bbox_inches='tight')



