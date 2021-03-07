import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from scipy.misc import imsave, imresize
from imageio import imsave

# o = np.load('01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1tce_ns_tul06_output_dict.npy', allow_pickle=True)
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1vce_ns_tul08_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1vce_ns_tul09_output_dict.npy" # use softmax
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1tce_ns_tul07_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_faks30i1t_faks30i1t_faks30i1vce_ns_tul07_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_tags90i1vce_ns_share87_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_tags90i1vce_ns_share88_output_dict.npy"
# fn = "01_s90_m256x128x256_F3_d64_M_c1_r.1_tags90i1vce_ns_share89_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test02_output_dict.npy"
# fn = "01_s90_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test03_output_dict.npy"
# fn = "01_s90_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test05_output_dict.npy"
# fn = "01_s90_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test06_output_dict.npy"
# fn = "01_s90_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test07_output_dict.npy"
# fn = "01_s90_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test07_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test10_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test11_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test12_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test12_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test12_output_dict.npy"
# fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test13_output_dict.npy"
# corr is 0.77! ok that's nice. 
fn = "01_s30_m256x128x256_F3_d64_M_c1_r.1_C_c1_tags90i1vce_ns_test14_output_dict.npy"
# corr is now 0.87289196
fn = "01_s60_m256x128x256_F3_d64_M_c1_r.1_C_c1_tans60i2t_ns_test15_output_dict.npy"
fn = "01_s90_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps90i2t_ns_test16_output_dict.npy"
fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test25_output_dict.npy"
fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test34_output_dict.npy"
# fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test60_output_dict.npy"
# fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test62_output_dict.npy"
# fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test63_output_dict.npy" # 0.48
# fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test64_output_dict.npy" # 0.64
# fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test67_output_dict.npy" # 0.71
fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test69_output_dict.npy" # 0.60
# fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test81_output_dict.npy" # 0.71
# fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test82_output_dict.npy" # 0.71
fn = "01_s100_m256x128x256_F3_d64_M_c1_r.1_C_c1_taps100i2t_ns_test84_output_dict.npy" # 0.71

o = np.load(fn, allow_pickle=True)
o = o.item()
# plt.plot(sizes, new_ious, 'o', markersize=8, color='w')

# o['all_ious'] = o['all_ious'][:,5:]
# # o['all_confs'] = o['all_min_confs'][:,5:]
# o['all_confs'] = o['all_confs'][:,5:]

# diff = o['all_confs'] - o['all_min_confs']
# o['all_confs'] = o['all_confs'] - diff*5


# o['all_confs'] = o['all_min_confs']



names = ['mean ious',
         'min ious',
         'max ious',
]
colors = ['indianred',
          'seagreen',
          'seagreen',
]
linestyles = ['o-',
              'o-',
              'o-',
]
fs1 = 18 # axis headers
fs2 = 18 # legend, ticks
fam = 'Times New Roman'


ious = o['all_ious']
confs = o['all_confs']
# these are N x S

# i don't care too much about time right now, so let's just flatten

ious = np.reshape(ious, (-1))
confs = np.reshape(confs, (-1))
print('ious', ious.shape)

c = np.corrcoef(ious, confs)
print('c', c)

print('overall mean iou', np.mean(ious))

inds = np.argsort(confs)
ious = ious[inds]
confs = confs[inds]

# inds = np.argsort(ious)
# ious = ious[inds]
# confs = confs[inds]

# sizes = list(range(1,31))

# plt.rcParams["font.family"] = "Times New Roman"
# fig = plt.figure()
# _, ax = plt.subplots()
# # _, ax = plt.subplot(2,2,1)
# # o_line, = plt.step(recall, precision, color=colors[num], linestyle=linestyles[num], linewidth=3, alpha=1.0, where='post')

# o_line, = plt.plot(confs, ious, '-', linewidth=2, color=colors[0])
# o_line, = plt.plot(confs, ious, '.', linewidth=2, color=colors[0])
o_line, = plt.plot(confs, ious, '.', linewidth=2)
o_line.set_label(names[0])
# plt.plot(sizes, new_ious, 'o', markersize=8, color='w')
# o_line, = plt.plot(sizes, new_ious, 'o', markersize=6, color=colors[0])

# o_line, = plt.plot(sizes, old_ious, '-', linewidth=2, color=colors[1])
# o_line.set_label(names[1])
# # plt.plot(sizes, old_ious, 'o', markersize=8, color='w')
# # o_line, = plt.plot(sizes, old_ious, 'o', markersize=6, color=colors[1])

# # plt.ylim([0.0, 0.7])
# # plt.plot(sizes, fc_ious, color=colors[0], linestyle=linestyles[0], linewidth=3, alpha=1.0, where='post')
# # o_line.set_label(names[0])

plt.xlabel('Confidence', fontsize=fs1, family=fam)
plt.ylabel('IOU', fontsize=fs1, family=fam)
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])

# plt.xticks(fontsize=fs2, family=fam)
# plt.yticks(fontsize=fs2, family=fam)
# plt.grid(True)
plt.grid(True,which="minor",ls="-", color='0.9')
plt.grid(True,which="major",ls="-", color='0.75')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# # plt.xscale('log')
# # # plt.legend(prop={'family':fam, 'size':fs2}, loc='upper right')
# plt.legend(prop={'family':fam, 'size':fs2}, loc='upper right')

# # plt.title('Precision-Recall')
# # plt.show()

fn = 'iou_over_conf.png' 
plt.savefig(fn, bbox_inches='tight')
# # plt.savefig(fn, bbox_inches='tight')


plt.clf()

# next i want to see conf over time
ious = o['all_ious']
confs = o['all_confs']
# these are N x S
N, S = list(confs.shape)

print('N, S', N, S)

time = list(range(S))
print(time[2:])

if True:
    for n in list(range(N)):
        # o_line, = plt.plot(time, confs[n], '.', linewidth=2, color=colors[0])
        # o_line, = plt.plot(time, confs[n], '.', linewidth=2, color=colors[0])
        o_line, = plt.plot(time, confs[n], '-', linewidth=1)#, color=colors[0])
        # o_line, = plt.plot(time[1:], confs[n,1:], '.', linewidth=2, color=colors[0])
        # o_line, = plt.plot(time[2:], confs[n,2:], '.', linewidth=2, color=colors[0])
        # o_line, = plt.plot(time[5:], confs[n,5:], '.', linewidth=2, color=colors[0])
    # # o_line.set_label(names[0])
    # o_line, = plt.plot(time, np.mean(confs, axis=0), '--', linewidth=5, color='red')#, color=colors[0])
else:
    o_line,_,_ = plt.errorbar(time, np.mean(confs, axis=0), np.std(ious, axis=0), barsabove=True)#, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False,
plt.xlabel('Time', fontsize=fs1, family=fam)
plt.ylabel('Confidence', fontsize=fs1, family=fam)
plt.ylim([0.0, 1.0])
# plt.xlim([0.0, 1.0])
# plt.ylim([0.035, 0.045])
plt.grid(True,which="minor",ls="-", color='0.9')
plt.grid(True,which="major",ls="-", color='0.75')
fn = 'conf_over_time.png' 
plt.savefig(fn, bbox_inches='tight')
# # plt.savefig(fn, bbox_inches='tight')



plt.clf()
# next i want to see iou over time
ious = o['all_ious']
confs = o['all_confs']
# these are N x S
N, S = list(confs.shape)
print('N, S', N, S)
time = list(range(S))

if True:
    for n in list(range(N)):
        # o_line, = plt.plot(time, ious[n], '.', linewidth=2, color=colors[0])
        o_line, = plt.plot(time, ious[n], '-', linewidth=1)#, color=colors[0])
    # o_line, = plt.plot(time, np.mean(ious, axis=0), '--', linewidth=5, color='red')#, color=colors[0])
else:
    o_line,_,_ = plt.errorbar(time, np.mean(ious, axis=0), np.std(ious, axis=0), barsabove=True)#, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False,
    
plt.xlabel('Time', fontsize=fs1, family=fam)
plt.ylabel('IOU', fontsize=fs1, family=fam)
plt.ylim([0.0, 1.0])
# plt.xlim([0.0, 1.0])
plt.grid(True,which="minor",ls="-", color='0.9')
plt.grid(True,which="major",ls="-", color='0.75')
fn = 'iou_over_time.png' 
plt.savefig(fn, bbox_inches='tight')









plot_errors = False
if plot_errors:
    plt.clf()
    N = len(confs)
    mean_ious = np.zeros(N)
    min_ious = np.zeros(N)
    max_ious = np.zeros(N)
    stds = np.zeros(N)
    mean_confs = np.zeros(N)
    for ind in list(range(N)):
        # mean_ious[ind] = np.mean(ious[ind:])
        mean_confs[ind] = confs[ind]# np.mean(confs[ind:])
        mean_ious[ind] = np.mean(ious[confs >= confs[ind]])
        # min_ious[ind] = np.min(ious[confs >= confs[ind]])
        # max_ious[ind] = np.max(ious[confs >= confs[ind]])
        std = np.std(ious[confs >= confs[ind]])
        # print('std', std)
        min_ious[ind] = np.clip(mean_ious[ind] - std, a_min=0.0, a_max=1.0)
        max_ious[ind] = np.clip(mean_ious[ind] + std, a_min=1.0, a_max=1.0)
        stds[ind] = std
        # max_ious[ind] = np.max(ious[confs >= confs[ind]])

    o_line,_,_ = plt.errorbar(mean_confs, mean_ious, yerr=stds, barsabove=True)#, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, *, data=None, **kwargs)
    # o_line, = plt.plot(mean_confs, mean_ious, '.', linewidth=2, color=colors[0])
    # o_line.set_label(names[0])
    # o_line, = plt.plot(mean_confs, min_ious, '.', linewidth=2, color=colors[1])
    # o_line.set_label(names[0])
    # o_line, = plt.plot(mean_confs, max_ious, '.', linewidth=2, color=colors[2])
    # o_line.set_label(names[0])

    plt.xlabel('Confidence', fontsize=fs1, family=fam)
    plt.ylabel('IOU', fontsize=fs1, family=fam)
    plt.ylim([0.0, 1.0])
    # plt.xlim([0.0, 11])
    # plt.xticks(fontsize=fs2, family=fam)
    # plt.yticks(fontsize=fs2, family=fam)
    # plt.grid(True)
    plt.grid(True,which="minor",ls="-", color='0.9')
    plt.grid(True,which="major",ls="-", color='0.75')

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # # plt.xscale('log')
    # # # plt.legend(prop={'family':fam, 'size':fs2}, loc='upper right')
    # plt.legend(prop={'family':fam, 'size':fs2}, loc='upper right')

    # # plt.title('Precision-Recall')
    # # plt.show()

    fn = 'iou_over_conf.png' 
    plt.savefig(fn, bbox_inches='tight')
    # # fn = 'iou_over_time.pdf' 
    # # plt.savefig(fn, bbox_inches='tight')

