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
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip35_output_dict.npy" # 0.73 
# fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip36_output_dict.npy" # 0.76
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip41_output_dict.npy"
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip48_output_dict.npy" # end 0.71
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip49_output_dict.npy" # end 0.710
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip50_output_dict.npy" # end 0.707
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip51_output_dict.npy" # end 0.681
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip52_output_dict.npy" # end 0.690
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip53_output_dict.npy" # end 0.671
fn = "01_s100_m256x128x256_F3f_d64_Mf_c1_r.1_Cf_c1_Mrf_p4_f36_k5_taps100i2vce_ns_pip54_output_dict.npy" # end 0.7277

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

# print('ious[2]', ious[2])

# print('%s; end mean iou' % names[ind], np.mean(ious[:,-1]))
print('end mean iou', np.mean(ious[:,-1]))

ious = np.reshape(ious, (-1))
confs = np.reshape(confs, (-1))
# print('ious', ious.shape)

c = np.corrcoef(ious, confs)
# print('c', c)

# print('overall mean iou', np.mean(ious))

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

