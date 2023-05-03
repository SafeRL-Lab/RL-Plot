import numpy as np
import matplotlib.pyplot as plt
# Plots code here for the frozen lake
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
x = np.linspace(0, 10, 150)
# x= len
threshold = 20*np.ones(11)
f = plt.figure()
# Get the cost plots
# /home/shangding/mycode/MO-SafeRL/CMORL-v02-with-momentum/CMORL/cmorl/data/walker2d
STEPS = 500
# strawman_mean = np.load("plot_data/pretrained_reward.npy").mean(axis=-1)[:STEPS]
# strawman_std = np.load("plot_data/pretrained_reward.npy").std(axis=-1)[:STEPS]
#
# metasrl_mean = np.load("plot_data/meta_reward.npy").mean(axis=-1)[:STEPS]
# metasrl_std = np.load("plot_data/meta_reward.npy").std(axis=-1)[:STEPS]
# randominit_mean = np.load("plot_data/random_reward.npy").mean(axis=-1)[:STEPS]
# randominit_std = np.load("plot_data/random_reward.npy").std(axis=-1)[:STEPS]
#
# simpleaveraging_mean = np.load("plot_data/simpleaveraging_reward.npy").mean(axis=-1)[:STEPS]
# simpleaveraging_std = np.load("plot_data/simpleaveraging_reward.npy").std(axis=-1)[:STEPS]

reward_mean1_without_momentum = np.load("/home/shangding/mycode/MO-SafeRL/CMORL-v02-without-momentum/CMORL/cmorl/data/walker2d/reward.npy").mean(axis=-1)[:STEPS]
reward_mean2_without_momentum = np.load("/home/shangding/mycode/MO-SafeRL/CMORL-v02-without-momentum/CMORL/cmorl/data/walker2d/reward2.npy").mean(axis=-1)[:STEPS]

reward_mean1_with_momentum = np.load("/home/shangding/mycode/MO-SafeRL/CMORL-v02-with-momentum/CMORL/cmorl/data/walker2d/reward.npy").mean(axis=-1)[:STEPS]
reward_mean2_with_momentum = np.load("/home/shangding/mycode/MO-SafeRL/CMORL-v02-with-momentum/CMORL/cmorl/data/walker2d/reward2.npy").mean(axis=-1)[:STEPS]

# x = np.linspace(0, STEPS)
x = np.arange(0, STEPS, 1)
plt.plot(x, reward_mean1_without_momentum, 'orange', label='Reward_1 (Without momentum)',linewidth = 2)
plt.plot(x, reward_mean2_without_momentum, 'hotpink', label='Reward_2 (Without momentum)',linewidth = 2)

plt.plot(x, reward_mean1_with_momentum, 'orange', label='Reward_1 (With momentum)',linewidth = 2, linestyle='--')
plt.plot(x, reward_mean2_with_momentum, 'hotpink', label='Reward_2 (With momentum)',linewidth = 2, linestyle='--')

# color = ['orange', 'hotpink', 'dodgerblue', 'mediumpurple', 'c', 'cadetblue', 'steelblue', 'mediumslateblue',
             # 'hotpink', 'mediumturquoise']
# print("reward_mean1------:", len(reward_mean1))
# plt.plot(x, simpleaveraging_mean, 'y', label='Simple Averaging',linewidth = 2)
# plt.plot(x, randominit_mean, 'c', label='Random Initialization',linewidth = 2)
# plt.plot(x, threshold, 'k-', color='b', linestyle='--')
# plt.legend(loc='upper left')
# plt.legend(loc="upper left", fontsize=14)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='best')
plt.xlabel('Number of Epochs')
# plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Reward')
# plt.yticks(np.linspace(0, 1000, 6))
plt.title('HalfCheetah Performance')
# plt.fill_between(x, metasrl_mean-metasrl_std, metasrl_mean+metasrl_std,
#                  alpha=0.2, color = 'g')
# plt.fill_between(x, strawman_mean-strawman_std, strawman_mean+strawman_std,
#     alpha=0.2, color = 'r')
# plt.fill_between(x, randominit_mean-randominit_std, randominit_mean+randominit_std,
#     alpha=0.2, color = 'c')
# plt.fill_between(x, simpleaveraging_mean-simpleaveraging_std, simpleaveraging_mean+simpleaveraging_std ,
#     alpha=0.2, color = 'y')
# plt.ylim((-100, 0))
# plt.show()
f.savefig("/home/shangding/mycode/MO-SafeRL/CMORL-v02-without-momentum/CMORL/cmorl/plot_figures/figures/CMORL-HalfVheetah=Rewards.pdf", bbox_inches='tight')
plt.show()
