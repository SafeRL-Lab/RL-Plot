# import numpy as np
# test=np.load('/home/shangding/mycode/Darm-safe-RL/rl_on_manifold/logs/collision_avoidance/exp/R-1.npy',encoding = "latin1")  #加载文件
# doc = open('R-1.csv', 'a')  #打开一个存储文件，并依次写入
# print(test, file=doc)  #将打印内容写入文件中

# 导入模块
import numpy as np
import pandas as pd

# path处填入npy文件具体路径
npfile = np.load(r'/home/andrew/MyCode20201108/Darm_Safe-RL/run_experiments_data/send_data/rl_on_manifold_safety_to_revise_reward/logs/circular_motion/exp/c_dq_max-3.npy')

# 将npy文件的数据格式转化为csv格式
np_to_csv = pd.DataFrame(data=npfile)

# 存入具体目录下的np_to_csv.csv 文件
np_to_csv.to_csv('/home/andrew/MyCode20201108/Darm_Safe-RL/plot_manifold_figures/rl_on_manifold_safety_to_revise_reward/c_dq_max-3.csv')
