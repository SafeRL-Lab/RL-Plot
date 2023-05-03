

# 导入模块
import numpy as np
import pandas as pd

# path处填入npy文件具体路径
npfile = np.load(r'/home/exp/c_dq_max-3.npy')

# 将npy文件的数据格式转化为csv格式
np_to_csv = pd.DataFrame(data=npfile)

# 存入具体目录下的np_to_csv.csv 文件
np_to_csv.to_csv('/home/exp/c_dq_max-3.csv')
