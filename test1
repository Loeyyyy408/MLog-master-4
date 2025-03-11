import pandas as pd
import numpy as np
input_file_path_1 = 'D:\\PycharmProjects\\MLog-master4\\data\\update\\BGL\\MLog_log_train.csv'
input_file_path_2 = 'D:\\PycharmProjects\\MLog-master4\\data\\example\\BGL\MLog_log_train.csv'

# Load the CSV files into DataFrames
df1 = pd.read_csv(input_file_path_1)
df2 = pd.read_csv(input_file_path_2)

# Merge the two DataFrames
data = pd.concat([df1, df2], ignore_index=True)

# 提取用于聚类的特征列（假设为 'Sequence' 列）
# 将字符串序列转换为数值特征向量
sequences = data['Sequence'].apply(lambda x: np.fromstring(x, sep=' '))

# 找到最长的序列长度
max_length = sequences.apply(len).max()