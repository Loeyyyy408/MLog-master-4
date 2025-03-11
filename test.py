import pandas as pd

# # 读取训练和测试数据
# train_data = pd.read_csv('data/BGL/MLog_log_train.csv')
# test_data = pd.read_csv('data/BGL/MLog_log_test.csv')
#
# # 合并训练和测试数据
# combined_data = pd.concat([train_data, test_data], ignore_index=True)
#
# # 根据 label 列的值进行筛选
# normal_data = combined_data[combined_data['label'] == 0]
# abnormal_data = combined_data[combined_data['label'] == 1]
#
# # 将结果保存到新的 CSV 文件
# normal_data.to_csv('total_normal.csv', index=False)
# abnormal_data.to_csv('total_abnormal.csv', index=False)
#
# print("合并完成，生成 total_normal.csv 和 total_abnormal.csv 文件。")


# # 读取数据
# normal_data = pd.read_csv('total_normal.csv')
# abnormal_data = pd.read_csv('total_abnormal.csv')

# # 计算划分的索引
# normal_split_index = int(len(normal_data) * 0.8)  # 80% 的索引
# abnormal_split_index = int(len(abnormal_data) * 0.8)  # 80% 的索引
#
# # 按照 4:1 的比例划分正常数据集
# old_normal = normal_data.iloc[:normal_split_index]
# update_normal = normal_data.iloc[normal_split_index:]
#
# # 按照 4:1 的比例划分异常数据集
# old_abnormal = abnormal_data.iloc[:abnormal_split_index]
# update_abnormal = abnormal_data.iloc[abnormal_split_index:]
#
# # 保存结果到新的 CSV 文件
# old_normal.to_csv('old_normal.csv', index=False)
# old_abnormal.to_csv('old_abnormal.csv', index=False)
# update_normal.to_csv('update_normal.csv', index=False)
# update_abnormal.to_csv('update_abnormal.csv', index=False)
#
# print("数据划分完成，生成 old_normal.csv, old_abnormal.csv, update_normal.csv, update_abnormal.csv 文件。")

# # 读取数据
# old_normal = pd.read_csv('update_normal.csv')
# old_abnormal = pd.read_csv('update_abnormal.csv')
#
# # 提取前 5000 条正常数据和前 1000 条异常数据
# train_normal = old_normal.iloc[:1000]
# valid_abnormal = old_abnormal.iloc[:200]
#
# # 生成训练集和验证集
# MLog_log_train = pd.concat([train_normal, valid_abnormal], ignore_index=True)
# MLog_log_valid = valid_abnormal
#
# # 剩余数据生成测试集
# test_normal = old_normal.iloc[1000:]  # 剩余正常数据
# test_abnormal = old_abnormal.iloc[200:]  # 剩余异常数据
#
# # 合并测试集
# MLog_log_test = pd.concat([test_normal, test_abnormal], ignore_index=True)
#
# # 保存结果到新的 CSV 文件
# MLog_log_train.to_csv('data/update/BGL/MLog_log_train.csv', index=False)
# MLog_log_valid.to_csv('data/update/BGL/MLog_log_valid.csv', index=False)
# MLog_log_test.to_csv('data/update/BGL/MLog_log_test.csv', index=False)
#
# print("数据集生成完成，生成 MLog_log_train.csv, MLog_log_valid.csv, MLog_log_test.csv 文件。")

# 读取数据
old_normal = pd.read_csv('old_normal.csv')
old_abnormal = pd.read_csv('old_abnormal.csv')

# 随机抽取 500 条正常数据和 1000 条异常数据
train_normal = old_normal.sample(n=500, random_state=None)  # 不设置随机种子
train_abnormal = old_abnormal.sample(n=1000, random_state=None)  # 不设置随机种子

# 合并数据
MLog_log_train = pd.concat([train_normal, train_abnormal], ignore_index=True)

# 保存结果到新的 CSV 文件
MLog_log_train.to_csv('MLog_log_train.csv', index=False)

print("数据集生成完成，生成 MLog_log_train.csv 文件。")


