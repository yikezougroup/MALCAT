import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 读取数据
df = pd.read_csv('data2.csv')  # 替换为你的文件路径

# 按靶标蛋白分组，确保同一蛋白不会同时出现在训练集和测试集
groups = df['target_sequence']  # 以靶标蛋白序列作为分组依据

# 初始化分组分割器（测试集比例设为20%）
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

# 执行分割
train_idx, test_idx = next(splitter.split(df, groups=groups))

# 获取训练集和测试集
train_set = df.iloc[train_idx]
test_set = df.iloc[test_idx]

# 验证结果
print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")
print(f"训练集蛋白数量: {train_set['target_sequence'].nunique()}")
print(f"测试集蛋白数量: {test_set['target_sequence'].nunique()}")

# 检查是否有重叠蛋白
common_proteins = set(train_set['target_sequence']).intersection(set(test_set['target_sequence']))
print(f"重叠蛋白数量: {len(common_proteins)}")  # 应该为0

# 保存分割结果（可选）
train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)