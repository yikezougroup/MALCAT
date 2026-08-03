import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def cold_start_split(df, test_size=0.2, random_state=42):
    """
    按照环肽和靶标蛋白均冷启动的方式分割数据集
    
    参数:
    df: DataFrame，包含四列数据
    test_size: 测试集比例
    random_state: 随机种子
    
    返回:
    train_df: 训练集DataFrame
    test_df: 测试集DataFrame
    """
    # 获取唯一的环肽和靶标蛋白
    unique_compounds = df['compound_iso_smiles'].unique()
    unique_targets = df['target_sequence'].unique()
    
    # 分割环肽
    train_compounds, test_compounds = train_test_split(
        unique_compounds, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 分割靶标蛋白
    train_targets, test_targets = train_test_split(
        unique_targets, 
        test_size=test_size, 
        random_state=random_state+1  # 使用不同的随机种子
    )
    
    # 创建训练集：环肽和靶标蛋白都在训练集中的样本
    train_mask = df['compound_iso_smiles'].isin(train_compounds) & \
                 df['target_sequence'].isin(train_targets)
    train_df = df[train_mask].copy()
    
    # 创建测试集：环肽和靶标蛋白都在测试集中的样本
    test_mask = df['compound_iso_smiles'].isin(test_compounds) & \
                df['target_sequence'].isin(test_targets)
    test_df = df[test_mask].copy()
    
    # 处理既不在训练集也不在测试集的样本
    # 这里可以选择丢弃这些样本或将其分配到训练集
    remaining_mask = ~(train_mask | test_mask)
    remaining_df = df[remaining_mask]
    
    # 将剩余样本添加到训练集（可选，也可以选择丢弃）
    train_df = pd.concat([train_df, remaining_df], ignore_index=True)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"训练集环肽数量: {train_df['compound_iso_smiles'].nunique()}")
    print(f"测试集环肽数量: {test_df['compound_iso_smiles'].nunique()}")
    print(f"训练集靶标蛋白数量: {train_df['target_sequence'].nunique()}")
    print(f"测试集靶标蛋白数量: {test_df['target_sequence'].nunique()}")
    
    return train_df, test_df

# 示例使用
if __name__ == "__main__":
    # 读取数据
    # 假设数据保存在CSV文件中
    df = pd.read_csv('data2.csv')
    
    # 执行冷启动分割
    train_df, test_df = cold_start_split(df, test_size=0.2, random_state=42)
    
    # 保存分割后的数据集
    train_df.to_csv('data_train.csv', index=False)
    test_df.to_csv('data_test.csv', index=False)