import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def cold_start_split_by_peptide(data_path, test_size=0.2, random_state=42):
    """
    环肽冷启动分割数据集
    
    参数:
    data_path: 数据文件路径
    test_size: 测试集比例，默认0.2
    random_state: 随机种子，默认42
    
    返回:
    train_df: 训练集DataFrame
    test_df: 测试集DataFrame
    """
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 确保数据包含所需列
    required_columns = ['compound_iso_smiles', 'target_sequence', 'affinity', 'protein_3d_path']
    assert all(col in df.columns for col in required_columns), "数据缺少必要列"
    
    # 使用GroupShuffleSplit确保测试集中的环肽不在训练集中出现
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    
    # 按环肽分组进行分割
    for train_idx, test_idx in splitter.split(df, groups=df['compound_iso_smiles']):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
    
    # 验证分割结果 - 确保测试集中的环肽不在训练集中
    train_peptides = set(train_df['compound_iso_smiles'].unique())
    test_peptides = set(test_df['compound_iso_smiles'].unique())
    
    assert len(train_peptides.intersection(test_peptides)) == 0, \
        "分割错误: 训练集和测试集中有重叠的环肽"
    
    print(f"数据集分割完成:")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"训练集环肽数量: {len(train_peptides)}")
    print(f"测试集环肽数量: {len(test_peptides)}")
    
    return train_df, test_df

# 使用示例
if __name__ == "__main__":
    # 假设数据文件名为 'peptide_protein_data.csv'
    data_file = 'data2.csv'
    
    try:
        train_data, test_data = cold_start_split_by_peptide(data_file, test_size=0.2)
        
        # 保存分割后的数据集
        train_data.to_csv('train_set.csv', index=False)
        test_data.to_csv('test_set.csv', index=False)
        print("数据集已保存为 'train_set.csv' 和 'test_set.csv'")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")