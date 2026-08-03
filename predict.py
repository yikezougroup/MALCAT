import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from dataset import GNNDataset
from model import MALCAT

def predict(model, dataloader, device):
    """使用模型进行预测"""
    model.eval()
    total_preds = torch.tensor([], device=device)
    
    # 如果有ID信息，也收集起来
    ids = []
    
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            total_preds = torch.cat((total_preds, pred.view(-1)), 0)
            
            # 如果数据中有ID信息，收集起来
            if hasattr(data, 'id'):
                ids.extend(data.id)
    
    return total_preds.cpu().numpy().flatten(), ids

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--dataset', required=True, help='数据集名称')
    parser.add_argument('--model_path', default='save/20260531_160842_peptide/model/best_model.pth', help='模型权重路径')
    parser.add_argument('--data_root', default='data', help='数据根目录')
    parser.add_argument('--output', default='drug_predictions_vaild.csv', help='输出CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载新数据
    fpath = os.path.join(args.data_root, args.dataset)
    try:
        # 假设新数据没有标签，所以设置train=False
        new_dataset = GNNDataset(fpath, train=False)
        print(f"成功加载新数据，样本数量: {len(new_dataset)}")
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return

    # 创建数据加载器
    new_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = MALCAT(
    block_num=3,vocab_drug_size=64,vocab_protein_size=32,filter_num=32,d_model=512,protein_3d_dim=256).to(device)
    
    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功加载模型权重: {args.model_path}")
    except Exception as e:
        print(f"加载模型权重失败: {str(e)}")
        return

    # 进行预测
    print("开始预测...")
    predictions, ids = predict(model, new_loader, device)
    
    # 准备保存结果
    result_df = pd.DataFrame({
        'prediction': predictions
    })
    
    # 如果有ID信息，添加到DataFrame中
    if ids:
        result_df['id'] = ids
    
    # 保存为CSV文件
    try:
        result_df.to_csv(args.output, index=False)
        print(f"预测结果已保存到: {args.output}")
        print(f"共预测了 {len(predictions)} 个样本")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")

if __name__ == '__main__':
    main()