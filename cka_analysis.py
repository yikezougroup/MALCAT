import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader

# 导入你的项目模块（根据实际路径调整）
sys.path.append('.')  # 确保当前目录在路径中
from model_Copy1 import MALCAT
from dataset import GNNDataset  # 假设数据集类如此命名

# ==================== 配置参数（请根据实际情况修改） ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'peptide'          # 与训练时一致
DATA_ROOT = 'data'                # 数据集根目录
MODEL_DIR = 'save/20260320_191432_peptide/model'  # 替换为你的模型保存目录（含best_model.pth）
BATCH_SIZE = 128                   # 应与验证集一致
NUM_SAMPLED_NODES = 1024           # 每个batch采样的节点数（控制内存）
RANDOM_SEED = 42                    # 固定随机种子
MAX_LAYERS = 50                    # 只分析前 N 层（设为 0 或大于总层数则分析全部）
# =======================================================================

def linear_CKA(X, Y):
    """
    计算两组表示之间的线性 Centered Kernel Alignment (CKA)
    X, Y: [n, d] 张量
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    # 使用 Frobenius 范数归一化
    return (K * L).sum() / (torch.norm(K, p='fro') * torch.norm(L, p='fro') + 1e-8)

def compute_layerwise_cka(layer_outputs, sample_size=512):
    """
    计算层间 CKA 矩阵
    layer_outputs: list of [N_i, d_i] 张量（已在 CPU）
    sample_size: 每层使用的节点数（如果 N_i > sample_size 则随机采样）
    返回形状为 (L, L) 的矩阵，其中 L = len(layer_outputs)
    """
    L = len(layer_outputs)          # 修正：之前写成了 L-1，导致少一行一列
    cka_mat = torch.zeros(L, L)

    # 对所有层使用相同的随机节点索引（基于第一层节点数）
    N0 = layer_outputs[0].size(0)
    if N0 > sample_size:
        torch.manual_seed(RANDOM_SEED)
        idx = torch.randperm(N0)[:sample_size]
    else:
        idx = torch.arange(N0)

    for i in range(L):
        Xi = layer_outputs[i][idx].float()
        for j in range(L):
            Xj = layer_outputs[j][idx].float()
            cka_mat[i, j] = linear_CKA(Xi, Xj)

    return cka_mat

def plot_cka_heatmap(cka_mat, title="GraphDenseNet Layer-wise CKA", 
                     save_path=None, format=None, fontsize=24):
    """绘制 CKA 热力图，可选保存为 PDF（矢量格式）"""
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图，并获取 axes 对象
    ax = sns.heatmap(cka_mat.numpy(), cmap='RdBu_r', vmin=0, vmax=1,
                     xticklabels=10, yticklabels=10,
                     cbar_kws={'label': 'CKA'})  # label 稍后单独设置字体大小
    
    # 设置标题和轴标签字体
    plt.title(title, fontsize=fontsize + 2)
    plt.xlabel('Layer Index', fontsize=fontsize)
    plt.ylabel('Layer Index', fontsize=fontsize)
    
    # 设置刻度标签字体
    ax.tick_params(axis='x', labelsize=fontsize - 1, rotation=45)  # 旋转避免重叠
    ax.tick_params(axis='y', labelsize=fontsize - 1)
    
    # 设置颜色条标签字体
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('CKA', fontsize=fontsize)
    
    # 调整布局，防止标签被裁剪
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        if format:
            plt.savefig(save_path, format=format, bbox_inches='tight')
        else:
            plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_feature_variance(layer_outputs, save_path=None, format=None, fontsize=24):
    """绘制各层特征平均方差，支持自定义字体大小"""
    variances = []
    for x in layer_outputs:
        var = x.float().var(dim=0).mean().item()
        variances.append(var)

    plt.figure(figsize=(10, 5))
    plt.plot(variances, marker='o', linestyle='-', color='r')
    
    # 设置标题和轴标签字体
    plt.title('Feature Variance Across Layers', fontsize=fontsize + 2)
    plt.xlabel('Layer Index', fontsize=fontsize)
    plt.ylabel('Average Feature Variance', fontsize=fontsize)
    
    # 设置刻度标签字体
    plt.xticks(fontsize=fontsize - 1)
    plt.yticks(fontsize=fontsize - 1)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # 防止标签被裁剪
    
    if save_path:
        if format:
            plt.savefig(save_path, format=format, bbox_inches='tight')
        else:
            plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def collect_gnn_layer_outputs(model, dataloader, device, max_nodes_per_batch=256):
    """
    遍历数据加载器，收集 ligand_encoder 所有层的输出（节点特征）。
    在每个batch中随机采样 max_nodes_per_batch 个节点，以控制内存。
    返回拼接后的层输出列表。
    """
    model.eval()
    ligand_encoder = model.ligand_encoder

    # 注册 hook
    ligand_encoder.layer_outputs = []
    ligand_encoder._hooks = []

    def hook_fn(module, input, output):
        # output 是 Data 对象，节点特征在 output.x
        ligand_encoder.layer_outputs.append(output.x.detach().cpu())

    for module in ligand_encoder.modules():
        if isinstance(module, torch.nn.Module) and module.__class__.__name__ == 'GraphConvBn':
            handle = module.register_forward_hook(hook_fn)
            ligand_encoder._hooks.append(handle)

    all_layer_outputs = []  # 最终将存储每个层的所有采样节点

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            ligand_encoder.layer_outputs = []  # 清空当前batch的hook记录
            model(batch)  # 前向传播，触发 hook

            # 获取当前 batch 的各层输出
            batch_outputs = ligand_encoder.layer_outputs
            if not batch_outputs:
                continue  # 可能没有GraphConvBn层？但你的模型有

            # 如果这是第一个batch，初始化收集列表
            if not all_layer_outputs:
                all_layer_outputs = [[] for _ in batch_outputs]

            # 对每一层，随机采样固定数量的节点
            for i, out in enumerate(batch_outputs):
                n_nodes = out.size(0)
                if n_nodes > max_nodes_per_batch:
                    torch.manual_seed(RANDOM_SEED)
                    idx = torch.randperm(n_nodes)[:max_nodes_per_batch]
                    sampled = out[idx]
                else:
                    sampled = out
                all_layer_outputs[i].append(sampled)

    # 移除 hook
    for handle in ligand_encoder._hooks:
        handle.remove()
    ligand_encoder._hooks.clear()

    # 拼接所有batch的采样节点
    final_outputs = [torch.cat(lst, dim=0) for lst in all_layer_outputs]
    return final_outputs

def load_model_and_data():
    """加载模型和测试数据加载器"""
    # 1. 初始化模型结构（必须与训练时完全一致）
    model = MALCAT(
        block_num=3,
        vocab_drug_size=64,
        vocab_protein_size=32,
        filter_num=32,
        d_model=512,
        protein_3d_dim=256
    ).to(DEVICE)

    # 2. 加载训练好的权重（优先加载 best_model.pth）
    best_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if os.path.exists(best_path):
        state_dict = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded best model from {best_path}")
    else:
        checkpoint_path = os.path.join(MODEL_DIR, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No model found in {MODEL_DIR}")

    # 3. 准备测试集 DataLoader
    data_path = os.path.join(DATA_ROOT, DATASET_NAME)
    full_dataset = GNNDataset(data_path, train=True)  # 假设数据集类有 train 参数
    # 划分测试集（与训练脚本保持一致：随机划分80/20）
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                     generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return model, test_loader

def main():
    print("Loading model and data...")
    model, test_loader = load_model_and_data()

    print("Collecting layer outputs from ligand_encoder...")
    layer_outputs = collect_gnn_layer_outputs(model, test_loader, DEVICE,
                                               max_nodes_per_batch=256)
    original_layer_count = len(layer_outputs)
    print(f"Collected {original_layer_count} layers total.")
    
    # 只保留前 MAX_LAYERS 层（若 MAX_LAYERS > 0 且小于总层数）
    if MAX_LAYERS > 0 and original_layer_count > MAX_LAYERS:
        layer_outputs = layer_outputs[:MAX_LAYERS]
        print(f"Using only the first {MAX_LAYERS} layers for analysis.")
    else:
        print(f"Using all {original_layer_count} layers (MAX_LAYERS={MAX_LAYERS} > total).")
    
    # 打印各层形状
    for i, out in enumerate(layer_outputs):
        print(f"Layer {i}: shape {out.shape}")

    print("Computing CKA matrix...")
    cka_mat = compute_layerwise_cka(layer_outputs, sample_size=512)
    print(f"CKA matrix computed, shape: {cka_mat.shape}")

    # 保存结果（显式指定 format='pdf' 确保输出 PDF）
    os.makedirs('cka_results', exist_ok=True)
    plot_cka_heatmap(cka_mat,
                     title=f"GraphDenseNet Layer-wise CKA ({DATASET_NAME})",
                     save_path='cka_results/heatmap.pdf',
                     format='pdf')               # 新增：明确指定 PDF 格式
    plot_feature_variance(layer_outputs,
                          save_path='cka_results/feature_variance.pdf',
                          format='pdf')          # 新增：明确指定 PDF 格式
    np.save('cka_results/cka_matrix.npy', cka_mat.numpy())
    print("Results saved to cka_results/")

    # 打印 CKA 矩阵的关键统计
    print("\nCKA statistics:")
    print(f"Mean off-diagonal CKA: {cka_mat[~torch.eye(cka_mat.size(0), dtype=bool)].mean():.4f}")
    print(f"Std off-diagonal CKA: {cka_mat[~torch.eye(cka_mat.size(0), dtype=bool)].std():.4f}")
    print(f"Last layer self-similarity: {cka_mat[-1, -1]:.4f}")
    print(f"First vs last layer CKA: {cka_mat[0, -1]:.4f}")

if __name__ == '__main__':
    main()