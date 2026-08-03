import argparse
import os
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from log.train_logger import TrainLogger
from dataset import *
from model import MALCAT
from untils import *
from mestic import *


def val(model, criterion, dataloader, device):
    model.eval()
    total_preds = torch.tensor([], device=device) 
    total_labels = torch.tensor([], device=device)
    run_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            run_loss.update(loss.item(), data.y.size(0))
            total_preds = torch.cat((total_preds, pred.view(-1)), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1)), 0)

    epoch_loss = run_loss.get_average()
    run_loss.reset()
    return epoch_loss, total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()


def check_write_permission(directory):
    """检查目录是否可写"""
    try:
        # 创建临时文件测试写入权限
        test_file = os.path.join(directory, "write_test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"目录 {directory} 不可写: {str(e)}")
        return False


def get_alternative_save_dir():
    """获取备选保存目录"""
    # 尝试用户主目录
    home_dir = os.path.expanduser("~")
    alt_dir = os.path.join(home_dir, "model_backup")
    os.makedirs(alt_dir, exist_ok=True)
    if check_write_permission(alt_dir):
        return alt_dir
    
    # 尝试当前工作目录
    cwd_dir = os.getcwd()
    if check_write_permission(cwd_dir):
        return cwd_dir
        
    # 尝试/tmp临时目录
    tmp_dir = "/tmp/model_backup"
    os.makedirs(tmp_dir, exist_ok=True)
    if check_write_permission(tmp_dir):
        return tmp_dir
        
    return None


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_test_loss, model_dir):
    """保存断点续训检查点，带错误处理和备选路径"""
    try:
        # 检查主目录是否可写
        if not check_write_permission(model_dir):
            alt_dir = get_alternative_save_dir()
            if alt_dir:
                print(f"使用备选目录保存检查点: {alt_dir}")
                model_dir = alt_dir
            else:
                print("所有目录都不可写，无法保存检查点!")
                return False

        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
        
        # 尝试分步骤保存，减少单次写入压力
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_test_loss": best_test_loss
        }
        
        # 先保存到临时文件，成功后再重命名，避免文件损坏
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)
        
        print(f"检查点保存成功: {checkpoint_path}")
        return True
    except Exception as e:
        print(f"检查点保存失败: {str(e)}")
        return False


def save_best_model(model, model_dir, epoch, test_loss):
    """保存最佳模型，带错误处理和备选路径"""
    try:
        # 检查主目录是否可写
        if not check_write_permission(model_dir):
            alt_dir = get_alternative_save_dir()
            if alt_dir:
                print(f"使用备选目录保存最佳模型: {alt_dir}")
                model_dir = alt_dir
            else:
                print("所有目录都不可写，无法保存最佳模型!")
                return False

        os.makedirs(model_dir, exist_ok=True)
        best_model_path = os.path.join(model_dir, "best_model.pth")
        
        # 同样使用临时文件中转
        temp_path = best_model_path + ".tmp"
        torch.save(model.state_dict(), temp_path)
        os.rename(temp_path, best_model_path)
        
        print(f"最佳模型更新于 epoch {epoch}，损失: {test_loss:.4f}")
        return True
    except Exception as e:
        print(f"最佳模型保存失败: {str(e)}")
        return False


def load_checkpoint(model, optimizer, scheduler, model_dir):
    """加载检查点，支持从备选目录加载"""
    # 先检查主目录
    checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        # 检查备选目录
        alt_dir = get_alternative_save_dir()
        if alt_dir:
            checkpoint_path = os.path.join(alt_dir, "checkpoint.pth")
            if not os.path.exists(checkpoint_path):
                print("未找到检查点，从头开始训练")
                return 0, 0, float('inf')
        else:
            print("未找到检查点，从头开始训练")
            return 0, 0, float('inf')

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_test_loss = checkpoint["best_test_loss"]
        print(f"从检查点恢复: epoch {epoch}, step {global_step}, 最佳损失 {best_test_loss:.4f}")
        return epoch, global_step, best_test_loss
    except Exception as e:
        print(f"检查点加载失败: {str(e)}，将从头开始训练")
        return 0, 0, float('inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='peptide')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    params = dict(
        data_root='data',
        save_dir='save',
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )
    logger = TrainLogger(params)
    logger.info(__file__)

    # 数据加载
    DATASET = params.get('dataset')
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)
    total_set = GNNDataset(fpath, train=True)

    # 划分数据集
    train_size = int(0.8 * len(total_set))
    test_size = len(total_set) - train_size
    train_set, test_set = torch.utils.data.random_split(total_set, [train_size, test_size])
    print(f"训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # 设备与模型初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MALCAT(
    block_num=3,vocab_drug_size=64,vocab_protein_size=32,filter_num=32,d_model=512,protein_3d_dim=256).to(device)

    # 优化器、调度器、损失函数
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()

    # 训练参数初始化（支持断点续训）
    epochs = 600
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

    # 从检查点恢复状态（若存在）
    model_dir = logger.get_model_dir()
    global_epoch, global_step, best_test_loss = load_checkpoint(model, optimizer, scheduler, model_dir)

    # 初始化指标记录器
    run_loss = AverageMeter()
    run_ci = AverageMeter()

    model.train()
    print(f"开始训练（总epochs: {epochs}）...")

    for i in range(num_iter):
        if global_epoch >= epochs:
            break
            
        for data in train_loader:
            global_step += 1
            data = data.to(device)
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            ci = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新训练指标
            run_loss.update(loss.item(), data.y.size(0))
            run_ci.update(ci, data.y.size(0))

            # 每个epoch结束后处理
            if global_step % steps_per_epoch == 0:
                global_epoch += 1
                if global_epoch > epochs:
                    break

                epoch_loss = run_loss.get_average()
                epoch_ci = run_ci.get_average()
                run_loss.reset()
                run_ci.reset()

                # 验证模型
                test_loss, G, P = val(model, criterion, test_loader, device)
                model.train()

                # 记录指标
                msg = (f"train_loss:%.4f, train_ci:%.4f, test_loss:%.4f, test_ci:%.4f, "
                       f"test_pearson1:%.4f, test_Rm2:%.4f" % (
                           epoch_loss, epoch_ci, test_loss, get_cindex(G, P), 
                           pearson1(G, P), spearman1(G, P)
                       ))
                logger.info(msg)

                # 保存测试结果（使用追加模式，避免丢失）
                test_result_file = f'result/test_result_BS1024_{DATASET}.csv'
                try:
                    os.makedirs(os.path.dirname(test_result_file), exist_ok=True)
                    ret = [global_epoch, test_loss, pearson1(G, P), get_cindex(G, P), get_rm2(G, P), spearman1(G, P)]
                    with open(test_result_file, 'a') as f:
                        f.write(','.join(map(str, ret)) + '\n')
                except Exception as e:
                    print(f"保存测试结果失败: {str(e)}")

                # 保存最佳模型（仅在loss更优时）
                if save_model and test_loss < best_test_loss:
                    best_test_loss = test_loss
                    save_best_model(model, model_dir, global_epoch, test_loss)

                # 保存检查点（每个epoch都保存）
                if save_model:
                    save_checkpoint(model, optimizer, scheduler, global_epoch, global_step, best_test_loss, model_dir)

        # 更新学习率
        scheduler.step()

    print(f"训练完成。最佳测试损失: {best_test_loss:.4f}")


if __name__ == '__main__':
    main()
