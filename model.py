import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch.nn import Sequential, Linear, ReLU
import numpy as np

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


from se3andmaif import advanced_extract_protein_features

os.environ["TOKENIZERS_PARALLELISM"] = "false"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 1. 原始基础组件（保持不变） ====================

class NodeLevelBatchNorm(_BatchNorm):
    r"""节点级批标准化层"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gelu = nn.ReLU()
        nn1 = Sequential(Linear(in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))
        self.conv = gnn.GINConv(nn1)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        data.x = self.gelu(self.norm(self.conv(x, edge_index)))
        return data


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features
        data = self.conv1(data)
        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]
        data = self.bn_function(data)
        data = self.conv2(data)
        return data


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features
        data.x = torch.cat(data.x, 1)
        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(15, 25, 20, 30), bn_sizes=[10, 5, 10, 5]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = global_mean_pool(data.x, data.batch)
        x = self.classifer(x)
        return x


class Protein3DFeatureExtractor(nn.Module):
    """蛋白质3D结构特征提取模块"""
    def __init__(self, input_dim=512, output_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, protein_3d_paths, device):
        features_list = []
        for path in protein_3d_paths:
            if not os.path.exists(path):
                feat = torch.zeros(512, device=device)
            else:
                feat_np = advanced_extract_protein_features(
                    cif_path=path,
                    device='cuda',
                    output_dim=512,
                    visualize=False,
                    verbose=False
                )
                if feat_np is None:
                    feat = torch.zeros(512, device=device)
                else:
                    feat = torch.from_numpy(feat_np).to(device)
            features_list.append(feat)

        batch_feat = torch.stack(features_list)
        return self.proj(batch_feat)

# ==================== 2. 因果不变性学习核心组件 ====================

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层 - 用于对抗训练"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DisentangledFeatureProjector(nn.Module):
    """解缠结特征投影器 - 分离共享和特有特征"""
    def __init__(self, input_dim, shared_dim=128, specific_dim=64, d_model=512):
        super().__init__()
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.d_model = d_model
        
        # 共享特征提取头
        self.shared_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, shared_dim),
            nn.LayerNorm(shared_dim)
        )
        
        # 特有特征提取头
        self.specific_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, specific_dim),
            nn.LayerNorm(specific_dim)
        )
        
        # 模态判别器（用于对抗训练）
        self.modality_discriminator = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4个模态：SMILES、蛋白序列、蛋白3D结构、分子图
            nn.LogSoftmax(dim=-1)
        )
        
        # 特征重构头（确保信息不丢失）
        self.reconstructor = nn.Sequential(
            nn.Linear(shared_dim + specific_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
        # 共享特征投影到d_model维度（用于Transformer）
        self.shared_to_dmodel = nn.Linear(shared_dim, d_model) if shared_dim != d_model else nn.Identity()
        
    def forward(self, x, modality_id=None, compute_recon=True):
        """
        输入特征解缠结
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            modality_id: 模态ID (0:SMILES, 1:蛋白序列, 2:蛋白3D结构, 3:分子图)
            compute_recon: 是否计算重构损失
        """
        # 解缠结
        shared_feat = self.shared_head(x)  # [batch_size, shared_dim]
        specific_feat = self.specific_head(x)  # [batch_size, specific_dim]
        
        # 将共享特征投影到d_model维度
        shared_feat_dmodel = self.shared_to_dmodel(shared_feat)  # [batch_size, d_model]
        
        # 重构损失
        recon_loss = None
        if compute_recon:
            reconstructed = self.reconstructor(torch.cat([shared_feat, specific_feat], dim=-1))
            recon_loss = F.mse_loss(reconstructed, x)
        
        # 模态判别（对抗训练）
        modality_pred = None
        if modality_id is not None:
            # 应用梯度反转
            shared_rev = GradientReversalFunction.apply(shared_feat, 0.1)
            modality_pred = self.modality_discriminator(shared_rev)
        
        return {
            'shared': shared_feat,  # 原始共享特征 [batch_size, shared_dim]
            'shared_dmodel': shared_feat_dmodel,  # 投影后的共享特征 [batch_size, d_model]
            'specific': specific_feat,  # 特有特征 [batch_size, specific_dim]
            'recon_loss': recon_loss,
            'modality_pred': modality_pred,
            'modality_id': modality_id
        }


class ConditionalCausalAttention(nn.Module):
    """
    条件因果注意力（Conditional Causal Attention, CCA）
    根据环境上下文动态调整跨模态注意力
    """
    def __init__(self, shared_dim=128, num_modalities=4, num_heads=4):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        
        # 跨模态多头注意力
        self.attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, batch_first=True)
        
        # 环境条件编码器
        self.env_encoder = nn.Sequential(
            nn.Linear(shared_dim * num_modalities, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim)
        )
        
        # 输出投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim)
        )
        
        # 不变性验证器
        self.invariance_validator = nn.Sequential(
            nn.Linear(shared_dim * num_modalities, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, shared_features):
        """
        shared_features: list [batch, shared_dim] × num_modalities
        """
        batch_size = shared_features[0].size(0)
        device = shared_features[0].device

        # 堆叠为 [batch, num_modalities, shared_dim]
        shared_stack = torch.stack(shared_features, dim=1)

        # 环境条件向量
        env_cond = self.env_encoder(shared_stack.flatten(start_dim=1))  # [batch, shared_dim]
        env_cond = env_cond.unsqueeze(1).repeat(1, self.num_modalities, 1)  # [batch, num_modalities, shared_dim]

        # 条件化 query/key/value = shared + env_cond
        qkv = shared_stack + env_cond

        # 跨模态注意力
        attended_shared, attn_weights = self.attn(qkv, qkv, qkv)

        # 融合加投影
        fused_shared = self.fusion_proj(attended_shared.sum(dim=1))  # [batch, shared_dim]

        # 不变性评分
        invariance_score = self.invariance_validator(shared_stack.flatten(start_dim=1))

        return {
            'fused_shared': fused_shared,
            'attn_weights': attn_weights,
            'invariance_score': invariance_score,
            'shared_stack': shared_stack
        }


# ==================== 3. 简化版序列特征提取器 ====================

class SimpleSequenceExtractor(nn.Module):
    """
    SMILES 分支由 MoLFormer-XL-both-10pct 替换为本地 LlaSMOL：
      base:    cyclicpeptide/LlaSMOL/Mistral-7B-v0.1
      adapter: cyclicpeptide/LlaSMOL/LlaSMol-Mistral-7B
    Protein 分支由 prot_bert_bfd 替换为本地 ESM-2：
      esm2:    cyclicpeptide/esm2_t36_3B_UR50D
    其余融合层、外部接口和 MALCAT 主体保持不变。
    """
    def __init__(
        self,
        d_model=512,
        llasmol_base_path=None,
        llasmol_adapter_path=None,
        esm2_path=None,
        freeze_llasmol=True,
        freeze_esm2=True
    ):
        super().__init__()

        # ========== SMILES: LlaSMOL / Mistral-7B LoRA ==========
        if PeftModel is None:
            raise ImportError(
                "未检测到 peft。LlaSMOL-Mistral-7B 是 LoRA adapter，"
                "请先安装：pip install peft accelerate"
            )

        # 重要：这里不要写 cyclicpeptide/xxx。
        # 因为训练脚本已经在 cyclicpeptide 目录下运行，继续写 cyclicpeptide/xxx 会变成二级路径。
        if llasmol_base_path is None:
            llasmol_base_path = os.path.join(BASE_DIR, "LlaSMOL", "Mistral-7B-v0.1")
        if llasmol_adapter_path is None:
            llasmol_adapter_path = os.path.join(BASE_DIR, "LlaSMOL", "LlaSMol-Mistral-7B")
        if esm2_path is None:
            esm2_path = os.path.join(BASE_DIR, "esm2_t36_3B_UR50D")

        for model_name, model_path in [
            ("LlaSMOL base", llasmol_base_path),
            ("LlaSMOL adapter", llasmol_adapter_path),
            ("ESM-2", esm2_path),
        ]:
            if not os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"{model_name} 本地路径不存在: {model_path}\n"
                    f"请检查模型文件夹是否放在 {BASE_DIR} 下，或在 SimpleSequenceExtractor 中传入绝对路径。"
                )

        self.freeze_llasmol = freeze_llasmol
        self.freeze_esm2 = freeze_esm2

        self.smiles_tokenizer = AutoTokenizer.from_pretrained(
            llasmol_base_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True
        )
        if self.smiles_tokenizer.pad_token is None:
            self.smiles_tokenizer.pad_token = self.smiles_tokenizer.eos_token
        self.smiles_tokenizer.padding_side = "right"

        llasmol_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        smiles_base_model = AutoModelForCausalLM.from_pretrained(
            llasmol_base_path,
            torch_dtype=llasmol_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            local_files_only=True
        )
        self.smiles_model = PeftModel.from_pretrained(
            smiles_base_model,
            llasmol_adapter_path,
            local_files_only=True
        )
        self.smiles_model.config.pad_token_id = self.smiles_tokenizer.pad_token_id
        self.smiles_model.config.output_hidden_states = True
        self.smiles_model.config.use_cache = False

        if self.freeze_llasmol:
            self.smiles_model.eval()
            for param in self.smiles_model.parameters():
                param.requires_grad = False

        # ========== Protein: 使用本地 ESM-2，替换原来的 prot_bert_bfd ==========
        self.protein_tokenizer = AutoTokenizer.from_pretrained(
            esm2_path,
            trust_remote_code=True,
            local_files_only=True
        )
        esm2_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.protein_model = AutoModel.from_pretrained(
            esm2_path,
            torch_dtype=esm2_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            local_files_only=True
        )

        if self.freeze_esm2:
            self.protein_model.eval()
            for param in self.protein_model.parameters():
                param.requires_grad = False

        smiles_dim = self._get_hidden_size(self.smiles_model)
        protein_dim = self.protein_model.config.hidden_size

        # 特征投影到统一维度
        self.smiles_proj = nn.Linear(smiles_dim, d_model)
        self.protein_proj = nn.Linear(protein_dim, d_model)
        
        # 简单的跨模态融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    @staticmethod
    def _get_hidden_size(model):
        """兼容 PeftModel / HuggingFace base model 的 hidden_size 获取。"""
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            base = model.base_model.model
            if hasattr(base, "config") and hasattr(base.config, "hidden_size"):
                return base.config.hidden_size
        raise AttributeError("无法从 LlaSMOL/Mistral 模型中读取 hidden_size")

    @staticmethod
    def _module_device(module):
        return next(module.parameters()).device

    @staticmethod
    def _masked_mean_pooling(hidden_states, attention_mask):
        """
        对 causal LM 的最后一层 hidden states 做 mask mean pooling，
        输出形状：[batch_size, hidden_size]
        """
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts
        
    def forward(self, smiles_list, protein_list):
        """
        提取序列特征 - 简化版本
        """
        batch_size = len(smiles_list)
        
        # SMILES特征：使用 LlaSMOL/Mistral 的最后一层 hidden states 做 mean pooling
        try:
            smiles_device = self._module_device(self.smiles_model)
            smiles_inputs = self.smiles_tokenizer(
                smiles_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            smiles_input_ids = smiles_inputs['input_ids'].to(smiles_device)
            smiles_attention_mask = smiles_inputs['attention_mask'].to(smiles_device)

            grad_context = torch.no_grad() if self.freeze_llasmol else torch.enable_grad()
            with grad_context:
                smiles_outputs = self.smiles_model(
                    input_ids=smiles_input_ids,
                    attention_mask=smiles_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False
                )
                last_hidden = smiles_outputs.hidden_states[-1]
                smiles_features = self._masked_mean_pooling(last_hidden, smiles_attention_mask)

            smiles_features = smiles_features.to(
                device=self.smiles_proj.weight.device,
                dtype=self.smiles_proj.weight.dtype
            )
            smiles_features = self.smiles_proj(smiles_features)
        except Exception as e:
            print(f"SMILES特征提取失败: {e}")
            smiles_features = torch.randn(
                batch_size,
                self.smiles_proj.out_features,
                device=self.smiles_proj.weight.device,
                dtype=self.smiles_proj.weight.dtype
            )
        
        # 蛋白质序列特征：保持原逻辑不变
        try:
            protein_device = self._module_device(self.protein_model)
            protein_inputs = self.protein_tokenizer(
                protein_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            protein_input_ids = protein_inputs['input_ids'].to(protein_device)
            protein_attention_mask = protein_inputs['attention_mask'].to(protein_device)
            
            protein_grad_context = torch.no_grad() if self.freeze_esm2 else torch.enable_grad()
            with protein_grad_context:
                protein_outputs = self.protein_model(
                    input_ids=protein_input_ids,
                    attention_mask=protein_attention_mask
                )
                protein_features = protein_outputs.last_hidden_state.mean(dim=1)
            protein_features = protein_features.to(
                device=self.protein_proj.weight.device,
                dtype=self.protein_proj.weight.dtype
            )
            protein_features = self.protein_proj(protein_features)
        except Exception as e:
            print(f"蛋白质特征提取失败: {e}")
            protein_features = torch.randn(
                batch_size,
                self.protein_proj.out_features,
                device=self.protein_proj.weight.device,
                dtype=self.protein_proj.weight.dtype
            )
        
        # 简单融合
        combined = torch.cat([smiles_features, protein_features], dim=-1)
        fused = self.fusion(combined)
        
        return fused


# ==================== 4. 完整的因果不变性DeepSGS-DTA模型 ====================
class MALCAT(nn.Module):
    """
    基于因果不变性学习的DeepSGS-DTA模型
    包含四个模态：SMILES序列、蛋白序列、蛋白3D结构、分子图
    """
    
    def __init__(self, block_num, vocab_drug_size, vocab_protein_size, embedding_size=128,
                 filter_num=32, out_dim=1, d_model=512, protein_3d_dim=256):
        super().__init__()
        
        print(f"初始化MALCAT模型 (4模态因果不变性学习)")
        print(f"d_model: {d_model}, protein_3d_dim: {protein_3d_dim}, filter_num: {filter_num}")
        
        # 保存关键参数
        self.d_model = d_model
        self.protein_3d_dim = protein_3d_dim
        self.filter_num = filter_num
        
        # ========== 1. 四个模态的特征提取器 ==========
        
        # a) 序列特征提取器（简化版）
        self.sequence_extractor = SimpleSequenceExtractor(d_model=d_model)
        
        # b) 分子图特征提取器（原始GraphDenseNet）
        self.ligand_encoder = GraphDenseNet(
            num_input_features=47,
            out_dim=filter_num * 3,
            block_config=[3, 3, 3, 3, 3, 3, 3],  # 7个block，各3个DenseLayer
            bn_sizes=[2, 2, 2, 2, 2, 2, 2]      # 7个值，每个block的bn_size=2
        )
        # self.ligand_encoder = GraphDenseNet(
        #     num_input_features=43,
        #     out_dim=filter_num * 3,
        #     block_config=[8, 8, 8],  # 7个block，各3个DenseLayer
        #     bn_sizes=[2, 2, 2]      # 7个值，每个block的bn_size=2
        # )

        # c) 蛋白质3D特征提取器
        self.protein_3d_extractor = Protein3DFeatureExtractor(
            input_dim=512,
            output_dim=protein_3d_dim
        )
        
        # ========== 2. 因果不变性学习组件 ==========
        self.shared_dim = 128
        self.specific_dim = 64
        
        # 计算序列特征的维度（序列提取器的输出是 d_model）
        self.seq_output_dim = d_model
        
        # 四个模态的解缠结投影器
        # 模态0: SMILES序列特征
        self.smiles_disentangler = DisentangledFeatureProjector(
            input_dim=self.seq_output_dim // 2,  # 使用一半的序列特征
            shared_dim=self.shared_dim,
            specific_dim=self.specific_dim,
            d_model=self.seq_output_dim // 2  # 注意：这里不需要投影到d_model
        )
        
        # 模态1: 蛋白序列特征
        self.protein_disentangler = DisentangledFeatureProjector(
            input_dim=self.seq_output_dim // 2,  # 使用另一半的序列特征
            shared_dim=self.shared_dim,
            specific_dim=self.specific_dim,
            d_model=self.seq_output_dim // 2
        )
        
        # 模态2: 蛋白3D结构特征
        self.struct_disentangler = DisentangledFeatureProjector(
            input_dim=protein_3d_dim,
            shared_dim=self.shared_dim,
            specific_dim=self.specific_dim,
            d_model=protein_3d_dim
        )
        
        # 模态3: 分子图特征
        self.graph_disentangler = DisentangledFeatureProjector(
            input_dim=filter_num * 3,  # GraphDenseNet的输出维度
            shared_dim=self.shared_dim,
            specific_dim=self.specific_dim,
            d_model=filter_num * 3
        )
        
        # 不变性特征融合（4模态）
        self.invariant_fusion = ConditionalCausalAttention(
            shared_dim=self.shared_dim,
            num_modalities=4,
            num_heads=4
        )
        
        # ========== 3. 特征对齐层（如果需要统一维度） ==========
        # 将融合后的共享特征和特有特征投影到统一维度
        self.shared_alignment = nn.Sequential(
            nn.Linear(self.shared_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        self.specific_alignment = nn.Sequential(
            nn.Linear(self.specific_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # ========== 4. 分类器 ==========
        # 对齐后的总维度：共享特征256 + 4个特有特征(128*4=512) = 768
        classifier_input_dim = 256 + (128 * 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )
        
        # ========== 5. 损失权重 ==========
        self.alpha_recon = 0.3  # 重构损失权重
        self.alpha_adv = 0.1    # 对抗损失权重
        self.alpha_invariance = 0.05  # 不变性损失权重
        
        # 设备管理
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _process_sequences(self, drug_tensor, target_tensor):
        """处理序列数据 - 更鲁棒的版本"""
        batch_size = len(drug_tensor)
        smiles_list = []
        protein_list = []
        
        # 处理SMILES
        for s in drug_tensor.cpu().numpy():
            s = s[s != 0]  # 去除填充的零
            if len(s) > 0:
                try:
                    # 简化的编码转换
                    s_str = ''.join([chr(65 + (x % 26)) for x in s[:20]])
                except:
                    s_str = "C" * 10  # 默认值
            else:
                s_str = "C" * 10  # 默认值
            smiles_list.append(s_str)
        
        # 处理蛋白质序列
        for p in target_tensor.cpu().numpy():
            p = p[p != 0]
            if len(p) > 0:
                try:
                    p_str = ''.join([chr(65 + (x % 26)) for x in p[:50]])
                except:
                    p_str = "A" * 20  # 默认值
            else:
                p_str = "A" * 20  # 默认值
            protein_list.append(p_str)
        
        return smiles_list, protein_list
    
    def _compute_adversarial_loss(self, modality_pred, modality_id):
        """计算对抗损失"""
        if modality_pred is None or modality_id is None:
            return torch.tensor(0.0, device=self.device)
        
        batch_size = modality_pred.size(0)
        target = torch.full((batch_size,), modality_id, 
                           dtype=torch.long, device=self.device)
        
        return F.cross_entropy(modality_pred, target)
    
    def _compute_invariance_loss(self, shared_features):
        """计算不变性损失（鼓励共享特征相似）"""
        if len(shared_features) < 2:
            return torch.tensor(0.0, device=self.device)
        
        loss = 0
        for i in range(len(shared_features)):
            for j in range(i + 1, len(shared_features)):
                loss += F.mse_loss(shared_features[i], shared_features[j])
        
        return loss / (len(shared_features) * (len(shared_features) - 1) / 2)
    
    def forward(self, data, return_details=False):
        """
        前向传播 - 四个模态的因果不变性学习
        """
        # 确保数据在正确设备上
        data = data.to(self.device)
        
        # ========== 1. 提取四个模态的原始特征 ==========
        
        # a) 分子图特征（模态3）
        graph_features = self.ligand_encoder(data)  # [batch, filter_num*3]
        
        # b) 序列特征（模态0和1）
        smiles_list, protein_list = self._process_sequences(data.drug, data.target)
        seq_features = self.sequence_extractor(smiles_list, protein_list)  # [batch, d_model]
        
        # c) 3D结构特征（模态2）
        protein_3d_features = self.protein_3d_extractor(
            protein_3d_paths=data.protein_3d_path if hasattr(data, 'protein_3d_path') else [],
            device=self.device
        )
        
        # ========== 2. 四个模态的特征解缠结 ==========
        
        # 分割序列特征
        batch_size = seq_features.size(0)
        seq_dim = seq_features.size(1)
        
        # 确保可以均匀分割
        if seq_dim % 2 != 0:
            seq_dim = seq_dim - 1  # 如果是奇数，减1变成偶数
            seq_features = seq_features[:, :seq_dim]
        
        half_dim = seq_dim // 2
        
        # 模态0: SMILES序列特征
        smiles_features = seq_features[:, :half_dim]
        
        smiles_result = self.smiles_disentangler(
            smiles_features,
            modality_id=0 if self.training else None,
            compute_recon=self.training
        )
        
        # 模态1: 蛋白质序列特征
        protein_features = seq_features[:, half_dim:2*half_dim] if half_dim*2 <= seq_features.size(1) else seq_features[:, :half_dim]
        
        protein_result = self.protein_disentangler(
            protein_features,
            modality_id=1 if self.training else None,
            compute_recon=self.training
        )
        
        # 模态2: 蛋白质3D结构特征
        struct_result = self.struct_disentangler(
            protein_3d_features,
            modality_id=2 if self.training else None,
            compute_recon=self.training
        )
        
        # 模态3: 分子图特征
        graph_result = self.graph_disentangler(
            graph_features,
            modality_id=3 if self.training else None,
            compute_recon=self.training
        )
        
        disentangled_outputs = {
            'smiles': smiles_result,
            'protein': protein_result,
            'struct': struct_result,
            'graph': graph_result
        }
        
        # ========== 3. 不变性特征融合（4模态） ==========
        
        # 使用原始的shared特征（shared_dim维度）
        shared_features = [
            disentangled_outputs['smiles']['shared'],     # [batch, shared_dim]
            disentangled_outputs['protein']['shared'],    # [batch, shared_dim]
            disentangled_outputs['struct']['shared'],     # [batch, shared_dim]
            disentangled_outputs['graph']['shared']       # [batch, shared_dim]
        ]
        
        fusion_result = self.invariant_fusion(shared_features)
        
        # ========== 4. 特征对齐 ==========
        
        # 对齐融合共享特征
        aligned_shared = self.shared_alignment(fusion_result['fused_shared'])
        
        # 对齐各个模态的特有特征
        aligned_specifics = []
        for modality in ['smiles', 'protein', 'struct', 'graph']:
            aligned_spec = self.specific_alignment(
                disentangled_outputs[modality]['specific']
            )
            aligned_specifics.append(aligned_spec)
        
        # ========== 5. 特征拼接与预测 ==========
        
        # 拼接所有特征：对齐后的共享特征 + 4个对齐后的特有特征
        all_features = torch.cat([
            aligned_shared,  # [batch, 256]
            aligned_specifics[0],  # smiles特有 [batch, 128]
            aligned_specifics[1],  # protein特有 [batch, 128]
            aligned_specifics[2],  # struct特有 [batch, 128]
            aligned_specifics[3],  # graph特有 [batch, 128]
        ], dim=-1)  # 总维度: 256 + 128*4 = 256 + 512 = 768
        
        # 亲和力预测
        affinity_pred = self.classifier(all_features)
        
        # ========== 6. 计算训练损失 ==========
        
        losses = {}
        if self.training:
            # 重构损失
            recon_loss = 0
            recon_count = 0
            for modality in ['smiles', 'protein', 'struct', 'graph']:
                if disentangled_outputs[modality]['recon_loss'] is not None:
                    recon_loss += disentangled_outputs[modality]['recon_loss']
                    recon_count += 1
            if recon_count > 0:
                recon_loss = recon_loss / recon_count
            losses['recon_loss'] = recon_loss
            
            # 对抗损失
            adv_loss = 0
            adv_count = 0
            for modality, modality_id in [('smiles', 0), ('protein', 1), 
                                          ('struct', 2), ('graph', 3)]:
                if disentangled_outputs[modality]['modality_pred'] is not None:
                    adv_loss += self._compute_adversarial_loss(
                        disentangled_outputs[modality]['modality_pred'],
                        modality_id
                    )
                    adv_count += 1
            if adv_count > 0:
                adv_loss = adv_loss / adv_count
            losses['adv_loss'] = adv_loss
            
            # 不变性损失
            invariance_loss = self._compute_invariance_loss(shared_features)
            losses['invariance_loss'] = invariance_loss
        
        # ========== 7. 准备输出 ==========
        
        output = affinity_pred
        
        if return_details:
            details = {
                'affinity_pred': affinity_pred,
                'shared_features': aligned_shared,
                'specific_features': torch.cat(aligned_specifics, dim=-1),
                'env_weights': fusion_result['env_weights'],  # 4个模态的权重
                'invariance_score': fusion_result['invariance_score'],
                'losses': losses if self.training else None
            }
            return output, details
        
        return output
    
    def get_total_loss(self, pred, target, extra_outputs=None):
        """计算总损失（用于训练）"""
        # 基础亲和力损失
        affinity_loss = F.mse_loss(pred.view(-1), target.view(-1))
        
        total_loss = affinity_loss
        
        # 添加因果不变性正则化损失
        if extra_outputs is not None and 'losses' in extra_outputs:
            extra_losses = extra_outputs['losses']
            total_loss = (
                affinity_loss +
                self.alpha_recon * extra_losses.get('recon_loss', 0.0) +
                self.alpha_adv * extra_losses.get('adv_loss', 0.0) +
                self.alpha_invariance * extra_losses.get('invariance_loss', 0.0)
            )
        
        return total_loss