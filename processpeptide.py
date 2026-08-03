import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
import numpy as np
from torch_geometric import data as DATA
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import math  # 新增用于位置编码

# 获取特征库，构建特征工厂
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25}

def seq2int(target):
    return [VOCAB_PROTEIN[s] for s in target]

VOCAB_DRUG = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
              "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
              "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, "X": 67,
              "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, "Q": 65,
              "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
              "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, "x": 66,
              "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, "q": 68,
              "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def seqs2int_drug(drug):
    return [VOCAB_DRUG[s] for s in drug]

class UniversalCyclicPeptideFeatureExtractor:
    """通用环肽特征提取器 - 专门针对非天然氨基酸和SMILES输入"""
    
    def __init__(self):
        self.functional_group_patterns = self._setup_functional_group_patterns()
        
    def _setup_functional_group_patterns(self) -> Dict[str, str]:
        """设置通用官能团SMARTS模式"""
        return {
            # 疏水基团
            'alkyl_chain': '[CH3,CH2,CH]C',
            'aryl_group': 'c1ccccc1',
            'cycloalkyl': 'C1CCCC1',
            
            # 极性基团
            'hydroxyl': '[OH]',
            'carbonyl': 'C=O',
            'carboxyl': 'C(=O)[OH]',
            'amide': 'C(=O)N',
            'amine': '[NH2,NH]',
            'nitro': 'N(=O)=O',
            
            # 带电基团
            'carboxylate': 'C(=O)[O-]',
            'ammonium': '[NH4+]',
            'guanidine': 'NC(=N)N',
            'sulfonate': 'S(=O)(=O)[O-]',
            
            # 特殊基团
            'halogen': '[F,Cl,Br,I]',
            'sulfur': '[SH,S]',
            'phosphate': 'P(=O)(O)O',
        }
    
    def extract_molecule_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """提取分子级别的环肽特征"""
        features = {}
        
        if mol is None:
            return self._get_default_features()
        
        try:
            # 1. 环系统特征
            features.update(self._get_ring_system_features(mol))
            
            # 2. 肽键特征
            features.update(self._get_peptide_bond_features(mol))
            
            # 3. 官能团特征
            features.update(self._get_functional_group_features(mol))
            
            # 4. 物理化学特征
            features.update(self._get_physicochemical_features(mol))
            
        except Exception as e:
            logger.warning(f"分子特征提取失败: {e}")
            features = self._get_default_features()
            
        return features
    
    def extract_atom_features(self, mol: Chem.Mol, atom_idx: int) -> List[float]:
        """提取原子级别的环肽特征"""
        atom_features = []
        
        try:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # 基础原子特征
            atom_features.extend(self._get_basic_atom_features(atom))
            
            # 环肽特异性原子特征
            atom_features.extend(self._get_cyclic_peptide_atom_features(mol, atom_idx))
            
        except Exception as e:
            logger.warning(f"原子特征提取失败: {e}")
            # 返回默认长度的特征向量
            atom_features = [0.0] * 25  # 基础特征15 + 环肽特征10
            
        return atom_features
    
    def extract_bond_features(self, mol: Chem.Mol, atom1_idx: int, atom2_idx: int) -> List[float]:
        """提取键级别的环肽特征"""
        bond_features = []
        
        try:
            bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
            if bond is None:
                return [0.0] * 12
            
            # 基础键特征
            bond_features.extend(self._get_basic_bond_features(bond))
            
            # 环肽特异性键特征
            bond_features.extend(self._get_cyclic_peptide_bond_features(mol, bond))
            
        except Exception as e:
            logger.warning(f"键特征提取失败: {e}")
            bond_features = [0.0] * 12
            
        return bond_features
    
    def _get_ring_system_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """环系统特征"""
        features = {}
        
        try:
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            
            if not atom_rings:
                return {
                    'num_rings': 0,
                    'largest_ring_size': 0,
                    'smallest_ring_size': 0,
                    'avg_ring_size': 0.0,
                    'macrocycle_present': 0,
                    'small_ring_present': 0,
                    'ring_complexity': 0.0,
                }
            
            ring_sizes = [len(ring) for ring in atom_rings]
            
            features.update({
                'num_rings': len(ring_sizes),
                'largest_ring_size': max(ring_sizes),
                'smallest_ring_size': min(ring_sizes),
                'avg_ring_size': float(np.mean(ring_sizes)),
                'macrocycle_present': int(any(size >= 8 for size in ring_sizes)),
                'small_ring_present': int(any(size <= 4 for size in ring_sizes)),
                'ring_complexity': self._calculate_ring_complexity(ring_sizes),
            })
            
        except Exception as e:
            logger.warning(f"环系统特征提取失败: {e}")
            features.update(self._get_default_ring_features())
            
        return features
    
    def _get_peptide_bond_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """肽键特征"""
        features = {}
        
        try:
            amide_bonds = 0
            for bond in mol.GetBonds():
                if self._is_amide_bond(bond, mol):
                    amide_bonds += 1
            
            total_bonds = mol.GetNumBonds()
            
            features.update({
                'amide_bond_count': amide_bonds,
                'amide_bond_ratio': amide_bonds / total_bonds if total_bonds > 0 else 0.0,
                'estimated_cyclic_units': amide_bonds + 1 if amide_bonds > 0 else 1,
                'cyclic_peptide_likelihood': self._assess_cyclic_peptide_likelihood(mol, amide_bonds),
            })
            
        except Exception as e:
            logger.warning(f"肽键特征提取失败: {e}")
            features.update({
                'amide_bond_count': 0,
                'amide_bond_ratio': 0.0,
                'estimated_cyclic_units': 1,
                'cyclic_peptide_likelihood': 0.0,
            })
            
        return features
    
    def _get_functional_group_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """官能团特征"""
        features = {}
        
        try:
            group_counts = {}
            for group_name, smarts_pattern in self.functional_group_patterns.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts_pattern)
                    if pattern:
                        matches = len(mol.GetSubstructMatches(pattern))
                        group_counts[group_name] = matches
                    else:
                        group_counts[group_name] = 0
                except:
                    group_counts[group_name] = 0
            
            features.update(group_counts)
            
            # 官能团多样性
            present_groups = sum(1 for count in group_counts.values() if count > 0)
            features['functional_group_diversity'] = present_groups
            features['functional_group_density'] = sum(group_counts.values()) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"官能团特征提取失败: {e}")
            for group_name in self.functional_group_patterns.keys():
                features[group_name] = 0
            features.update({
                'functional_group_diversity': 0,
                'functional_group_density': 0.0,
            })
            
        return features
    
    def _get_physicochemical_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """物理化学特征"""
        features = {}
        
        try:
            from rdkit.Chem import Descriptors
            
            features.update({
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd_count': Descriptors.NumHDonors(mol),
                'hba_count': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'heavy_atom_count': mol.GetNumHeavyAtoms(),
            })
            
        except Exception as e:
            logger.warning(f"物化特征提取失败: {e}")
            features.update({
                'molecular_weight': 0.0,
                'logp': 0.0,
                'tpsa': 0.0,
                'hbd_count': 0,
                'hba_count': 0,
                'rotatable_bonds': 0,
                'heavy_atom_count': 0,
            })
            
        return features
    
    def _get_basic_atom_features(self, atom: Chem.Atom) -> List[float]:
        """基础原子特征"""
        features = []
        
        # 原子类型
        symbols = ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', 'P']
        features.extend([int(atom.GetSymbol() == sym) for sym in symbols])
        
        # 原子属性
        features.append(atom.GetAtomicNum())
        features.append(int(atom.GetIsAromatic()))
        features.append(atom.GetDegree())
        features.append(atom.GetTotalNumHs())
        features.append(atom.GetFormalCharge())
        
        return features
    
    def _get_cyclic_peptide_atom_features(self, mol: Chem.Mol, atom_idx: int) -> List[float]:
        """环肽特异性原子特征"""
        features = []
        
        try:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # 是否在环中
            features.append(float(atom.IsInRing()))
            
            # 是否在肽键中
            in_peptide_bond = 0
            for bond in atom.GetBonds():
                if self._is_amide_bond(bond, mol):
                    in_peptide_bond = 1
                    break
            features.append(float(in_peptide_bond))
            
            # 环大小（如果在环中）
            ring_size = 0
            if atom.IsInRing():
                ring_info = mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if atom_idx in ring:
                        ring_size = len(ring)
                        break
            features.append(float(ring_size))
            
            # 是否为大环原子
            features.append(float(ring_size >= 8 if ring_size > 0 else 0))
            
            # 是否为小环原子
            features.append(float(3 <= ring_size <= 4 if ring_size > 0 else 0))
            
            # 原子在分子中的中心性（简化版本）
            features.append(float(atom.GetDegree() / max(1, mol.GetNumAtoms())))
            
        except Exception as e:
            features.extend([0.0] * 6)  # 6个环肽特异性特征
            
        return features
    
    def _get_basic_bond_features(self, bond: Chem.Bond) -> List[float]:
        """基础键特征"""
        features = []
        
        # 键类型
        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        features.extend([int(bond.GetBondType() == bt) for bt in bond_types])
        
        # 键属性
        features.append(float(bond.GetIsConjugated()))
        features.append(float(bond.IsInRing()))
        
        return features
    
    def _get_cyclic_peptide_bond_features(self, mol: Chem.Mol, bond: Chem.Bond) -> List[float]:
        """环肽特异性键特征"""
        features = []
        
        try:
            # 是否为肽键
            features.append(float(self._is_amide_bond(bond, mol)))
            
            # 键在环中的重要性
            importance = 0
            if bond.IsInRing():
                importance = 1
                if self._is_amide_bond(bond, mol):
                    importance = 2
            features.append(float(importance))
            
            # 键长度估计（基于原子类型）
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            features.append(float(self._estimate_bond_length(atom1, atom2)))
            
        except Exception as e:
            features.extend([0.0] * 3)  # 3个环肽特异性键特征
            
        return features
    
    def _is_amide_bond(self, bond: Chem.Bond, mol: Chem.Mol) -> bool:
        """判断是否为酰胺键"""
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        if (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or \
           (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C'):
            
            carbon_atom = atom1 if atom1.GetSymbol() == 'C' else atom2
            for neighbor in carbon_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    bond_to_o = mol.GetBondBetweenAtoms(carbon_atom.GetIdx(), neighbor.GetIdx())
                    if bond_to_o and bond_to_o.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        return True
        return False
    
    def _calculate_ring_complexity(self, ring_sizes: List[int]) -> float:
        """计算环复杂性"""
        if not ring_sizes:
            return 0.0
        
        complexity = len(ring_sizes)  # 环数量
        if len(ring_sizes) > 1:
            complexity *= (1 + np.std(ring_sizes))  # 环大小多样性
        
        return float(complexity)
    
    def _assess_cyclic_peptide_likelihood(self, mol: Chem.Mol, amide_bond_count: int) -> float:
        """评估环肽可能性"""
        score = 0.0
        
        # 基于酰胺键数量
        if amide_bond_count >= 2:
            score += 0.4
        if amide_bond_count >= 3:
            score += 0.3
        
        # 基于环的存在
        if mol.GetRingInfo().NumRings() > 0:
            score += 0.2
            
        # 基于分子大小
        heavy_atoms = mol.GetNumHeavyAtoms()
        if 15 <= heavy_atoms <= 100:  # 环肽典型大小
            score += 0.1
            
        return min(1.0, score)
    
    def _estimate_bond_length(self, atom1: Chem.Atom, atom2: Chem.Atom) -> float:
        """估计键长度（基于原子类型）"""
        # 简化的键长估计
        symbols = sorted([atom1.GetSymbol(), atom2.GetSymbol()])
        bond_pair = ''.join(symbols)
        
        length_map = {
            'CC': 1.54, 'CN': 1.47, 'CO': 1.43, 'CS': 1.82,
            'NN': 1.45, 'NO': 1.36, 'OO': 1.48,
            'CH': 1.09, 'NH': 1.01, 'OH': 0.96,
        }
        
        return length_map.get(bond_pair, 1.5)  # 默认值
    
    def _get_default_features(self) -> Dict[str, Any]:
        """获取默认特征值"""
        default_features = {}
        
        # 环系统默认值
        default_features.update({
            'num_rings': 0, 'largest_ring_size': 0, 'smallest_ring_size': 0,
            'avg_ring_size': 0.0, 'macrocycle_present': 0, 'small_ring_present': 0,
            'ring_complexity': 0.0,
        })
        
        # 肽键默认值
        default_features.update({
            'amide_bond_count': 0, 'amide_bond_ratio': 0.0,
            'estimated_cyclic_units': 1, 'cyclic_peptide_likelihood': 0.0,
        })
        
        # 官能团默认值
        for group_name in self.functional_group_patterns.keys():
            default_features[group_name] = 0
        default_features.update({
            'functional_group_diversity': 0, 'functional_group_density': 0.0,
        })
        
        # 物化默认值
        default_features.update({
            'molecular_weight': 0.0, 'logp': 0.0, 'tpsa': 0.0,
            'hbd_count': 0, 'hba_count': 0, 'rotatable_bonds': 0,
            'heavy_atom_count': 0,
        })
        
        return default_features
    
    def _get_default_ring_features(self) -> Dict[str, Any]:
        """获取环特征默认值"""
        return {
            'num_rings': 0, 'largest_ring_size': 0, 'smallest_ring_size': 0,
            'avg_ring_size': 0.0, 'macrocycle_present': 0, 'small_ring_present': 0,
            'ring_complexity': 0.0,
        }

class GNNDataset(InMemoryDataset):
    def __init__(self, root, train=True, transformer=None, pre_transformer=None, pre_filter=None):
        # 先初始化特征提取器
        self.cyclic_feature_extractor = UniversalCyclicPeptideFeatureExtractor()
        
        # 然后调用父类初始化
        super().__init__(root, transformer, pre_transformer, pre_filter)
        
        # 加载数据
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        # 检查raw文件夹中是否已有分割好的文件
        if all(osp.exists(osp.join(self.raw_dir, f)) for f in ['data_train.csv', 'data_test.csv']):
            return ['data_train.csv', 'data_test.csv']
        return ['data.csv']  # 只有单个数据集时

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def split_dataset(self):
        """将单个data.csv分割为训练集和测试集"""
        if all(osp.exists(osp.join(self.raw_dir, f)) for f in ['data_train.csv', 'data_test.csv']):
            print("训练集和测试集文件已存在，跳过分割")
            return
            
        data_path = osp.join(self.raw_dir, 'data.csv')
        if not osp.exists(data_path):
            raise FileNotFoundError(f"未找到数据集文件: {data_path}")
            
        print(f"从 {data_path} 分割训练集和测试集")
        df = pd.read_csv(data_path)
        
        # 使用stratify确保类别分布均匀（如果有类别标签）
        if 'class' in df.columns:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
        else:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
        train_path = osp.join(self.raw_dir, 'data_train.csv')
        test_path = osp.join(self.raw_dir, 'data_test.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"训练集保存至: {train_path} ({len(train_df)} 样本)")
        print(f"测试集保存至: {test_path} ({len(test_df)} 样本)")

    def process_data(self, data_path, graph_dict):
        df = pd.read_csv(data_path)
        data_list = []
        
        failed_count = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {osp.basename(data_path)}"):
            try:
                smi = row['compound_iso_smiles']
                sequence = row['target_sequence']
                label = row['affinity']
                protein_3d_path = row.get('protein_3d_path', '')  # 可选字段

                if smi not in graph_dict:
                    failed_count += 1
                    continue
                    
                x, edge_index, edge_attr = graph_dict[smi]
                
                # 节点特征标准化
                if x.numel() > 0 and x.shape[0] > 0:
                    x_min = x.min(dim=0)[0]
                    x_max = x.max(dim=0)[0]
                    mask = (x_max - x_min) > 1e-6  # 避免除零
                    if mask.any():
                        x[:, mask] = (x[:, mask] - x_min[mask]) / (x_max[mask] - x_min[mask])

                # 蛋白质序列编码
                target = seq2int(sequence)
                target_len = 1200
                if len(target) < target_len:
                    target = np.pad(target, (0, target_len - len(target)))
                else:
                    target = target[:target_len]

                # 药物SMILES编码
                drug = seqs2int_drug(smi)
                drug_len = 150
                if len(drug) < drug_len:
                    drug = np.pad(drug, (0, drug_len - len(drug)))
                else:
                    drug = drug[:drug_len]
                
                label = float(label)
                
                # 创建数据对象
                data = DATA.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.FloatTensor([label]),
                    target=torch.LongTensor([target]),
                    drug=torch.LongTensor([drug]),
                    protein_3d_path=protein_3d_path,
                    smiles=smi  # 保存SMILES用于调试
                )

                data_list.append(data)
                
            except Exception as e:
                logger.warning(f"处理第 {i} 行数据失败: {e}")
                failed_count += 1
                continue
        
        if failed_count > 0:
            logger.warning(f"共 {failed_count} 个样本处理失败")
            
        return data_list

    def process(self):
        # 先分割数据集
        self.split_dataset()
        
        # 读取分割后的训练集和测试集
        df_train = pd.read_csv(osp.join(self.raw_dir, 'data_train.csv'))
        df_test = pd.read_csv(osp.join(self.raw_dir, 'data_test.csv'))
        df = pd.concat([df_train, df_test])

        # 构建分子图字典
        smiles_list = df['compound_iso_smiles'].unique()
        graph_dict = self.build_graph_dict(smiles_list)

        # 处理训练和测试数据
        train_list = self.process_data(osp.join(self.raw_dir, 'data_train.csv'), graph_dict)
        test_list = self.process_data(osp.join(self.raw_dir, 'data_test.csv'), graph_dict)

        # 应用预过滤和预变换
        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            test_list = [self.pre_transform(test) for test in test_list]

        print('图构建完成，保存文件...')
        if train_list:  # 确保列表不为空
            data, slices = self.collate(train_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            logger.warning("训练集为空，跳过保存")

        if test_list:  # 确保列表不为空
            data, slices = self.collate(test_list)
            torch.save((data, slices), self.processed_paths[1])
        else:
            logger.warning("测试集为空，跳过保存")

    def build_graph_dict(self, smiles_list):
        """构建分子图字典"""
        graph_dict = {}
        failed_smiles = []
        
        for smile in tqdm(smiles_list, desc="构建分子图"):
            if smile in graph_dict:
                continue
                
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    graph = self.mol2graph(mol)
                    if graph is not None and graph[0] is not None:
                        graph_dict[smile] = graph
                    else:
                        failed_smiles.append(smile)
                else:
                    failed_smiles.append(smile)
            except Exception as e:
                logger.warning(f"处理SMILES {smile} 失败: {e}")
                failed_smiles.append(smile)
        
        if failed_smiles:
            logger.warning(f"共 {len(failed_smiles)} 个SMILES处理失败")
            
        return graph_dict

    def get_nodes(self, g, mol):
        """增强的节点特征，包含环肽特异性特征"""
        feat = []
        
        for n, d in g.nodes(data=True):
            h_t = []
            
            # 基础原子特征
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I']]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            
            # 5个额外的基础特征
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            
            # 环肽特异性原子特征
            cyclic_atom_features = self.cyclic_feature_extractor.extract_atom_features(mol, n)
            h_t.extend(cyclic_atom_features)
            
            feat.append((n, h_t))
        
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def get_edges(self, g, mol):
        """增强的边特征，包含环肽特异性特征"""
        e = {}
        
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            
            # 环肽特异性边特征
            cyclic_bond_features = self.cyclic_feature_extractor.extract_bond_features(mol, n1, n2)
            e_t.extend(cyclic_bond_features)
            
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0] * 12])  # 12个特征维度

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    # ==================== 新增环肽感知增强方法 ====================
    def _find_peptide_bonds(self, mol: Chem.Mol):
        """
        识别分子中所有的肽键（酰胺键）。
        返回：列表，每个元素是 (carbon_idx, nitrogen_idx) 元组，
             其中 carbon_idx 是羰基碳原子索引，nitrogen_idx 是与之形成肽键的氮原子索引。
        """
        peptide_bonds = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            # 检查是否为 C-N 键
            if (a1.GetSymbol() == 'C' and a2.GetSymbol() == 'N') or (a1.GetSymbol() == 'N' and a2.GetSymbol() == 'C'):
                # 确定碳原子和氮原子
                carbon = a1 if a1.GetSymbol() == 'C' else a2
                nitrogen = a2 if a1.GetSymbol() == 'C' else a1
                # 检查碳原子上是否有双键氧（羰基）
                has_oxo = False
                for neighbor in carbon.GetNeighbors():
                    if neighbor.GetSymbol() == 'O':
                        bond_to_o = mol.GetBondBetweenAtoms(carbon.GetIdx(), neighbor.GetIdx())
                        if bond_to_o and bond_to_o.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            has_oxo = True
                            break
                if has_oxo:
                    peptide_bonds.append((carbon.GetIdx(), nitrogen.GetIdx()))
        return peptide_bonds

    def _build_backbone_cycle(self, mol: Chem.Mol, peptide_bonds):
        """
        根据肽键列表构建主链环上的代表原子序列（使用羰基碳作为代表）。
        返回：一个列表，包含按顺序排列的羰基碳原子索引（代表原子）。
        """
        if not peptide_bonds:
            return []

        # 构建映射：碳->氮，氮->碳
        c_to_n = {c: n for c, n in peptide_bonds}
        n_to_c = {n: c for c, n in peptide_bonds}

        # 找一个起点（任意碳）
        start_c = peptide_bonds[0][0]
        sequence = []
        current_c = start_c
        visited = set()
        
        # 沿着碳->氮->碳... 走，直到回到起点
        while current_c not in visited:
            visited.add(current_c)
            # 碳对应的氮
            if current_c not in c_to_n:
                break
            current_n = c_to_n[current_c]
            # 氮对应的碳
            if current_n not in n_to_c:
                break
            sequence.append(current_c)
            current_c = n_to_c[current_n]
            if current_c == start_c:
                # 闭合
                sequence.append(current_c)  # 为了闭合，最后加上起点
                break
        
        # 如果未能形成环，返回空
        if len(sequence) <= 1:
            return []
        
        # 去重：可能最后一个元素重复了起点，去掉
        if sequence[0] == sequence[-1]:
            sequence = sequence[:-1]
        
        return sequence

    def _add_cycle_closing_edge(self, edge_index, edge_attr, rep_atom_sequence):
        """
        在代表原子序列的首尾之间添加一条虚拟边。
        """
        if len(rep_atom_sequence) < 2:
            return edge_index, edge_attr
        
        first = rep_atom_sequence[0]
        last = rep_atom_sequence[-1]
        
        # 如果已经存在这条边，就不重复添加
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        if (first, last) in existing_edges or (last, first) in existing_edges:
            return edge_index, edge_attr
        
        # 添加新边
        new_edge = torch.tensor([[first], [last]], dtype=torch.long)
        edge_index = torch.cat([edge_index, new_edge], dim=1)
        
        # 为这条新边添加边特征（用零填充）
        if edge_attr is not None:
            feat_dim = edge_attr.size(1)
            new_edge_attr = torch.zeros(1, feat_dim, dtype=edge_attr.dtype)
            # 可选：在某个维度上标记为环闭合边（比如第0维设为1）
            # new_edge_attr[0, 0] = 1.0
            edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
        
        return edge_index, edge_attr

    def _compute_position_encoding(self, mol: Chem.Mol, rep_atom_sequence):
        """
        计算每个原子的环位置编码（sin/cos）。
        返回：一个字典 {atom_idx: tensor(4,)} 或 None（如果无法计算）
        """
        if not rep_atom_sequence:
            return None
        
        n = len(rep_atom_sequence)
        # 为每个代表原子计算位置编码（基于其索引在环上的比例）
        pos_enc = {}
        for i, atom_idx in enumerate(rep_atom_sequence):
            angle = 2 * math.pi * i / n
            enc = [math.sin(angle), math.cos(angle), math.sin(2*angle), math.cos(2*angle)]
            pos_enc[atom_idx] = torch.tensor(enc, dtype=torch.float)
        
        # 对于非主链原子，默认使用全零编码
        # 可以通过BFS传播，但这里简化：所有原子都赋予编码，非主链原子为零
        all_atom_enc = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in pos_enc:
                all_atom_enc[idx] = pos_enc[idx]
            else:
                all_atom_enc[idx] = torch.zeros(4)
        
        return all_atom_enc

    # ==================== 修改 mol2graph 方法，集成增强 ====================
    def mol2graph(self, mol):
        """将分子转换为图，并添加环肽感知增强（环闭合边和位置编码）"""
        if mol is None:
            return None

        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        # 创建节点
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        # 添加供体/受体特征
        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        # 创建边
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        # 获取原始节点和边特征
        node_attr = self.get_nodes(g, mol)
        edge_index, edge_attr = self.get_edges(g, mol)

        # ---------- 环肽感知增强 ----------
        # 1. 识别肽键
        peptide_bonds = self._find_peptide_bonds(mol)
        # 2. 构建主链环序列
        backbone_cycle = self._build_backbone_cycle(mol, peptide_bonds)

        # 3. 始终添加位置编码（有环则用环编码，无环则全零）
        if backbone_cycle:
            pos_enc_dict = self._compute_position_encoding(mol, backbone_cycle)
        else:
            pos_enc_dict = None

        if pos_enc_dict:
            # 使用计算出的位置编码
            pos_enc_tensor = torch.stack([pos_enc_dict[i] for i in range(node_attr.size(0))], dim=0)
        else:
            # 没有环，使用全零编码
            pos_enc_tensor = torch.zeros(node_attr.size(0), 4, dtype=node_attr.dtype)

        # 拼接到节点特征
        node_attr = torch.cat([node_attr, pos_enc_tensor], dim=1)

        # 4. 添加环闭合边（仅当有环时）
        if backbone_cycle:
            edge_index, edge_attr = self._add_cycle_closing_edge(edge_index, edge_attr, backbone_cycle)
        # ---------------------------------

        return node_attr, edge_index, edge_attr


if __name__ == "__main__":
    # 使用示例
    dataset = GNNDataset('data/peptide')
    print(f"数据集处理完成!")
    print(f"训练集样本数: {len(dataset) if dataset.data else 0}")