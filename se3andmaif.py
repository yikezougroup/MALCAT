import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, euclidean, pdist, squareform
from scipy.sparse import csr_matrix
import os
import time
import pandas as pd
from typing import Callable, List, Optional, Any, Tuple, Union, Dict
from concurrent.futures import ProcessPoolExecutor
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import Structure

# 忽略特定警告
warnings.filterwarnings("ignore")

# ==================== 共价键实现 ====================
# 原子键状态定义
DEFAULT_BOND_STATE: Dict[str, str] = {
    "N": "Nsb", "CA": "Csb", "C": "Cdb", "O": "Odb", "OXT": "Osb",
    "CB": "Csb", "H": "Hsb", "HG1": "Hsb", "HE": "Hsb", "1HH1": "Hsb",
    "1HH2": "Hsb", "2HH1": "Hsb", "2HH2": "Hsb", "HG": "Hsb", "HH": "Hsb",
    "1HD2": "Hsb", "2HD2": "Hsb", "HZ1": "Hsb", "HZ2": "Hsb", "HZ3": "Hsb",
}

RESIDUE_ATOM_BOND_STATE: Dict[str, Dict[str, str]] = {
    "XXX": {"N": "Nsb", "CA": "Csb", "C": "Cdb", "O": "Odb", "OXT": "Osb", "CB": "Csb"},
    "VAL": {"CG1": "Csb", "CG2": "Csb"},
    "LEU": {"CG": "Csb", "CD1": "Csb", "CD2": "Csb"},
    "ILE": {"CG1": "Csb", "CG2": "Csb", "CD1": "Csb"},
    "MET": {"CG": "Csb", "SD": "Ssb", "CE": "Csb"},
    "PHE": {"CG": "Cdb", "CD1": "Cres", "CD2": "Cres", "CE1": "Cdb", "CE2": "Cdb", "CZ": "Cres"},
    "PRO": {"CG": "Csb", "CD": "Csb"},
    "SER": {"OG": "Osb"},
    "THR": {"OG1": "Osb", "CG2": "Csb"},
    "CYS": {"SG": "Ssb"},
    "ASN": {"CG": "Csb", "OD1": "Odb", "ND2": "Ndb"},
    "GLN": {"CG": "Csb", "CD": "Csb", "OE1": "Odb", "NE2": "Ndb"},
    "TYR": {"CG": "Cdb", "CD1": "Cres", "CD2": "Cres", "CE1": "Cdb", "CE2": "Cdb", "CZ": "Cres", "OH": "Osb"},
    "TRP": {"CG": "Cdb", "CD1": "Cdb", "CD2": "Cres", "NE1": "Nsb", "CE2": "Cdb", "CE3": "Cdb", "CZ2": "Cres", "CZ3": "Cres", "CH2": "Cdb"},
    "ASP": {"CG": "Csb", "OD1": "Ores", "OD2": "Ores"},
    "GLU": {"CG": "Csb", "CD": "Csb", "OE1": "Ores", "OE2": "Ores"},
    "HIS": {"CG": "Cdb", "CD2": "Cdb", "ND1": "Nsb", "CE1": "Cdb", "NE2": "Ndb"},
    "LYS": {"CG": "Csb", "CD": "Csb", "CE": "Csb", "NZ": "Nsb"},
    "ARG": {"CG": "Csb", "CD": "Csb", "NE": "Nsb", "CZ": "Cdb", "NH1": "Nres", "NH2": "Nres"},
}

# 共价半径定义
COVALENT_RADII: Dict[str, float] = {
    "Csb": 0.77, "Cres": 0.72, "Cdb": 0.67,
    "Osb": 0.67, "Ores": 0.635, "Odb": 0.60,
    "Nsb": 0.70, "Nres": 0.66, "Ndb": 0.62,
    "Hsb": 0.37, "Ssb": 1.04,
}

# 残基类型编码
RESIDUE_TYPES = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

def get_residue_type(res_name):
    """获取残基类型编码"""
    return RESIDUE_TYPES.get(res_name, 20) / 20.0  # 归一化

def filter_dataframe(dataframe: pd.DataFrame, by_column: str, 
                    list_of_values: List[Any], boolean: bool) -> pd.DataFrame:
    """
    过滤DataFrame
    """
    df = dataframe.copy()
    df = df[df[by_column].isin(list_of_values) == boolean]
    return df.reset_index(drop=True)

def deprotonate_structure(df: pd.DataFrame) -> pd.DataFrame:
    """移除氢原子"""
    hydrogen_atoms = ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 
                     'HZ', 'HG2', 'HG3', 'HE21', 'HE22', 'HG', 'HD11', 'HD12', 'HD13', 
                     'HD21', 'HD22', 'HD23', 'HB', 'HG21', 'HG22', 'HG23', 'HG1', 'HD3', 
                     'HH', 'HG11', 'HG12', 'HG13', 'HE3', 'HZ1', 'HZ2', 'HZ3', 'HH2', 
                     'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HA2', 'HA3', 'HB1']
    return filter_dataframe(df, "atom_name", hydrogen_atoms, False)

def compute_distmat(pdb_df: pd.DataFrame) -> pd.DataFrame:
    """计算原子间的欧几里得距离矩阵"""
    eucl_dists = pdist(pdb_df[["x_coord", "y_coord", "z_coord"]], metric="euclidean")
    return pd.DataFrame(squareform(eucl_dists), index=pdb_df.index, columns=pdb_df.index)

def assign_bond_states_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """为DataFrame中的原子分配键状态"""
    # 映射标准键状态
    naive_bond_states = pd.Series(df["atom_name"].map(DEFAULT_BOND_STATE))
    
    # 创建非标准键状态序列
    ss = (pd.DataFrame(RESIDUE_ATOM_BOND_STATE)
          .unstack()
          .rename_axis(("residue_name", "atom_name"))
          .rename("atom_bond_state"))
    
    # 合并到DataFrame
    df = df.join(ss, on=["residue_name", "atom_name"])
    return df.fillna(value={"atom_bond_state": naive_bond_states})

def assign_covalent_radii_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """为DataFrame中的原子分配共价半径"""
    df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)
    return df

def calculate_covalent_bonds(atoms: List[dict]) -> csr_matrix:
    """
    计算原子间的共价键
    返回一个稀疏邻接矩阵，其中1表示存在共价键
    """
    # 创建原子DataFrame
    atom_data = []
    for atom in atoms:
        atom_data.append({
            'atom_name': atom['atom_name'],
            'residue_name': atom['res_name'],
            'residue_number': atom['res_id'],
            'chain_id': atom['chain_id'],
            'x_coord': atom['x'],
            'y_coord': atom['y'],
            'z_coord': atom['z'],
            'element': atom['element']
        })
    
    df = pd.DataFrame(atom_data)
    
    # 去质子化
    df = deprotonate_structure(df)
    
    # 分配键状态和共价半径
    df = assign_bond_states_to_dataframe(df)
    df = assign_covalent_radii_to_dataframe(df)
    
    # 计算距离矩阵
    dist_mat = compute_distmat(df)
    
    # 创建共价半径距离矩阵
    covalent_radii = np.array(df["covalent_radius"])
    covalent_radius_distance_matrix = np.add.outer(covalent_radii, covalent_radii) + 0.56
    
    # 阈值处理距离矩阵
    t_distmat = np.where(
        (dist_mat.values > 0.4) & 
        (dist_mat.values < covalent_radius_distance_matrix) & 
        (dist_mat.values > 0), 
        1, 0
    ).astype(np.float32)
    
    # 使用稀疏矩阵存储
    row, col = np.where(t_distmat > 0)
    data = np.ones_like(row, dtype=np.float32)
    return csr_matrix((data, (row, col)), shape=t_distmat.shape)

# ==================== CIF解析和图构建 ====================
def parse_cif(cif_path):
    """解析CIF文件，返回原子信息"""
    parser = MMCIFParser()
    structure = parser.get_structure('protein', cif_path)
    
    atoms = []
    atom_counter = 0
    residue_dict = {}
    chain_counter = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in chain_counter:
                chain_counter[chain_id] = 0
            
            for residue in chain:
                res_name = residue.resname
                res_id = residue.id[1]
                
                # 为每个残基创建唯一标识符
                residue_key = f"{chain_id}_{res_id}"
                if residue_key not in residue_dict:
                    chain_counter[chain_id] += 1
                    residue_dict[residue_key] = {
                        'res_name': res_name,
                        'res_id': res_id,
                        'chain_id': chain_id,
                        'chain_res_idx': chain_counter[chain_id]
                    }
                
                for atom in residue:
                    atom_name = atom.name
                    element = atom.element
                    
                    # 添加到原子列表
                    atom_data = {
                        'atom_id': atom_counter,
                        'atom_name': atom_name,
                        'res_name': res_name,
                        'chain_id': chain_id,
                        'res_id': res_id,
                        'x': atom.coord[0],
                        'y': atom.coord[1],
                        'z': atom.coord[2],
                        'occupancy': atom.occupancy,
                        'bfactor': atom.bfactor,
                        'element': element,
                        'residue_key': residue_key
                    }
                    
                    # 分配键状态
                    atom_data['atom_bond_state'] = DEFAULT_BOND_STATE.get(atom_name, 'Csb')
                    
                    # 特殊残基的键状态处理
                    if res_name in RESIDUE_ATOM_BOND_STATE and atom_name in RESIDUE_ATOM_BOND_STATE[res_name]:
                        atom_data['atom_bond_state'] = RESIDUE_ATOM_BOND_STATE[res_name][atom_name]
                    
                    atoms.append(atom_data)
                    atom_counter += 1
    
    # 添加残基级信息到每个原子
    for atom in atoms:
        residue_info = residue_dict.get(atom['residue_key'], {})
        atom['residue_info'] = residue_info
    
    return atoms

def load_cif_to_graph(cif_path, cutoff=5.0, top_k=16, verbose=True):
    """从CIF文件构建蛋白质图，包含更多生物物理特征"""
    try:
        atoms = parse_cif(cif_path)
        if not atoms:
            print(f"警告: CIF文件 {cif_path} 中未找到原子信息")
            return None
        
        if verbose:
            print(f"从 {cif_path} 中提取了 {len(atoms)} 个原子")
        
        coords = []
        features = []
        atom_type_mapping = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
        
        # 收集坐标和基础特征
        for atom in atoms:
            coords.append([atom['x'], atom['y'], atom['z']])
            
            # 基础特征: [原子类型, B因子, 占据率, 残基疏水性, 电荷, 残基类型]
            atom_type = atom['element']
            res_name = atom['res_name']
            
            # 计算特征
            hydrophobicity = get_hydrophobicity(res_name)
            charge = get_charge(res_name, atom['atom_name'])
            res_type = get_residue_type(res_name)
            
            features.append([
                atom_type_mapping.get(atom_type, 4),
                min(atom['bfactor'] / 100.0, 1.0),  # 归一化并限制在[0,1]
                atom['occupancy'],
                hydrophobicity,
                charge,
                res_type
            ])
        
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        
        # 转换为PyTorch张量
        pos = torch.tensor(coords, dtype=torch.float)
        x = torch.tensor(features, dtype=torch.float)
        
        # 计算共价键
        covalent_adj = calculate_covalent_bonds(atoms)
        # 获取非零元素的行列索引
        covalent_rows, covalent_cols = covalent_adj.nonzero()
        # 转换为边索引格式
        covalent_edges = np.stack([covalent_rows, covalent_cols], axis=0)
        
        # 使用cKDTree构建空间邻近边
        coords_np = np.array(coords)
        tree = cKDTree(coords_np)
        
        # 构建空间邻近边
        spatial_edges = []
        for i, coord in enumerate(coords_np):
            # 查询距离在cutoff内的邻居
            indices = tree.query_ball_point(coord, cutoff)
            # 移除自身
            indices = [j for j in indices if j != i]
            
            # 如果邻居数量超过top_k，则取距离最近的top_k个
            if len(indices) > top_k:
                dists = np.linalg.norm(coords_np[indices] - coord, axis=1)
                sorted_indices = np.argsort(dists)[:top_k]
                indices = [indices[j] for j in sorted_indices]
            
            for j in indices:
                spatial_edges.append((i, j))
        
        spatial_edges = np.array(spatial_edges).T if spatial_edges else np.empty((2, 0), dtype=int)
        
        # 合并共价键和空间邻近边
        all_edges = np.concatenate([spatial_edges, covalent_edges], axis=1)
        all_edges = np.unique(all_edges, axis=1)  # 去重
        
        if verbose:
            print(f"构建了 {all_edges.shape[1]} 条边 (空间邻近: {spatial_edges.shape[1]}, 共价键: {covalent_edges.shape[1]})")
        
        # 计算边特征
        row, col = all_edges
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        
        # 边特征: [距离, 键类型]
        bond_types = calculate_bond_types(atoms, all_edges, covalent_adj)
        
        edge_attr = torch.cat([
            edge_dist,
            torch.tensor(bond_types, dtype=torch.float)
        ], dim=-1)
        
        # 创建图数据对象
        graph_data = {
            'x': x, 
            'pos': pos, 
            'edge_index': torch.tensor(all_edges, dtype=torch.long),
            'edge_attr': edge_attr
        }
        
        return graph_data
    
    except Exception as e:
        print(f"图构建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
# ==================== 其余代码保持不变 ====================
# 3. 优化的生物物理特征辅助函数
def get_hydrophobicity(res_name):
    """获取残基疏水性（更精细的尺度）"""
    # Kyte-Doolittle疏水性量表
    hydrophobicity = {
        'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
        'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
        'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
        'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
    }
    return hydrophobicity.get(res_name, 0.0) / 4.5  # 归一化

def get_charge(res_name, atom_name):
    """获取原子电荷（更精细的模型）"""
    # 带正电的原子
    positive_atoms = {
        'ARG': ['NH1', 'NH2', 'NE'],
        'LYS': ['NZ'],
        'HIS': ['ND1', 'NE2']
    }
    
    # 带负电的原子
    negative_atoms = {
        'ASP': ['OD1', 'OD2'],
        'GLU': ['OE1', 'OE2']
    }
    
    # 检查正电荷
    if res_name in positive_atoms and atom_name in positive_atoms[res_name]:
        return 1.0
    
    # 检查负电荷
    if res_name in negative_atoms and atom_name in negative_atoms[res_name]:
        return -1.0
    
    return 0.0

def calculate_bond_types(atoms, edge_index, covalent_adj):
    """计算键类型特征（优化性能）"""
    bond_types = np.zeros((edge_index.shape[1], 4))  # [无键, 共价键, 氢键, 盐桥]
    atom_dict = {atom['atom_id']: atom for atom in atoms}
    
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        
        src_atom = atom_dict[src_idx]
        dst_atom = atom_dict[dst_idx]
        
        # 检查是否共价键
        if covalent_adj[src_idx, dst_idx] > 0.5:
            bond_types[i, 1] = 1.0
            continue
        
        # 计算距离
        dist = np.linalg.norm(np.array([
            src_atom['x'] - dst_atom['x'],
            src_atom['y'] - dst_atom['y'],
            src_atom['z'] - dst_atom['z']
        ]))
        
        # 氢键 (简化)
        if dist < 3.5 and is_hbond(src_atom, dst_atom):
            bond_types[i, 2] = 1.0
        
        # 盐桥 (简化)
        elif dist < 4.5 and is_salt_bridge(src_atom, dst_atom):
            bond_types[i, 3] = 1.0
        
        else:
            bond_types[i, 0] = 1.0
    
    return bond_types

def is_hbond(atom1, atom2):
    """优化氢键判断"""
    donor_elements = ['N', 'O']
    acceptor_elements = ['O', 'N']
    
    # 氢键供体-受体对
    if (atom1['element'] in donor_elements and 
        atom2['element'] in acceptor_elements):
        return True
    
    if (atom2['element'] in donor_elements and 
        atom1['element'] in acceptor_elements):
        return True
    
    return False

def is_salt_bridge(atom1, atom2):
    """优化盐桥判断"""
    res1 = atom1['res_name']
    res2 = atom2['res_name']
    
    positive_res = ['ARG', 'LYS', 'HIS']
    negative_res = ['ASP', 'GLU']
    
    if (res1 in positive_res and res2 in negative_res) or \
       (res1 in negative_res and res2 in positive_res):
        return True
    
    return False

# ==================== 新增特征计算模块 ====================
def calculate_solvent_accessibility(atoms, probe_radius=1.4, grid_spacing=0.5):
    """计算原子溶剂可及表面积（使用简化网格法）"""
    coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])
    radii = np.array([COVALENT_RADII.get(atom['atom_bond_state'], 1.0) 
                     for atom in atoms]) + probe_radius
    
    # 边界计算
    min_coords = coords.min(axis=0) - radii.max() - probe_radius
    max_coords = coords.max(axis=0) + radii.max() + probe_radius
    
    # 网格划分
    nx = int((max_coords[0] - min_coords[0]) / grid_spacing) + 1
    ny = int((max_coords[1] - min_coords[1]) / grid_spacing) + 1
    nz = int((max_coords[2] - min_coords[2]) / grid_spacing) + 1
    
    grid = np.zeros((nx, ny, nz))
    accessible = np.zeros(len(atoms))
    
    # 填充原子占据的网格
    for i, (coord, radius) in enumerate(zip(coords, radii)):
        grid_indices = np.round((coord - min_coords) / grid_spacing).astype(int)
        # 简化的球形网格填充
        for x in range(max(0, grid_indices[0]-int(radius/grid_spacing)), 
                       min(nx, grid_indices[0]+int(radius/grid_spacing)+1)):
            for y in range(max(0, grid_indices[1]-int(radius/grid_spacing)), 
                           min(ny, grid_indices[1]+int(radius/grid_spacing)+1)):
                for z in range(max(0, grid_indices[2]-int(radius/grid_spacing)), 
                               min(nz, grid_indices[2]+int(radius/grid_spacing)+1)):
                    dist = np.linalg.norm(np.array([x, y, z]) * grid_spacing + min_coords - coord)
                    if dist <= radius:
                        grid[x, y, z] = max(grid[x, y, z], 1)
    
    # 计算可及表面
    for i, (coord, radius) in enumerate(zip(coords, radii)):
        grid_indices = np.round((coord - min_coords) / grid_spacing).astype(int)
        # 检查表面网格点
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            x, y, z = grid_indices[0]+dx, grid_indices[1]+dy, grid_indices[2]+dz
            if (0 <= x < nx and 0 <= y < ny and 0 <= z < nz and 
                grid[x, y, z] == 0):
                accessible[i] += 1
    
    # 归一化
    return accessible / (np.sum(4 * np.pi * radii**2) + 1e-6) * 100

def calculate_secondary_structure(atoms):
    """使用DSSP算法简化版计算二级结构"""
    # 实际应用中可替换为pyDSSP或biopython的DSSP实现
    # 简化版仅返回预测的二级结构类型编码
    ss_types = {'H': 0, 'E': 1, 'C': 2}  # 螺旋、片层、无规卷曲
    ss_codes = []
    
    for atom in atoms:
        res_name = atom['res_name']
        res_id = atom['res_id']
        
        # 简化的二级结构预测逻辑（实际应替换为专业算法）
        if res_name in ['ALA', 'LEU', 'MET'] and res_id % 7 in [0, 1, 2]:
            ss_codes.append(ss_types['H'])  # 螺旋
        elif res_name in ['VAL', 'ILE', 'PHE'] and res_id % 5 in [0, 1]:
            ss_codes.append(ss_types['E'])  # 片层
        else:
            ss_codes.append(ss_types['C'])  # 无规卷曲
    
    return ss_codes

def calculate_dihedral_angles(atoms):
    """计算φ和ψ二面角（Ramachandran角）"""
    phi_angles = []
    psi_angles = []
    
    # 查找N, CA, C, CB原子（简化处理）
    atom_dict = {atom['atom_id']: atom for atom in atoms}
    residue_atoms = {}
    
    for atom_id, atom in atom_dict.items():
        key = f"{atom['chain_id']}_{atom['res_id']}"
        if key not in residue_atoms:
            residue_atoms[key] = {}
        residue_atoms[key][atom['atom_name']] = atom
    
    # 计算二面角
    for key in residue_atoms:
        curr_res = residue_atoms[key]
        next_key = f"{curr_res['N']['chain_id']}_{curr_res['N']['res_id'] + 1}"
        next_res = residue_atoms.get(next_key, {})
        
        if 'N' in curr_res and 'CA' in curr_res and 'C' in curr_res and 'N' in next_res:
            # 原子坐标
            N = np.array([curr_res['N']['x'], curr_res['N']['y'], curr_res['N']['z']])
            CA = np.array([curr_res['CA']['x'], curr_res['CA']['y'], curr_res['CA']['z']])
            C = np.array([curr_res['C']['x'], curr_res['C']['y'], curr_res['C']['z']])
            N_next = np.array([next_res['N']['x'], next_res['N']['y'], next_res['N']['z']])
            
            # 计算φ角 (N-CA-C-N_next)
            v1 = CA - N
            v2 = C - CA
            v3 = N_next - C
            phi = _calculate_dihedral(v1, v2, v3)
            phi_angles.append((phi + 180) / 360.0)  # 归一化到[0,1]
            
            # 计算ψ角 (CA-C-N_next-CA_next)
            if 'CA' in next_res:
                CA_next = np.array([next_res['CA']['x'], next_res['CA']['y'], next_res['CA']['z']])
                v1 = C - CA
                v2 = N_next - C
                v3 = CA_next - N_next
                psi = _calculate_dihedral(v1, v2, v3)
                psi_angles.append((psi + 180) / 360.0)  # 归一化到[0,1]
            else:
                psi_angles.append(0.5)  # 缺失值设为中间值
    
    return phi_angles, psi_angles

def _calculate_dihedral(v1, v2, v3):
    """计算三个向量形成的二面角（单位：度）"""
    n1 = np.cross(v1, v2)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v2, v3)
    n2 /= np.linalg.norm(n2)
    
    # 计算法向量夹角
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    # 确定符号
    cross = np.cross(n1, n2)
    if np.dot(cross, v2) < 0:
        angle = -angle
    
    return angle

# ==================== 特征提取模型 ====================
# 4. 优化的MaSIF特征提取器
class MaSIFEncoder(nn.Module):
    """高效MaSIF特征提取器，包含曲率估计"""
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(64, feature_dim, kernel_size=5, padding=2),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 曲率估计分支
        self.curv_net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 3)  # 平均曲率, 高斯曲率, 形状指数
        )
    
    def estimate_curvature(self, patch):
        """更稳健的曲率估计"""
        # 中心化
        centroid = patch.mean(dim=1, keepdim=True)
        centered = patch - centroid
        
        # 计算协方差矩阵
        cov = torch.bmm(centered.permute(0, 2, 1), centered) / (centered.shape[1] - 1)
        
        # 使用SVD计算特征值
        eigenvalues = []
        for i in range(cov.size(0)):
            _, s, _ = torch.svd(cov[i])
            eigenvalues.append(s)
        eigenvalues = torch.stack(eigenvalues)
        
        # 计算曲率
        lambda1 = eigenvalues[:, 0]
        lambda2 = eigenvalues[:, 1]
        lambda3 = eigenvalues[:, 2]
        
        # 曲率特征
        mean_curvature = (lambda1 + lambda2) / 2
        gaussian_curvature = lambda1 * lambda2
        shape_index = (lambda2 - lambda1) / (lambda1 + lambda2 + 1e-6)
        
        return torch.stack([mean_curvature, gaussian_curvature, shape_index], dim=1)
    
    def forward(self, local_patch):
        """处理局部点云补丁"""
        # 转换为局部相对坐标
        centroid = local_patch.mean(dim=1, keepdim=True)
        local_patch = local_patch - centroid
        
        # 转换为通道优先格式 [B, 3, N]
        x = local_patch.permute(0, 2, 1)
        
        # 应用卷积网络
        conv_features = self.conv_net(x).squeeze(-1)
        
        # 计算曲率特征
        curv_features = self.estimate_curvature(local_patch)
        
        # 合并特征
        return torch.cat([conv_features, curv_features], dim=1)

# 5. 优化的SE(3)等变层
class SE3EquivariantLayer(nn.Module):
    """增强的SE(3)等变层，包含注意力机制和残差连接"""
    def __init__(self, in_dim, out_dim, num_degrees=2):
        super().__init__()
        self.num_degrees = num_degrees
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 方向相关的变换
        self.direction_proj = nn.ModuleList([
            nn.Linear(3, out_dim, bias=False) for _ in range(num_degrees)
        ])
        
        # 可学习的径向函数
        self.radial_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, in_dim * out_dim)
        )
        
        # 注意力机制
        self.attn_net = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)
        )
        
        # 层归一化和残差连接
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, pos, edge_index):
        row, col = edge_index
        
        # 计算相对位置
        rel_pos = pos[col] - pos[row]
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-6
        rel_dir = rel_pos / rel_dist
        
        # 径向函数
        radial_weight = self.radial_net(rel_dist).view(-1, self.in_dim, self.out_dim)
        
        # 方向投影
        dir_features = []
        for i in range(self.num_degrees):
            dir_features.append(self.direction_proj[i](rel_dir))
        dir_features = torch.stack(dir_features, dim=-1).sum(dim=-1)
        
        # 注意力分数
        attn_input = torch.cat([x[row], x[col]], dim=-1)
        attn_score = torch.sigmoid(self.attn_net(attn_input))
        
        # 消息构建
        messages = torch.einsum('bij,bi->bj', radial_weight, x[row])
        messages = messages * dir_features * attn_score
        
        # 聚合消息
        aggregated = torch.zeros_like(x)
        aggregated = aggregated.index_add_(0, col, messages)
        
        # 残差连接和更新
        residual = self.residual(x)
        return self.norm(residual + aggregated)

# 6. 优化的图注意力层
class GraphAttentionLayer(nn.Module):
    """优化的图注意力网络层，支持多头注意力"""
    def __init__(self, in_features, out_features, heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        
        # 每个头的输出维度
        self.head_dim = out_features // heads
        
        # 线性变换矩阵
        self.W = nn.Parameter(torch.Tensor(in_features, heads * self.head_dim))
        self.a_src = nn.Parameter(torch.Tensor(heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.Tensor(heads, self.head_dim, 1))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        
        # 激活函数
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x, edge_index):
        # 线性变换
        h = torch.mm(x, self.W).view(-1, self.heads, self.head_dim)
        
        # 计算注意力分数
        src = h[edge_index[0]]
        dst = h[edge_index[1]]
        attn_scores = torch.einsum('nhd,nhd->nh', src, dst) / np.sqrt(self.head_dim)
        
        # 应用LeakyReLU
        attn_scores = self.leakyrelu(attn_scores)
        
        # 创建注意力矩阵
        attn_matrix = torch.zeros((x.size(0), x.size(0), self.heads), 
                                 device=x.device)
        attn_matrix[edge_index[0], edge_index[1]] = attn_scores
        
        # 应用softmax
        attn_matrix = F.softmax(attn_matrix, dim=1)
        
        # 应用dropout
        attn_matrix = F.dropout(attn_matrix, p=self.dropout, training=self.training)
        
        # 消息聚合
        out = torch.einsum('nih,nhd->nhd', attn_matrix, h)
        
        # 合并多头
        if self.concat:
            out = out.view(-1, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)
            
        return F.elu(out)

# 7. 优化的特征提取模型
class ProteinFeatureExtractor(nn.Module):
    """结合MaSIF、SE(3)等变性和GAT的蛋白质特征提取器"""
    def __init__(self, 
                 atom_feature_dim=10,  # 更新为10个特征
                 masif_dim=128,
                 hidden_dim=256,
                 output_dim=512,
                 num_degrees=3,
                 heads=8):
        super().__init__()
        # MaSIF特征提取器
        self.masif_encoder = MaSIFEncoder(in_channels=3, feature_dim=masif_dim)
        self.masif_radius = 12.0
        
        # 初始特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(atom_feature_dim + masif_dim + 3, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # SE(3)等变层
        self.equivariant_layers = nn.ModuleList([
            SE3EquivariantLayer(hidden_dim, hidden_dim, num_degrees) for _ in range(4)
        ])
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, heads=heads) for _ in range(3)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, graph_data):
        # 提取图数据
        x = graph_data['x']
        pos = graph_data['pos']
        edge_index = graph_data['edge_index']
        
        # 批量提取MaSIF局部特征
        masif_features = self.batch_extract_masif(pos)
        
        # 拼接原子特征和MaSIF特征
        atom_features = torch.cat([x, masif_features], dim=1)
        x_fused = self.feature_fusion(atom_features)
        
        # 保存初始特征用于残差连接
        x_init = x_fused.clone()
        
        # 应用SE(3)等变层
        for layer in self.equivariant_layers:
            x_fused = layer(x_fused, pos, edge_index)
        
        # 残差连接
        x_fused = x_fused + x_init
        
        # 应用图注意力层
        for gat_layer in self.gat_layers:
            x_fused = gat_layer(x_fused, edge_index)
        
        # 输出投影
        return self.output_proj(x_fused)
    
    def batch_extract_masif(self, pos):
        """批量提取MaSIF特征以提高效率"""
        batch_size = pos.size(0)
        max_neighbors = 0
        patches = []
        valid_indices = []
        
        # 收集所有局部补丁
        for i in range(batch_size):
            dists = torch.norm(pos - pos[i], dim=1)
            neighbors = torch.nonzero(dists < self.masif_radius, as_tuple=True)[0]
            
            if len(neighbors) > 0:
                patch = pos[neighbors] - pos[i]
                patches.append(patch)
                max_neighbors = max(max_neighbors, len(neighbors))
                valid_indices.append(i)
        
        # 如果没有有效的补丁，返回零向量
        if not patches:
            return torch.zeros(batch_size, self.masif_encoder.conv_net[-1].out_channels + 3, 
                             device=pos.device)
        
        # 填充补丁到相同大小
        padded_patches = []
        for patch in patches:
            pad_size = max_neighbors - patch.size(0)
            padded = F.pad(patch, (0, 0, 0, pad_size), value=0)
            padded_patches.append(padded)
        
        # 批量处理
        patch_tensor = torch.stack(padded_patches)
        features = self.masif_encoder(patch_tensor)
        
        # 创建完整特征向量
        full_features = torch.zeros(batch_size, features.size(1), device=pos.device)
        full_features[valid_indices] = features
        
        return full_features

# ==================== 多尺度特征提取器 ====================
class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器（原子级、残基级、结构域级）"""
    def __init__(self, atom_feature_dim=10, residue_feature_dim=64, 
                 domain_feature_dim=256, output_dim=512):
        super().__init__()
        
        # 原子级特征提取器
        self.atom_extractor = ProteinFeatureExtractor(
            atom_feature_dim=atom_feature_dim,
            output_dim=atom_feature_dim*4
        )
        
        # 残基级特征聚合
        self.residue_agg = nn.Sequential(
            nn.Linear(atom_feature_dim*4, residue_feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(residue_feature_dim)
        )
        
        # 结构域级特征提取
        self.domain_gcn = nn.Sequential(
            nn.Linear(residue_feature_dim, domain_feature_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(domain_feature_dim, domain_feature_dim)
        )
        
        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Linear(atom_feature_dim*4 + residue_feature_dim + domain_feature_dim, 
                     output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, graph_data, residue_mapping):
        """
        graph_data: 原子级图数据
        residue_mapping: 原子到残基的映射关系
        """
        # 原子级特征
        atom_features = self.atom_extractor(graph_data)
        
        # 残基级特征聚合
        residue_features = []
        res_indices = []
        for res_id in np.unique(residue_mapping):
            atom_indices = np.where(residue_mapping == res_id)[0]
            if len(atom_indices) > 0:
                # 残基特征为所属原子特征的均值
                res_feat = atom_features[atom_indices].mean(dim=0)
                residue_features.append(res_feat)
                res_indices.append(res_id)
        
        if not residue_features:
            return torch.zeros(1, self.fusion[0].in_features, device=graph_data['x'].device)
        
        residue_features = torch.stack(residue_features)
        residue_features = self.residue_agg(residue_features)
        
        # 结构域级特征（简化为残基特征的图卷积）
        domain_features = self.domain_gcn(residue_features)
        domain_features = domain_features.mean(dim=0, keepdim=True)
        
        # 多尺度融合
        atom_rep = atom_features.mean(dim=0, keepdim=True)
        res_rep = residue_features.mean(dim=0, keepdim=True)
        full_rep = torch.cat([atom_rep, res_rep, domain_features], dim=1)
        return self.fusion(full_rep).squeeze(0)

# ==================== 特征可视化与解释 ====================
def visualize_features(atom_features, pos, feature_idx=0, output_file="feature_visualization.png"):
    """可视化原子特征分布"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 特征值归一化
    feat = atom_features[:, feature_idx]
    norm_feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
    
    # 绘制原子点（颜色映射特征值）
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=norm_feat, cmap='viridis',
        s=50 * (norm_feat * 0.5 + 0.5),  # 大小随特征值变化
        alpha=0.8
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"Feature {feature_idx}")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Protein Feature Visualization (Feature {feature_idx})")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"特征可视化已保存至 {output_file}")

# ==================== 性能优化 ====================
def batch_load_cif_to_graph(cif_paths, cutoff=5.0, top_k=16, num_workers=4):
    """批量加载CIF文件并构建图（支持多进程）"""
    def process_single_cif(cif_path):
        try:
            graph = load_cif_to_graph(cif_path, cutoff, top_k)
            if graph is not None:
                return graph
            return None
        except Exception as e:
            print(f"处理CIF {cif_path} 失败: {str(e)}")
            return None
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_cif, cif_paths))
    
    # 过滤成功的图数据
    return [graph for graph in results if graph is not None]

# ==================== 完整处理流程 ====================
def advanced_extract_protein_features(cif_path, device='auto', output_dim=512, visualize=False, verbose=True):
    """高级蛋白质特征提取函数（含多尺度特征）"""
    print(f"正在处理CIF文件: {cif_path}")
    
    # 1. 解析CIF并构建图
    atoms = parse_cif(cif_path)
    if not atoms:
        print("错误：CIF文件无有效原子")
        return None
    
    if verbose:
        print(f"从CIF文件中提取了 {len(atoms)} 个原子")
    
    # 计算新增特征
    solvent_acc = calculate_solvent_accessibility(atoms)
    ss_codes = calculate_secondary_structure(atoms)
    phi_angles, psi_angles = calculate_dihedral_angles(atoms)
    
    # 2. 构建图数据（包含新增特征）
    graph_data = load_cif_to_graph(cif_path, verbose=verbose)
    if graph_data is None:
        return None
    
    # 扩展原子特征
    atom_features = graph_data['x'].cpu().numpy()
    
    # 处理二面角特征长度不匹配问题（简化处理）
    if len(phi_angles) < len(atoms):
        phi_angles = np.pad(phi_angles, (0, len(atoms) - len(phi_angles)), 'constant', constant_values=0.5)
    if len(psi_angles) < len(atoms):
        psi_angles = np.pad(psi_angles, (0, len(atoms) - len(psi_angles)), 'constant', constant_values=0.5)
    
    # 合并新增特征
    new_features = np.column_stack([
        solvent_acc,
        ss_codes,
        phi_angles[:len(atoms)],
        psi_angles[:len(atoms)]
    ])
    
    # 确保特征维度匹配
    if new_features.shape[0] < atom_features.shape[0]:
        new_features = np.pad(new_features, ((0, atom_features.shape[0] - new_features.shape[0]), (0, 0)), 
                             'constant', constant_values=0)
    
    # 更新图数据中的特征
    graph_data['x'] = torch.tensor(np.hstack([atom_features, new_features]), dtype=torch.float)
    
    # 3. 准备原子到残基的映射
    residue_mapping = np.array([atom['res_id'] for atom in atoms])
    
    # 4. 自动选择设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"使用设备: {device}")
    
    # 5. 初始化模型
    model = MultiScaleFeatureExtractor(
        atom_feature_dim=10,  # 6个原始特征 + 4个新增特征
        output_dim=output_dim
    ).to(device)
    
    # 6. 提取特征
    model.eval()
    with torch.no_grad():
        # 将图数据移动到设备
        for key in graph_data:
            if isinstance(graph_data[key], torch.Tensor):
                graph_data[key] = graph_data[key].to(device)
        
        # 提取多尺度特征
        features = model(graph_data, residue_mapping)
    
    print(f"成功提取特征，维度: {features.shape}")
    
    # 7. 可视化特征（如果需要）
    if visualize and isinstance(graph_data['x'], torch.Tensor):
        atom_features = graph_data['x'].cpu().numpy()
        visualize_features(atom_features, graph_data['pos'].cpu().numpy())
    
    return features.cpu().numpy()

# ==================== 使用示例 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='蛋白质全原子特征提取器')
    parser.add_argument('--cif_file', type=str, help='输入CIF文件路径')
    parser.add_argument('--output_dim', type=int, default=512, help='输出特征维度')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (cpu, cuda, auto)')
    parser.add_argument('--visualize', action='store_true', help='是否可视化特征')
    args = parser.parse_args()

    
    print(f"开始处理CIF文件: {args.cif_file} (大小: {file_size/1024:.2f} KB)")
    
    total_start = time.time()
    features = advanced_extract_protein_features(
        args.cif_file, 
        device=args.device,
        output_dim=args.output_dim,
        visualize=args.visualize
    )
    
    if features is not None:
        total_time = time.time() - total_start
        print(f"成功提取到特征: shape={features.shape}")
        print(f"总处理时间: {total_time:.2f}秒")
        
        # 保存特征
        output_file = os.path.splitext(os.path.basename(args.cif_file))[0] + "_features.npy"
        np.save(output_file, features)
        print(f"特征已保存到 {output_file}")
    else:
        print("特征提取失败")