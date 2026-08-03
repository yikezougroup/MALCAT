import pandas as pd
import numpy as np
from rdkit import Chem
import threading

# 定义必要的类
class Iterator(object):
    """Abstract base class for data iterators."""
    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class SmilesIterator(Iterator):
    """Iterator yielding data from a SMILES array."""
    def __init__(self, x, y, smiles_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32
                 ):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.smiles_data_generator = smiles_data_generator
        self.dtype = dtype
        super(SmilesIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + [ self.smiles_data_generator.pad, self.smiles_data_generator._charlen]), dtype=self.dtype)
        for i, j in enumerate(index_array):
            smiles = self.x[j:j+1]
            x = self.smiles_data_generator.transform(smiles)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer"""
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        
    def fit(self, smiles, extra_chars=[], extra_pad = 5):
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles  # 如果无法解析，返回原始SMILES
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        if self.leftpad:
            for i, ss in enumerate(smiles):
                if self.enumerate: 
                    ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j, c in enumerate(ss):
                    if c in self._char_to_int:  # 确保字符在字符集中
                        one_hot[i, j+diff, self._char_to_int[c]] = 1
            return one_hot
        else:
            for i, ss in enumerate(smiles):
                if self.enumerate: 
                    ss = self.randomize_smiles(ss)
                for j, c in enumerate(ss):
                    if c in self._char_to_int:
                        one_hot[i, j, self._char_to_int[c]] = 1
            return one_hot
      
    def reverse_transform(self, vect):
        smiles = []
        for v in vect:
            v = v[v.sum(axis=1) == 1]
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)

# 修改后的增强函数
def augment_smiles_csv(input_file, output_file, num_variants=5):
    # 读取CSV文件（包含表头）
    df = pd.read_csv(input_file)
    
    # 初始化增强器
    sm_en = SmilesEnumerator(
        canonical=False,  # 生成随机变体
        enum=True,        # 启用枚举
        isomericSmiles=True
    )
    
    # 自动适配字符集和填充长度
    sm_en.fit(np.array(df['compound_iso_smiles']), extra_chars=["\\"])
    
    # 创建新DataFrame存储增强数据
    augmented_data = []
    
    # 遍历原始数据
    for _, row in df.iterrows():
        original_smiles = row['compound_iso_smiles']
        target_sequence = row['target_sequence']
        affinity = row['affinity']
        protein_3d_path = row['protein_3d_path']
        
        # 保留原始数据
        augmented_data.append({
            'compound_iso_smiles': original_smiles,
            'target_sequence': target_sequence,
            'affinity': affinity,
            'protein_3d_path': protein_3d_path
        })
        
        # 生成多个变体
        valid_variants = 0
        for _ in range(num_variants):
            try:
                augmented_smiles = sm_en.randomize_smiles(original_smiles)
                # 检查增强后的SMILES是否有效（可选）
                if Chem.MolFromSmiles(augmented_smiles) is not None:
                    augmented_data.append({
                        'compound_iso_smiles': augmented_smiles,
                        'target_sequence': target_sequence,
                        'affinity': affinity,
                        'protein_3d_path': protein_3d_path
                    })
                    valid_variants += 1
            except Exception as e:
                print(f"Error processing {original_smiles}: {str(e)}")
        
        # 如果未能生成足够的有效变体，复制原始数据填充
        while valid_variants < num_variants:
            augmented_data.append({
                'compound_iso_smiles': original_smiles,
                'target_sequence': target_sequence,
                'affinity': affinity,
                'protein_3d_path': protein_3d_path
            })
            valid_variants += 1
    
    # 创建新DataFrame并保存
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_file, index=False)
    
    original_count = len(df)
    augmented_count = len(augmented_df)
    print(f"数据增强完成！原始数据: {original_count} 行 → 增强后: {augmented_count} 行")
    print(f"增强比例: 1:{augmented_count/original_count:.1f}")

# 使用示例
if __name__ == "__main__":
    augment_smiles_csv(
        input_file="cyclicpeptide/data/allcold/raw/data_train.csv",
        output_file="cyclicpeptide/data/allcold/raw/data_train2.csv",
        num_variants=10
    )