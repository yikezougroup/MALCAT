import torch
from torch_geometric.data import InMemoryDataset

class GNNDataset(InMemoryDataset):
    '''
    transformer:函数用于数据对象被加载使用之前进行的动态转换(一般用于数据增强)
    pre_transformer:预处理过程
    pre_filter:过滤
    '''

    def __init__(self, root, train=True, transformer=None, pre_transformer=None, pre_filter=None):
        super().__init__(root, transformer, pre_transformer, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            self.data, self.slices = torch.load(self.processed_paths[1], weights_only=False)

    @property
    def raw_file_names(self):
        # 假设根目录下只有一个csv文件
        raw_files = [f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]
        return raw_files

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        pass

    def _download(self):
        pass


    def process(self):
        pass

if __name__ == "__main__":
    GNNDataset('data/peptide')


