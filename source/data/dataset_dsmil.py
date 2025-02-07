import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset


class DSMILDataset(Dataset):
    def __init__(self,
                 csv_file_df: pd.DataFrame,
                 feats_path_col_name: str,
                 feats_size: int,
                 patch_loc_size: int,
                 label_col_name: str,
                 num_classes: int,
                 ):
        self.csv_file_df = csv_file_df
        self.feats_path_col_name = feats_path_col_name
        self.feats_size = feats_size
        self.patch_loc_size = patch_loc_size
        self.label_col_name = label_col_name
        self.num_classes = num_classes

    def __len__(self):
        return len(self.csv_file_df)

    def __getitem__(self, idx):
        bag_record = self.csv_file_df.iloc[idx]
        feats_csv_path = bag_record[self.feats_path_col_name]
        feats_df = pd.read_csv(feats_csv_path)
        feats = shuffle(feats_df).reset_index(drop=True)

        assert feats.shape[1] >= self.feats_size + self.patch_loc_size, \
            (f'feats.shape[1] should be >= {self.feats_size + self.patch_loc_size}. '
             f'Got feats.shape={feats.shape} and '
             f'self.feats_size={self.feats_size}, self.patch_loc_size={self.patch_loc_size} instead. '
             f'Check if self.feats_size and self.patch_loc_size are correct.')

        if self.patch_loc_size <= 0:
            patch_locs = np.empty((feats.shape[0], 0))
        else:
            patch_locs = feats.to_numpy()[:, -self.patch_loc_size:].astype(int)
        feats = feats.to_numpy()[:, :self.feats_size]

        label = np.zeros(self.num_classes)
        if self.num_classes == 1:
            label[0] = bag_record[self.label_col_name]
        else:
            if int(bag_record[self.label_col_name]) <= (len(label) - 1):
                label[int(bag_record[self.label_col_name])] = 1

        return torch.tensor(label, dtype=torch.float32), torch.tensor(feats, dtype=torch.float32), torch.tensor(
            patch_locs, dtype=torch.int)


if __name__ == '__main__':
    csv_file_df = ...  # Assuming you have the data frame ready
    kwargs = {
        'feats_path_col_name': ...,
        'feats_size': ...,
        'patch_loc_size': ...,
        'label_col_name': ...,
        'num_classes': ...
    }
    dataset = DSMILDataset(csv_file_df, **kwargs)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
