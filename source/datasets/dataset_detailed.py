import typing as tp

# import albumentations as albu
import h5py
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset


class DFRow(BaseModel):
    features_csv_file_path: str
    h5_file_path: str  # same features as in csv, but in hdf5 format; locations recorded
    pt_file_path: str  # same features as in csv, but in pytorch format; locations not recorded
    patient_id: str
    source: str
    LUAD: int
    LUSC: int
    Benign: int
    LUAD_acinar: int
    LUAD_lepidic: int
    LUAD_micropapillary: int
    LUAD_papillary: int
    LUAD_solid: int
    """
    Represents a row in a DataFrame with detailed information about a dataset.

    Attributes:
        features_csv_file_path (str): The file path to the CSV file containing the features and possibly patch locations as the last 2 columns.
        h5_file_path (str): The file path to the HDF5 file containing the features and tile coordinates in HDF5 format.
        pt_file_path (str): The file path to the PyTorch file containing the features in PyTorch format.
        patient_id (str): The ID of the patient.
        source (str): The source of the dataset.
        LUAD (int): The label for LUAD.
        LUSC (int): The label for LUSC.
        Benign (int): The label for Benign.
        LUAD_acinar (int): The label for LUAD acinar.
        LUAD_lepidic (int): The label for LUAD lepidic.
        LUAD_micropapillary (int): The label for LUAD micropapillary.
        LUAD_papillary (int): The label for LUAD papillary.
        LUAD_solid (int): The label for LUAD solid.

    Label conventions:
    - unknown: -1
    - absent: 0
    - present: 1
    - predominant pattern: 2
    """

    @property
    def labels_luad_lusc(self) -> np.ndarray:
        """
        Get the labels for LUAD and LUSC as a NumPy array.

        Returns:
            np.ndarray: The labels for LUAD and LUSC.
        """
        return np.array([self.LUAD, self.LUSC])

    @property
    def labels_luad_lusc_benign(self) -> np.ndarray:
        """
        Get the labels for LUAD, LUSC, and Benign as a NumPy array.

        Returns:
            np.ndarray: The labels for LUAD, LUSC, and Benign.
        """
        return np.array([self.LUAD, self.LUSC, self.Benign])

    @property
    def labels_luad_lusc_benign_luad_patterns(self) -> np.ndarray:
        """
        Get the labels for LUAD, LUSC, Benign, and LUAD patterns as a NumPy array.

        Returns:
            np.ndarray: The labels for LUAD, LUSC, Benign, and LUAD patterns.
            Order: LUAD, LUSC, Benign, acinar, lepidic, micropapillary, papillary, solid.
        """
        return np.array([
            self.LUAD, self.LUSC, self.Benign,
            self.LUAD_acinar, self.LUAD_lepidic, self.LUAD_micropapillary, self.LUAD_papillary, self.LUAD_solid
        ])

    @property
    def luad_patterns(self) -> np.ndarray:
        """
        Get the LUAD patterns as a NumPy array.

        Returns:
            np.ndarray: The LUAD patterns: acinar, lepidic, micropapillary, papillary, and solid.
        """
        return np.array([
            self.LUAD_acinar, self.LUAD_lepidic, self.LUAD_micropapillary, self.LUAD_papillary, self.LUAD_solid
        ])


class LungSubtypingDataset(Dataset):
    """
    Dataset class for lung subtyping.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset information.
        feats_size (int): The size of the features.
        patch_loc_size (int): The size of the patch locations.
        num_classes (int): The number of classes.
        label_group (str): The label group.
        device (torch.device): The device to use.

    Attributes:
        _df (pd.DataFrame): The input DataFrame containing the dataset information.
        feats_size (int): The size of the features.
        patch_loc_size (int): The size of the patch locations.
        num_classes (int): The number of classes.
        label_group (str): The label group.
        device (torch.device): The device to use.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            feats_size: int,
            patch_loc_size: int,
            num_classes: int,
            label_group: str,
            device: torch.device,
    ):
        self._df = df
        self.feats_size = feats_size
        self.patch_loc_size = patch_loc_size
        self.num_classes = num_classes
        self.label_group = label_group
        self.device = device

    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the features, labels, label weight mask, and patch locations.
        """
        row = DFRow(**self._df.iloc[idx])

        # no shuffling of the features order
        #   MIL models are invariant to the order of the features
        #   ViT models have positional encodings that take care of that
        feats = torch.load(row.pt_file_path, map_location=self.device)
        feats = feats[:, :self.feats_size]
        assert feats.dtype == torch.float32, f"Expected torch.float32, got {feats.dtype}"

        with h5py.File(row.h5_file_path, "r") as h5_file:
            patch_locs = h5_file["col_row_loc_at_smallest_downsample_level"][:].astype(int)

        # add options for different label granularity
        if self.label_group == 'luad_lusc':
            labels = row.labels_luad_lusc
        elif self.label_group == 'luad_lusc_benign':
            labels = row.labels_luad_lusc_benign
        elif self.label_group == 'luad_lusc_benign_luad_patterns':
            labels = row.labels_luad_lusc_benign_luad_patterns
        elif self.label_group == 'luad_patterns':
            labels = row.luad_patterns
        else:
            raise NotImplementedError(
                f"Unknown label group: {self.label_group}. Select one of the following: 'luad_lusc', "
                f"'luad_lusc_benign', 'luad_lusc_benign_luad_patterns'.")
        assert len(labels) == self.num_classes, \
            f'len(labels) should be {self.num_classes}. Got len(labels)={len(labels)} instead. Check if self.num_classes is correct.'

        labels, label_weight_mask = self.compute_weights_mask(labels)

        return feats, torch.tensor(labels, dtype=torch.float32), torch.tensor(
            label_weight_mask, dtype=torch.float32), torch.tensor(patch_locs, dtype=torch.int)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._df)

    def compute_weights_mask(self, labels: np.ndarray) -> np.ndarray:
        """
        Compute the label weights mask.

        Args:
            labels (np.ndarray): The input labels.

        Returns:
            np.ndarray: The binary labels and label weights mask.

        Examples:
            >>> dataset = LungSubtypingDataset(...)
            >>> labels = np.array([1, 0, 1]) # all labels known
            >>> dataset.compute_weights_mask(labels)
            (array([1, 0, 0, 1]), array([1., 1., 1., 1.]))

            >>> dataset = LungSubtypingDataset(...)
            >>> labels = np.array([1, 0, 0, -1]) # last label unknown
            >>> dataset.compute_weights_mask(labels)
            (array([1, 0, 0, 0]), array([1., 1., 1., 0.]))

        """
        binary_labels = (labels > 0).astype(int)
        label_weights = np.ones_like(labels)
        label_weights[labels == -1] = 0
        return binary_labels, label_weights


def pad_1D_collate(batch):
    """
    Collates a batch of data by padding bags and patch locations to the maximum length.

    Args:
        batch (list): A list of tuples containing bags, labels, label weight masks, and patch locations.

    Returns:
        tuple: A tuple containing the padded bags, labels, label weight masks, bag lengths, and padded patch locations.
    """
    bags, labels, label_weight_mask, patch_locs = zip(*batch)
    bag_lens = [len(x) for x in bags]
    max_len = max(bag_lens)
    # pad bags to max length with zeros
    zeros_for_bags = torch.zeros(bags[0][0].shape, device=bags[0].device)  # torch.zeros(embedding size)
    pads_for_bags = [zeros_for_bags.repeat(max_len - bag_len, 1) for bag_len in bag_lens]
    bags_padded = [torch.concat((bag, pad), 0) for bag, pad in
                   zip(bags, pads_for_bags)]  # batch size, max bag len, embedding size
    # pad patch locs to max length - padded with [-1, -1] so there is no chance to clash with patch [0, 0]
    minusones_for_patch_locs = -torch.ones(patch_locs[0][0].shape)  # alternatively can be torch.zeros(patch loc size)
    pads_for_patch_locs = [minusones_for_patch_locs.repeat(max_len - bag_len, 1) for bag_len in bag_lens]
    patch_locs_padded = [torch.concat((patch_loc, pad), 0)
                         for patch_loc, pad in zip(patch_locs, pads_for_patch_locs)]
    # batch size, max bag len, patch loc size

    # return
    return (
        torch.stack((bags_padded), 0),
        torch.stack(labels, 0),
        torch.stack(label_weight_mask, 0),
        bag_lens,
        torch.stack(patch_locs_padded, 0)
    )
