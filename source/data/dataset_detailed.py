import os
import typing as tp

# import albumentations as alb
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset


class DFRow(BaseModel):
    dataset: str
    wsi_id: str
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

    # extract loss weights and checks for each class
    #     unknown: -1
    #     absent: 0
    #     present: 1
    #     predominant pattern: 2

    @property
    def labels_luad(self) -> np.ndarray:
        return np.array([self.LUAD])
    
    @property
    def labels_lusc(self) -> np.ndarray:
        return np.array([self.LUSC])
    
    @property
    def labels_benign(self) -> np.ndarray:
        return np.array([self.Benign])

    @property
    def labels_luad_lusc(self) -> np.ndarray:
        return np.array([self.LUAD, self.LUSC])

    @property
    def labels_luad_lusc_benign(self) -> np.ndarray:
        return np.array([self.LUAD, self.LUSC, self.Benign])

    @property
    def labels_luad_lusc_benign_luad_patterns(self) -> np.ndarray:
        return np.array([
            self.LUAD, self.LUSC, self.Benign,
            self.LUAD_acinar, self.LUAD_lepidic, self.LUAD_micropapillary, self.LUAD_papillary, self.LUAD_solid
        ])

    @property
    def luad_patterns(self) -> np.ndarray:
        return np.array([
            self.LUAD_acinar, self.LUAD_lepidic, self.LUAD_micropapillary, self.LUAD_papillary, self.LUAD_solid
        ])


class LungSubtypingDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            feats_dir: str,
            feats_size: int,
            num_classes: int,
            label_group: str,
            min_mask_ratio: float,
            subsample_num_patches: int,  # Add subsample_num_patches parameter
            device: torch.device,  # Add device parameter
    ):
        self._df = df
        self.feats_dir = feats_dir
        self.feats_size = feats_size
        self.num_classes = num_classes
        self.label_group = label_group
        self.min_mask_ratio = min_mask_ratio
        self.subsample_num_patches = subsample_num_patches  # Store subsample_num_patches
        self.device = device  # Store device

        self._check_files_exist()

    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, np.ndarray]:
        row = DFRow(**self._df.iloc[idx])
        pt_features_path, pt_positions_path, npy_indexes_path, npy_patch_status_path = (
            self._get_file_paths(row)
        )

        # no shuffling of the features order
        #   MIL models are invariant to the order of the features
        #   ViT models have positional encodings that take care of that
        feats = torch.load(pt_features_path, map_location=self.device)
        assert feats.shape[1] == self.feats_size, f"Expected feats.shape[1] == {self.feats_size}, got {feats.shape[1]} instead."
        assert feats.dtype == torch.float32, f"Expected torch.float32, got {feats.dtype}"

        patch_locs = torch.load(pt_positions_path, map_location=self.device).to(dtype=torch.int)
        assert patch_locs.shape[1] in [2, 4], f"Expected patch_locs.shape[1] to be either 2 or 4, got {patch_locs.shape[1]} instead."
        assert patch_locs.shape[0] == feats.shape[0], f"Expected patch_locs.shape[0] == feats.shape[0], got {patch_locs.shape[0]} != {feats.shape[0]} instead."


        # TODO: make less manual - now there is an if statement dependent on row.Benign value
        # patch status is
        #   -1 for unknown
        #    0 for benign
        #    1 for diagnostic
        if row.Benign == 1:
            patch_status_array = np.zeros(feats.shape[0])
            # print("Benign case, setting patch_status_array to 0 for all patches.")
        elif not os.path.exists(npy_patch_status_path):
            patch_status_array = -1 * np.ones(feats.shape[0])
            # print(f"File does not exist. Setting patch_status_array to -1 for all patches. File path: {npy_patch_status_path}")
        else:
            patch_status_array = np.load(npy_patch_status_path)
            assert len(patch_status_array) == feats.shape[0], (f"Expected len(status_array) == feats.shape[0], got {len(patch_status_array)} != {feats.shape[0]} instead.")
            # print(f"Not a benign case, loading patch_status_array from {npy_patch_status_path}")
            # print("Diagnostic tiles:", (patch_status_array == 1).sum())
            # print("Benign tiles:", (patch_status_array == 0).sum())
            # print("Unknown tiles:", (patch_status_array == -1).sum())

        # use compute_weights_mask to
        #   1. convert patch_status_array to binary labels 0 or 1 (all unknonws go to 0)
        #   2. create a weight mask for the labels
        #       unkowns get weight 0
        #       known get weight 1
        patch_status, patch_status_weight_mask = self.compute_weights_mask(patch_status_array)
        # they should be the same type as labels and label_weight_mask
        patch_status = torch.from_numpy(patch_status).to(dtype=torch.float32, device=self.device)
        patch_status_weight_mask = torch.from_numpy(patch_status_weight_mask).to(dtype=torch.float32, device=self.device)


        # Subset feats and patch_locs using npy_indexes without moving them away from GPU
        if self.min_mask_ratio > 0:
            subset_indexes = np.load(npy_indexes_path)
            subset_indexes_tensor = torch.from_numpy(subset_indexes).to(self.device)
            feats = feats[subset_indexes_tensor]
            patch_locs = patch_locs[subset_indexes_tensor]
            patch_status = patch_status[subset_indexes_tensor]
            patch_status_weight_mask = patch_status_weight_mask[subset_indexes_tensor]

        # Subsample patches if subsample_num_patches is set
        if self.subsample_num_patches > 0 and feats.shape[0] > self.subsample_num_patches:
            subset_indexes = np.random.choice(feats.shape[0], size=self.subsample_num_patches, replace=False)
            subset_indexes_tensor = torch.from_numpy(subset_indexes).to(self.device)
            feats = feats[subset_indexes_tensor]
            patch_locs = patch_locs[subset_indexes_tensor]
            patch_status = patch_status[subset_indexes_tensor]
            patch_status_weight_mask = patch_status_weight_mask[subset_indexes_tensor]

        # add options for different label granularity
        if self.label_group == 'luad':
            labels = row.labels_luad
        elif self.label_group == 'lusc':
            labels = row.labels_lusc
        elif self.label_group == 'benign':
            labels = row.labels_benign
        elif self.label_group == 'luad_lusc':
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

        return (
            feats,
            torch.tensor(labels, dtype=torch.float32, device=self.device),  # Load onto device
            torch.tensor(label_weight_mask, dtype=torch.float32, device=self.device),  # Load onto device
            patch_locs,
            patch_status,
            patch_status_weight_mask,
        )

    def __len__(self) -> int:
        return len(self._df)

    def _get_file_paths(self, row: DFRow) -> tp.Tuple[str, str]:
        pt_features_path = os.path.join(
            "datasets",
            row.dataset,
            self.feats_dir,
            f"{row.wsi_id}.features.pt",
        )
        # self.feats_dir = 'features/UNI/imagenet/patch_224_0.5_mpp'
        # 'datasets/{row.dataset}/{self.feats_dir}/{row.wsi_id}.features.pt'
        pt_positions_path = pt_features_path.replace(".features.pt", ".positions.pt")

        # npy_indexes_path = wsi_masked_positions_intersections/{row.dataset}/UNI/imagenet/patch_224_0.5_mpp/min_mask_ratio_0.0/1064_19_5H.indexes.npy
        assert self.feats_dir.startswith("features/"), f"Expected self.feats_dir to start with 'features/', got {self.feats_dir} instead."
        npy_indexes_path = os.path.join(
            "wsi_masked_positions_intersections",
            row.dataset,
            self.feats_dir.replace("features/", ""),
            f"min_mask_ratio_{self.min_mask_ratio}",
            f"{row.wsi_id}.indexes.npy",
        )

        # wsi_annotated_coords/ouh_batch1_20x/simclr-tcga-lung_resnet18-10x/imagenet/patch_224_1.0_mpp/1064_19_5H/status_array.npy
        npy_patch_status_path = os.path.join(
            "wsi_annotated_coords",
            row.dataset,
            self.feats_dir.replace("features/", ""),
            row.wsi_id,
            "status_array.npy",
        )

        return (
            pt_features_path,
            pt_positions_path,
            npy_indexes_path,
            npy_patch_status_path,
        )

    def _check_files_exist(self):
        missing_files = []
        for idx in range(len(self._df)):
            row = DFRow(**self._df.iloc[idx])
            pt_features_path, pt_positions_path, npy_indexes_path, npy_patch_status_path = self._get_file_paths(row)

            if not os.path.exists(pt_features_path):
                missing_files.append(pt_features_path)
            if not os.path.exists(pt_positions_path):
                missing_files.append(pt_positions_path)
            if not os.path.exists(npy_indexes_path) and self.min_mask_ratio > 0:
                missing_files.append(npy_indexes_path)
            # we do not check if npy_patch_status_path exists, as it is optional
        if missing_files:
            raise FileNotFoundError(
                f"The following {len(missing_files)} files are missing: {missing_files}"
            )

    def compute_weights_mask(self, labels: np.ndarray) -> np.ndarray:

        # 1, 0, 0, -1
        #   labels: 1, 0, 0, 0
        #   weight: 1, 1, 1, 0

        binary_labels = (labels > 0).astype(int)
        label_weights = np.ones_like(labels)
        label_weights[labels == -1] = 0
        return binary_labels, label_weights


def pad_1D_collate(batch):
    bags, labels, label_weight_mask, patch_locs, patch_status, patch_status_weight_mask = zip(*batch)
    bag_lens = [len(x) for x in bags]
    max_len = max(bag_lens)
    # pad bags to max length with zeros
    zeros_for_bags = torch.zeros(bags[0][0].shape, device=bags[0].device)  # torch.zeros(embedding size)
    pads_for_bags = [zeros_for_bags.repeat(max_len - bag_len, 1) for bag_len in bag_lens]
    bags_padded = [
        torch.concat((bag, pad), 0)
        for bag, pad in zip(bags, pads_for_bags)
    ]  # batch size, max bag len, embedding size
    
    # pad patch locs to max length - padded with [-1, -1] so there is no chance to clash with patch [0, 0]
    minusones_for_patch_locs = -torch.ones(patch_locs[0][0].shape, device=patch_locs[0].device)  # ensure same device
    pads_for_patch_locs = [minusones_for_patch_locs.repeat(max_len - bag_len, 1) for bag_len in bag_lens]
    patch_locs_padded = [
        torch.concat((patch_loc, pad), 0)
        for patch_loc, pad in zip(patch_locs, pads_for_patch_locs)
    ]
    # batch size, max bag len, patch loc size

    # pad patch_status and patch_status_weight_mask to max length with zeros
    #   padding patch_status_weight_mask with 0s means it does not matter what we use to pad patch status with, bu we keep 0 for consistency with `compute_weights_mask`
    #   patch_status is a list of 1d tensors each of the length of the bag
    #   patch_status_weight_mask is a list of 1d tensors each of the length of the bag
    patch_status_padded = [
        torch.cat((status, torch.zeros(max_len - len(status), device=status.device)), 0)
        for status in patch_status
    ]
    patch_status_weight_mask_padded = [
        torch.cat((mask, torch.zeros(max_len - len(mask), device=mask.device)), 0)
        for mask in patch_status_weight_mask
    ]


    # return
    return (
        torch.stack((bags_padded), 0),
        torch.stack(labels, 0),
        torch.stack(label_weight_mask, 0),
        bag_lens,
        torch.stack(patch_locs_padded, 0),
        torch.stack(patch_status_padded, 0),
        torch.stack(patch_status_weight_mask_padded, 0),
    )


class LungSubtypingSlideEmbeddingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feats_dir: str,
        feats_size: int,
        num_classes: int,
        label_group: str,
        # min_mask_ratio: float,
        device: torch.device,  # Add device parameter
    ):
        self._df = df
        self.feats_dir = feats_dir
        self.feats_size = feats_size
        self.num_classes = num_classes
        self.label_group = label_group
        # self.min_mask_ratio = min_mask_ratio
        self.device = device  # Store device

        self._check_files_exist()

    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, np.ndarray]:
        row = DFRow(**self._df.iloc[idx])
        pt_features_path = self._get_file_path(row)

        # no shuffling of the features order
        #   MIL models are invariant to the order of the features
        #   ViT models have positional encodings that take care of that
        feats = torch.load(pt_features_path, map_location=self.device)
        assert feats.shape == (1, self.feats_size), (
            f"Expected feats.shape[1] == {self.feats_size}, got {feats.shape[1]} instead."
        )
        assert feats.dtype == torch.float32, (
            f"Expected torch.float32, got {feats.dtype}"
        )
        feats = feats.squeeze(dim=0)
        assert feats.shape == (self.feats_size,), "Expected feats.shape == (self.feats_size,)"

        # add options for different label granularity
        if self.label_group == "luad":
            labels = row.labels_luad
        elif self.label_group == "lusc":
            labels = row.labels_lusc
        elif self.label_group == "benign":
            labels = row.labels_benign
        elif self.label_group == "luad_lusc":
            labels = row.labels_luad_lusc
        elif self.label_group == "luad_lusc_benign":
            labels = row.labels_luad_lusc_benign
        elif self.label_group == "luad_lusc_benign_luad_patterns":
            labels = row.labels_luad_lusc_benign_luad_patterns
        elif self.label_group == "luad_patterns":
            labels = row.luad_patterns
        else:
            raise NotImplementedError(
                f"Unknown label group: {self.label_group}. Select one of the following: 'luad_lusc', "
                f"'luad_lusc_benign', 'luad_lusc_benign_luad_patterns'."
            )
        assert len(labels) == self.num_classes, (
            f"len(labels) should be {self.num_classes}. Got len(labels)={len(labels)} instead. Check if self.num_classes is correct."
        )

        labels, label_weight_mask = self.compute_weights_mask(labels)

        return (
            feats,
            torch.tensor(labels, dtype=torch.float32, device=self.device),  # Load onto device
            torch.tensor(label_weight_mask, dtype=torch.float32, device=self.device),  # Load onto device
        )

    def __len__(self) -> int:
        return len(self._df)

    def _get_file_path(self, row: DFRow) -> tp.Tuple[str, str]:
        if "PRISM" in self.feats_dir:
            pt_features_path = os.path.join(
                "datasets",
                row.dataset,
                self.feats_dir,
                f"{row.wsi_id}.image_embedding.pt",
            )
        elif "prov-gigapath" in self.feats_dir:
            pt_features_path = os.path.join(
                "datasets",
                row.dataset,
                self.feats_dir,
                f"{row.wsi_id}.last_layer_embed.pt",
            )
        else:
            raise ValueError(f"Unknown feats_dir: {self.feats_dir}.")
        
        return pt_features_path
    
    def _check_files_exist(self):
        missing_files = []
        for idx in range(len(self._df)):
            row = DFRow(**self._df.iloc[idx])
            pt_features_path = self._get_file_path(row)
            if not os.path.exists(pt_features_path):
                missing_files.append(pt_features_path)
        if missing_files:
            raise FileNotFoundError(
                f"The following {len(missing_files)} files are missing: {missing_files}"
            )

    def compute_weights_mask(self, labels: np.ndarray) -> np.ndarray:
        # 1, 0, 0, -1
        #   labels: 1, 0, 0, 0
        #   weight: 1, 1, 1, 0

        binary_labels = (labels > 0).astype(int)
        label_weights = np.ones_like(labels)
        label_weights[labels == -1] = 0
        return binary_labels, label_weights
