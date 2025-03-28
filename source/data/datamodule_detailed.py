import typing as tp

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from configs.config import Config
# from src.constants import DATA_PATH, ANNOTATIONS_PATH
from source.data.dataset_detailed import (
    LungSubtypingDataset,
    LungSubtypingSlideEmbeddingDataset,
    pad_1D_collate,
)
from source.stratified_patient_split import split_dataset_by_patient

# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class LungSubtypingDM(LightningDataModule):
    def __init__(self, config: Config, device):
        super().__init__()

        self.device = device

        self._batch_size = config.batch_size
        self._eval_batch_size_size_proportion = config.eval_batch_size_size_proportion
        # self._n_workers = config.n_workers

        self._feats_dir = config.feats_dir
        self._min_mask_ratio = config.min_mask_ratio
        self._feats_size = config.feats_size
        self._num_classes = config.num_classes
        self._label_group = config.label_group

        self._train_test_mode = config.train_test_mode
        self._trainval_dataset_csv_path = config.trainval_dataset_csv_path
        self._test_dataset_csv_path = config.test_dataset_csv_path

        self._split = config.split
        self._patient_id_col_name = config.patient_id_col_name

        self._subsample_num_patches = config.subsample_num_patches

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: tp.Optional[str] = None):

        if "train" in self._train_test_mode:
            train_val_dataset_bags_df = pd.read_csv(self._trainval_dataset_csv_path)

            train_bags_df, valid_bags_df = split_dataset_by_patient(
                dataframe=train_val_dataset_bags_df,
                groups_col_name=self._patient_id_col_name,
                feature_col_names=['LUAD', 'LUSC'],
                # TODO: unify this and the config label_group OR make another parameter for this
                test_size=self._split,
                random_state=42,
            )

            self.train_dataset = LungSubtypingDataset(
                train_bags_df,
                feats_dir=self._feats_dir,
                feats_size=self._feats_size,
                num_classes=self._num_classes,
                label_group=self._label_group,
                min_mask_ratio=self._min_mask_ratio,
                subsample_num_patches=self._subsample_num_patches,
                device=self.device,
            )
            self.val_dataset = LungSubtypingDataset(
                valid_bags_df,
                feats_dir=self._feats_dir,
                feats_size=self._feats_size,
                num_classes=self._num_classes,
                label_group=self._label_group,
                min_mask_ratio=self._min_mask_ratio,
                subsample_num_patches=-1,  # Do not subsample during validation
                device=self.device,
            )

        if "test" in self._train_test_mode:
            test_dataset_bags_df = pd.read_csv(self._test_dataset_csv_path)
            self.test_dataset = LungSubtypingDataset(
                test_dataset_bags_df,
                feats_dir=self._feats_dir,
                feats_size=self._feats_size,
                num_classes=self._num_classes,
                label_group=self._label_group,
                min_mask_ratio=self._min_mask_ratio,
                subsample_num_patches=-1,  # Do not subsample during testing
                device=self.device,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            # num_workers=self._n_workers,
            collate_fn=pad_1D_collate,
            shuffle=True,
            # pin_memory=True,
            # drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=max(int(self._batch_size * self._eval_batch_size_size_proportion), 1),
            # num_workers=self._n_workers,
            collate_fn=pad_1D_collate,
            shuffle=False,
            # pin_memory=True,
            # drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=max(int(self._batch_size * self._eval_batch_size_size_proportion), 1),
            # num_workers=self._n_workers,
            collate_fn=pad_1D_collate,
            shuffle=False,
            # pin_memory=True,
            # drop_last=False,
        )


class LungSubtypingSlideEmbeddingDM(LightningDataModule):
    def __init__(self, config: Config, device):
        super().__init__()

        self.device = device

        self._batch_size = config.batch_size
        self._eval_batch_size_size_proportion = config.eval_batch_size_size_proportion
        # self._n_workers = config.n_workers

        self._feats_dir = config.feats_dir
        # self._min_mask_ratio = config.min_mask_ratio
        self._feats_size = config.feats_size
        self._num_classes = config.num_classes
        self._label_group = config.label_group

        self._train_test_mode = config.train_test_mode
        self._trainval_dataset_csv_path = config.trainval_dataset_csv_path
        self._test_dataset_csv_path = config.test_dataset_csv_path

        self._split = config.split
        self._patient_id_col_name = config.patient_id_col_name

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: tp.Optional[str] = None):
        if "train" in self._train_test_mode:
            train_val_dataset_bags_df = pd.read_csv(self._trainval_dataset_csv_path)

            train_bags_df, valid_bags_df = split_dataset_by_patient(
                dataframe=train_val_dataset_bags_df,
                groups_col_name=self._patient_id_col_name,
                feature_col_names=["LUAD", "LUSC"],
                # TODO: unify this and the config label_group OR make another parameter for this
                test_size=self._split,
                random_state=42,
            )

            self.train_dataset = LungSubtypingSlideEmbeddingDataset(
                train_bags_df,
                feats_dir=self._feats_dir,
                feats_size=self._feats_size,
                num_classes=self._num_classes,
                label_group=self._label_group,
                # min_mask_ratio=self._min_mask_ratio,
                device=self.device,
            )
            self.val_dataset = LungSubtypingSlideEmbeddingDataset(
                valid_bags_df,
                feats_dir=self._feats_dir,
                feats_size=self._feats_size,
                num_classes=self._num_classes,
                label_group=self._label_group,
                # min_mask_ratio=self._min_mask_ratio,
                device=self.device,
            )

        if "test" in self._train_test_mode:
            test_dataset_bags_df = pd.read_csv(self._test_dataset_csv_path)
            self.test_dataset = LungSubtypingSlideEmbeddingDataset(
                test_dataset_bags_df,
                feats_dir=self._feats_dir,
                feats_size=self._feats_size,
                num_classes=self._num_classes,
                label_group=self._label_group,
                # min_mask_ratio=self._min_mask_ratio,
                device=self.device,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
        )
