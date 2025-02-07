import typing as tp

from omegaconf import OmegaConf
from pydantic import BaseModel


class Config(BaseModel):
    project_name: str
    experiment_name: str

    accelerator: str
    gpu_device: int

    random_seed: int

    train_test_mode: str
    split: float
    trainval_dataset_csv_path: str
    test_dataset_csv_path: str

    label_group: str
    num_classes: int

    task_type: str
    patch_loc_size: int
    label_col_name: str
    feats_dir: str
    min_mask_ratio: float
    patient_id_col_name: str
    subsample_num_patches: int
    dropout_patch: float

    feats_size: int
    # for slide-level embeddings
    linear_probing: bool
    # for patch-level embeddings
    instance_embedder_name: str
    instance_classifier_name: tp.Optional[str] = None
    instance_loss_coef: float
    instance_embedder_output_size: int
    architecture_name: str
    aggregator_kwargs: dict
    class_connector_name: str
    classifier_name: str
    saved_weights_pth: str
    dropout_node: float

    num_epochs: int
    batch_size: int
    eval_batch_size_size_proportion: float # should be in (0, 1]
    limit_num_batches: tp.Union[float, int]

    optimizer: tp.Optional[str] = None          # not needed in test mode
    optimizer_kwargs: tp.Optional[dict] = None  # not needed in test mode

    scheduler: tp.Optional[str] = None
    scheduler_kwargs: tp.Optional[dict] = None

    average_instance_and_bag_pred: bool


    @classmethod
    def from_yaml(cls, base_path: str, current_path: str) -> 'Config':
        base_cfg = OmegaConf.load(base_path)
        current_cfg = OmegaConf.load(current_path)
        merged_cfg = OmegaConf.merge(base_cfg, current_cfg)
        cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
        return cls(**cfg_dict)

    @classmethod
    def from_yaml_standalone(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
