import os
import sys
import importlib

import numpy as np
import torch
from torch import nn


# get tiatoolbox
try:
    tiatoolbox = importlib.import_module("tiatoolbox")
    if tiatoolbox.__version__ < "1.6":
        raise ImportError(
            f"tiatoolbox version {tiatoolbox.__version__} is installed, but version >= 1.6 is required."
        )
except ImportError as e:
    print(e)
    sys.path.append("../")
    # use local path tiatoolbox imports
    if os.getcwd().startswith("/gpfs3/well/rittscher-dart/"):
        sys.path.append(
            "/well/rittscher-dart/users/qun786/projects/current/comp-path/tiatoolbox/"
        )
    elif os.getcwd().startswith("/Users/gbatch"):
        sys.path.append("/Users/gbatch/Developer/projects/current/comp-path/tiatoolbox")
    elif os.getcwd().startswith("/home/georgebatchkala"):
        sys.path.append(
            "/home/georgebatchkala/Developer/projects/current/comp-path/tiatoolbox"
        )
    else:
        raise NotImplementedError("Local path for tiatoolbox import not defined.")
    # Try importing again from the local path
    try:
        tiatoolbox = importlib.import_module("tiatoolbox")
    except ImportError:
        raise ImportError(
            "tiatoolbox is not installed and could not be found in the local path."
        )
# tiatoolbox imports
from tiatoolbox.models.architecture.vanilla import ModelABC, CNNBackbone, TimmBackbone, _infer_batch

# local imports
from source.feature_extraction.models.resnet_clam import resnet50_baseline
from source.feature_extraction.models.resnet_dsmil import get_resnet18_dsmil
from source.feature_extraction.models.phikon import PhikonFeatureExtractor
from source.feature_extraction.models.hibou import HibouFeatureExtractor
from source.feature_extraction.models.virchow import VirchowFeatureExtractor
from source.constants import EXTRACTOR_NAMES_2_WEIGHTS_PATHS


class CustomBackbone(ModelABC):
    def __init__(self: ModelABC, extractor_name: str) -> None:
        super().__init__()

        # -----------------------------------------------------------------------------------------------
        # Natural Image models
        # -----------------------------------------------------------------------------------------------
        if extractor_name == "imagenet_resnet50-clam-extractor":
            feature_extractor = resnet50_baseline(pretrained=True)
        
        elif extractor_name.startswith("dinov2_"):
            # DINOv2 family of models from Facebook Research: "dinov2_vit{s,b,l,g}14{_reg}{_lc}",
            feature_extractor = torch.hub.load(
                "facebookresearch/dinov2", extractor_name
            )

        # -----------------------------------------------------------------------------------------------
        # Pathology-specific models
        # -----------------------------------------------------------------------------------------------
        # Implemented: Virchow v1 and v2, Phikon v1 and v2, Hibou -b and -L
        # -----------------------------------------------------------------------------------------------
        elif extractor_name.startswith("VirchowFeatureExtractor"):
            # "VirchowFeatureExtractor_v1_concat" -> ['VirchowFeatureExtractor', 'v1', 'concat']
            # "VirchowFeatureExtractor_v2_cls-token" -> ['VirchowFeatureExtractor', 'v2', 'cls-token']
            name_split_list = extractor_name.split("_")
            feature_extractor = VirchowFeatureExtractor(
                version=name_split_list[1],
                features_mode=name_split_list[2],
            )

        elif extractor_name.startswith("PhikonFeatureExtractor"):
            # "PhikonFeatureExtractor_v2" -> ['PhikonFeatureExtractor', 'v2']
            name_split_list = extractor_name.split("_")
            feature_extractor = PhikonFeatureExtractor(
                version=name_split_list[1],
            )

        elif extractor_name.startswith("HibouFeatureExtractor"):
            # "HibouFeatureExtractor_b" -> ['HibouFeatureExtractor', 'b']
            # "HibouFeatureExtractor_L" -> ['HibouFeatureExtractor', 'L']
            name_split_list = extractor_name.split("_")
            feature_extractor = HibouFeatureExtractor(
                version=name_split_list[1],
            )

        # ResNet18 trained with SimCLR on TCGA-Lung images (2.5x magnification): https://github.com/binli123/dsmil-wsi/issues/41
        elif extractor_name == 'simclr-tcga-lung_resnet18-2.5x':
            feature_extractor = get_resnet18_dsmil(
                weights_path=EXTRACTOR_NAMES_2_WEIGHTS_PATHS[extractor_name])

        # ResNet18 trained with SimCLR on TCGA-Lung images (10x magnification): https://github.com/binli123/dsmil-wsi/issues/41
        elif extractor_name == 'simclr-tcga-lung_resnet18-10x':
            feature_extractor = get_resnet18_dsmil(
                weights_path=EXTRACTOR_NAMES_2_WEIGHTS_PATHS[extractor_name])
        
        # ResNet18 trained with SimCLR on Camelyon16 images (5x magnification): https://github.com/binli123/dsmil-wsi/
        elif extractor_name == 'simclr-camelyon16_resnet18-5x':
            feature_extractor = get_resnet18_dsmil(
                weights_path=EXTRACTOR_NAMES_2_WEIGHTS_PATHS[extractor_name])

        # ResNet18 trained with SimCLR on Camelyon16 images (20x magnification): https://github.com/binli123/dsmil-wsi/
        elif extractor_name == 'simclr-camelyon16_resnet18-20x':
            feature_extractor = get_resnet18_dsmil(
                weights_path=EXTRACTOR_NAMES_2_WEIGHTS_PATHS[extractor_name])

        else:
            msg = f"Backbone {extractor_name} not supported."
            raise ValueError(msg)

        self.feature_extractor = feature_extractor

    def forward(self, imgs) -> None:
        feats = self.feature_extractor(imgs)
        return torch.flatten(feats, 1)

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        device: str,
    ) -> list[np.ndarray]:
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (torch.Tensor):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            device (str):
                Transfers model to the specified device. Default is "cpu".

        Returns:
            list[np.ndarray]:
                list of numpy arrays.

        """
        return [_infer_batch(model=model, batch_data=batch_data, device=device)]


def get_feature_extractor_model(feature_extractor):
    model = None
    try:
        print(f"Trying to load feature extractor {feature_extractor} via CNNBackbone.")
        model = CNNBackbone(backbone=feature_extractor)
        print(f"Loaded feature extractor {feature_extractor} via CNNBackbone.")
    except ValueError as e:
        print(e)
    if model is None:
        try:
            print(
                f"Trying to load feature extractor {feature_extractor} via TimmBackbone."
            )
            model = TimmBackbone(backbone=feature_extractor, pretrained=True)
            print(f"Loaded feature extractor {feature_extractor} via TimmBackbone.")
        except ValueError as e:
            print(e)
    if model is None:
        try:
            print(
                f"Trying to load feature extractor {feature_extractor} via CustomBackbone."
            )
            model = CustomBackbone(extractor_name=feature_extractor)
            print(f"Loaded feature extractor {feature_extractor} via CustomBackbone.")
        except ValueError as e:
            print(e)
    if model is None:
        msg = f"Feature extractor {feature_extractor} not implemented via CNNBackbone, TimmBackbone, or CustomBackbone."
        raise ValueError(msg)
    
    # model = torch.compile(model)  # maybe the model needs to be compiled before using nn.DataParallel
    return model
