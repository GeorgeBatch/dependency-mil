from PIL import Image

import torch
from torch import nn

import timm
from timm.layers import SwiGLUPacked
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform


class VirchowFeatureExtractor(nn.Module):
    """
    VirchowFeatureExtractor is a feature extractor model based on different versions of the ViTModel.

    Args:
        version (str): The version of the model to use. Options are "v1" and "v2".
        features_mode (str): which features to give as output. Options are "cls_token", "mean_patch_tokens", "concat".
    Raises:
        ValueError: If an invalid version is provided.

    Version Details:
        - v1 (paige-ai/Virchow):
            - output tensor [batch_size, 257, 1280]
                - batch_size images
                - 257 tokens = 1 cls token + 16*16 patch tokens
                - 1280 features
        - v2 (paige-ai/Virchow2):
            - output tensor [batch_size, 261, 1280]
                - batch_size images
                - 261 tokens = 1 cls token + 4 DINOv2 register tokens + 16*16 patch tokens
                - 1280 features
    """
    def __init__(self, version, features_mode):
        super().__init__()
        
        assert features_mode in [
            "cls-token",
            "mean-patch-tokens",
            "concat",
        ], f"Invalid features_mode: {features_mode}"
        self.features_mode = features_mode

        self.version = version
        if version == "v1":
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        elif version == "v2":
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        else:
            raise ValueError(f"Invalid version: {version}")

    def forward(self, x):
        output = self.model(x)
        cls_token = output[:, 0, :]

        if self.version == "v1":
            patch_tokens = output[:, 1:, :]
        elif self.version == "v2":
            # register_tokens = output[:, 1:5, :]
            patch_tokens = output[:, 5:, :]

        if self.features_mode == 'cls_token':
            return cls_token
        elif self.features_mode == 'mean_patch_tokens':
            return patch_tokens.mean(1)
        elif self.features_mode == 'concat':
            return torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        else:
            raise ValueError(f"Invalid features_mode: {self.features_mode}")


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = VirchowFeatureExtractor(version="v1", features_mode="concat")

    # imagenet transform
    #
    # transforms = create_transform(
    #     **resolve_data_config(
    #         feature_extractor.model.pretrained_cfg, model=feature_extractor.model
    #     )
    # )
    # print("transforms", transforms)

    # batch_size, 3, 224, 224
    image = torch.randint(0, 255, (8, 3, 224, 224)) / 255
    # image = Image.open()
    # image = transforms(image).unsqueeze(0)

    model = feature_extractor.to(device)
    image = image.to(device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device, dtype=torch.float16),
    ):
        
        for _ in range(5):
            output = model(image)
    print(output.shape)
    