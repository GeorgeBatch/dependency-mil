"""
Extracts patch features for WSIs in a dataset.

Masks are produced by thresholding the tissue regions in the WSI at the previous steps.
"""

import shutil
import sys
import importlib
import random
import os
import glob
import argparse
import json

from pathlib import Path

import cv2
import numpy as np

import torch

import joblib

# get tiatoolbox>=1.6
import tiatoolbox
assert tiatoolbox.__version__ >= "1.6", (
    f"tiatoolbox version {tiatoolbox.__version__} is installed, but version >= 1.6 is required."
)
# tiatoolbox imports
from tiatoolbox import rcParam
from tiatoolbox.models import DeepFeatureExtractor, IOSegmentorConfig, WSIStreamDataset
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader
from tiatoolbox.utils.misc import imread

import logging
import warnings

from source.feature_extraction.get_model import get_feature_extractor_model
from source.feature_extraction.data import transform_imagenet, transform_uniform, transform_none

# Set logging level based on environment variable
logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, logging_level))

# Suppress specific repeating warnings
warnings.filterwarnings(
    "once",
    "Scale > 1.This means that the desired resolution is higher than the WSI baseline",
)
warnings.filterwarnings(
    "once",
    "Read: Scale > 1.This means that the desired resolution is higher than the WSI baseline (maximum encoded resolution). Interpolation of read regions may occur.",
)

# Set random seed for reproducibility
SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def xreader(img_path: str | Path, mpp: float, power: float) -> WSIReader:
    """Multiplex tif image reader."""
    return WSIReader.open(img_path, mpp=mpp, power=power)


class XReader(WSIStreamDataset):
    """Multiplex image reader as a WSIStreamDataset."""

    def __init__(self, metadata, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata

    def _get_reader(self: WSIStreamDataset, img_path: Path) -> WSIStreamDataset:
        """Get appropriate reader for input path."""
        slide_name = img_path.stem
        metadata = self.metadata[slide_name]
        mpp = metadata["mpp"]
        power = metadata["objective_power"]
        return xreader(img_path, mpp, power)


class XExtractor(DeepFeatureExtractor):
    """
    Multiplex image extractor engine.

    Original code:
    https://github.com/TissueImageAnalytics/tiatoolbox/blob/12d435ec94f99ee312412f0848f399b4ec632464/tiatoolbox/models/engine/semantic_segmentor.py#L706-L735

    The only change is using `xreader` instead of `WSIReader.open`.
    """
    
    def __init__(self, metadata=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'metadata'):
            raise AttributeError("metadata already exists as an attribute.")
        self.metadata = metadata

    def get_reader(
        self,
        img_path: str | Path,
        mask_path: str | Path,
        mode: str,
        *,
        auto_get_mask: bool,
    ) -> tuple[WSIReader, WSIReader]:
        """Get reader for mask and source image."""
        img_path = Path(img_path)
        slide_name = img_path.stem

        metadata = self.metadata[slide_name] if self.metadata else {}
        mpp = metadata["mpp"]
        power = metadata["objective_power"]

        reader = xreader(img_path, mpp, power)
        mask_reader = None
        if mask_path is not None:
            mask_path = Path(mask_path)
            if not Path.is_file(mask_path):
                msg = "`mask_path` must be a valid file path."
                raise ValueError(msg)
            mask = imread(mask_path)  # assume to be gray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.info = reader.info
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = reader.tissue_mask(resolution=1.25, units="power")
            mask_reader.info = reader.info
        return reader, mask_reader

    def get_dataset(self, *args, **kwargs):
        """Get dataset class with metadata."""
        return self.dataset_class(self.metadata, *args, **kwargs)


def convert_npy_to_pt_file(npy_file):
    data = np.load(npy_file)
    pt_file = npy_file.with_suffix(".pt")
    torch.save(torch.from_numpy(data), pt_file)
    os.remove(npy_file)


def convert_all_npy_to_pt(save_dir_path):
    """Convert all .npy files in the directory to .pt files."""
    for npy_file in Path(save_dir_path).rglob("*.npy"):
        convert_npy_to_pt_file(npy_file)


def extract_deep_features(
    wsi_paths,
    mask_paths,
    metadata,
    save_dir,
    mode="wsi",
    feature_extractor="UNI",
    preproc_func="imagenet",
    patch_size=224,
    patch_resolution=0.5,
    patch_units="mpp",
    batch_size=32,
    num_workers=8,
    verbose=False,
    crash_on_exception=False,
    save_format="npy",  # New argument to specify save format
    device="cuda",
):
    """
    Extract Deep features from whole slide images (WSIs).

    Args:
        wsi_paths (list): List of paths to whole slide images.
        mask_paths (list): List of paths to corresponding tissue masks.
        metadata (dict): Dictionary containing metadata for slides.
        save_dir (str): Directory to save extracted features.
        mode (str): Extraction mode, either 'wsi' or 'tile'.
        feature_extractor (str, optional): Feature extractor to use. Defaults to "UNI".
        preproc_func (str, optional): Preprocessing function to use. Defaults to "imagenet".
        patch_size (int, optional): Size of the patches to extract. Defaults to 224.
        patch_resolution (float, optional): Resolution for patch extraction. Defaults to 0.5.
        patch_units (str, optional): Units for patch resolution. Defaults to "mpp" (microns per pixel).
        batch_size (int, optional): Batch size for feature extraction. Defaults to 32.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        verbose (bool, optional): Print verbose output. Defaults to False.
        crash_on_exception (bool, optional): Crash on exception in `extractor.predict()`. Defaults to False.
        save_format (str, optional): Format to save features and positions files. Defaults to "npy".
        device (str, optional): Device to run feature extraction on. Defaults to "cuda".

    Returns:
        list: List of tuples containing input and output paths for each processed WSI.

    Raises:
        ValueError: If the save_dir already exists.
    """
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": patch_units, "resolution": patch_resolution},
        ],
        output_resolutions=[
            {"units": patch_units, "resolution": patch_resolution},
        ],
        patch_input_shape=[patch_size, patch_size],
        patch_output_shape=[patch_size, patch_size],
        stride_shape=[patch_size, patch_size],
        save_resolution=None,
    )

    model = get_feature_extractor_model(feature_extractor)

    def _preproc_func(img):
        """Preprocess function to transform input image."""
        if preproc_func == 'imagenet':
            transform = transform_imagenet()
        elif preproc_func == 'uniform':
            transform = transform_uniform()
        elif preproc_func == 'none':
            transform = transform_none()
        else:
            raise ValueError(f"Preprocessing function {preproc_func} not implemented.")
        return transform(img)

    def _postproc_func(img):
        """Postprocess function (identity function in this case)."""
        return img

    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    if metadata:
        extractor = XExtractor(
            batch_size=batch_size,
            model=model,
            num_loader_workers=num_workers,
            num_postproc_workers=num_workers,
            # dataset_class=XReader,  # TODO: figure out why it works (default WSIStreamDataset)
            metadata=metadata,
        )
        print("Using XExtractor for feature extraction.")
    else:
        extractor = DeepFeatureExtractor(
            batch_size=batch_size,
            model=model,
            num_loader_workers=num_workers,
            num_postproc_workers=num_workers,
        )
        print("Using DeepFeatureExtractor for feature extraction.")

    output_map_list = extractor.predict(
        imgs=wsi_paths,
        masks=mask_paths,
        mode=mode,
        ioconfig=ioconfig,
        device=device,
        crash_on_exception=crash_on_exception,
        save_dir=save_dir,
    )

    # Move and rename output files to match input names
    for input_path, output_path in output_map_list:
        input_name = Path(input_path).stem
        output_parent_dir = Path(output_path).parent

        # Move and rename position file
        src_pos_path = Path(f"{output_path}.position.npy")
        new_pos_path = Path(f"{output_parent_dir}/{input_name}.positions.npy")
        src_pos_path.rename(new_pos_path)

        # Move and rename features file
        src_feats_path = Path(f"{output_path}.features.0.npy")
        new_feats_path = Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_feats_path.rename(new_feats_path)

        if verbose:
            print(f"Extracted features for {input_name} saved in {output_parent_dir}")
            positions = np.load(new_pos_path)
            print(f"Extracted positions shape for {input_name}: {positions.shape}")
            features = np.load(new_feats_path)
            print(f"Extracted features shape for {input_name}: {features.shape}")

    return output_map_list


def rename_files_from_map(save_dir_path):
    """Rename files based on file_map.dat or file_map_old.dat."""
    map_file_path = Path(save_dir_path) / "file_map.dat"
    old_map_file_path = Path(save_dir_path) / "file_map_old.dat"
    
    if map_file_path.exists():
        map_file = map_file_path
        # remove old map file
        if old_map_file_path.exists():
            os.remove(old_map_file_path)
    elif old_map_file_path.exists():
        map_file = old_map_file_path
    else:
        raise FileNotFoundError(f"No file_map.dat or file_map_old.dat found in the save directory: {save_dir_path}")

    with open(map_file, 'rb') as f:
        file_map = joblib.load(f)

    for input_path, output_path in file_map:
        input_name = Path(input_path).stem
        output_parent_dir = Path(output_path).parent

        # Rename position file
        src_pos_path = Path(f"{output_path}.position.npy")
        new_pos_path = Path(f"{output_parent_dir}/{input_name}.positions.npy")
        if src_pos_path.exists():
            src_pos_path.rename(new_pos_path)

        # Rename features file
        src_feats_path = Path(f"{output_path}.features.0.npy")
        new_feats_path = Path(f"{output_parent_dir}/{input_name}.features.npy")
        if src_feats_path.exists():
            src_feats_path.rename(new_feats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Feature Extraction from WSIs")
    parser.add_argument("--dataset", type=str, default="TCIA-CPTAC_test", help="Dataset name")
    parser.add_argument("--slide_format", type=str, default="svs", help="Image format for tiles.", choices=["svs", "tif", "ndpi"])
    # save directory for masks and thumbnails
    parser.add_argument("--masks_dir", default="wsi_thumbnails_and_masks/", type=str, help="Directory where masks have been saved.")
    parser.add_argument("--save_dir_root", default="datasets/", type=str, help="Root directory to save extracted features.")
    # feature extractor and preprocessing
    parser.add_argument("--feature_extractor", default="UNI", type=str)   # batch size 64 for a single A6000 GPU when using 8 workers
    parser.add_argument("--preproc_func", default="imagenet", type=str)
    # patch extraction arguments
    parser.add_argument("--patch_size", default=224, type=int)
    parser.add_argument("--patch_resolution", default=1, type=float)
    parser.add_argument("--patch_units", default="mpp", type=str)
    # performance arguments
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # flags
    parser.add_argument("--verbose", action="store_true", help="Print verbose output", default=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features directory.", default=False)
    parser.add_argument("--resume", action="store_true", help="Resume feature extraction, appending to existing features.", default=False)
    parser.add_argument("--crash_on_exception", action="store_true", help="Crash on exception during feature extraction.", default=False)
    parser.add_argument("--tiatoolbox_metadata_path", type=str, help="Path to the tiatoolbox metadata file")
    parser.add_argument("--save_format", type=str, choices=["npy", "pt"], default="pt", help="Format to save features and positions files.")
    parser.add_argument("--cpu_only", action="store_true", help="Run only on CPU, even if GPU is available.")
    parser.add_argument("--torch_compile_mode", type=str, default="disable",
                        choices=["default", "reduce-overhead", "max-autotune", "disable"],
                        help="Torch compile mode implemented through `tiatoolbox`.")
    # all arguments
    args = parser.parse_args()

    # Ensure only one of overwrite or resume is enabled
    if args.overwrite and args.resume:
        raise ValueError("Cannot use both --overwrite and --resume options at the same time. Please choose one.")

    print('\n', "#" * 200, '\n', "#" * 200, '\n')
    print("New run of c_compute_tiatoolbox_feats.py")
    print("\nargs - raw:")
    print(sys.argv)
    print("\nargs:")
    print(args)
    print('\n', "#" * 200, '\n', "#" * 200, '\n')

    # set device
    args.on_gpu = not args.cpu_only and torch.cuda.is_available()
    device = "cuda" if args.on_gpu else "cpu"
    # torch compile mode
    if args.on_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        assert args.torch_compile_mode == "disable", "Multi-GPU training is not supported by `tiatoolbox` with torch compile mode enabled."
    torch._dynamo.reset()  # include this line every time `torch.compile` mode is to be changed
    rcParam["torch_compile_mode"] = args.torch_compile_mode

    # load metadata if provided
    metadata = None
    if args.tiatoolbox_metadata_path:
        with open(args.tiatoolbox_metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata for {len(metadata)} slides loaded from {args.tiatoolbox_metadata_path}", end="\n\n")

    # get all slide paths in the dataset
    path_base = os.path.join('WSI', args.dataset)
    all_slide_paths = (
        glob.glob(os.path.join(path_base, '*/*.' + args.slide_format))
        + glob.glob(os.path.join(path_base, '*/*/*.' + args.slide_format))
    )
    all_slide_names = [Path(slide_path).stem for slide_path in all_slide_paths]
    all_mask_paths = [
        os.path.join(
            args.masks_dir,
            args.dataset,
            slide_name + "_mask.png",
        )
        for slide_name in all_slide_names
    ]
    # filter out slides without corresponding masks
    valid_slide_paths = [slide_path for slide_path, mask_path in zip(all_slide_paths, all_mask_paths) if os.path.exists(mask_path)]
    valid_mask_paths = [mask_path for mask_path in all_mask_paths if os.path.exists(mask_path)]
    assert len(valid_slide_paths) == len(valid_mask_paths), "Number of valid slides and masks do not match."
    assert len(valid_slide_paths) > 0, "No valid slides found."

    # directory path for saving extracted features
    save_dir_path = os.path.join(
        args.save_dir_root,
        args.dataset,
        "features",
        args.feature_extractor,
        args.preproc_func,
        f"patch_{args.patch_size}_{args.patch_resolution}_{args.patch_units}",
    )
    # directory for handling multiple runs of the script with --resume option enabled
    done_dir_path = save_dir_path + "_done"
    if os.path.exists(done_dir_path) and not args.resume:
        raise ValueError(f"Directory {done_dir_path} already exists, but --resume is not enabled.")

    # what to do if save_dir_path already exists
    if os.path.exists(save_dir_path):
        if args.overwrite:
            print(f"Overwriting existing features in {save_dir_path}")
            shutil.rmtree(save_dir_path)
        elif args.resume:
            # Remove /cache directories from save_dir_path
            cache_dir = os.path.join(save_dir_path, "cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"Removed cache directory: {cache_dir}")

            if len(os.listdir(save_dir_path)) > 0:
                # Rename files based on file_map before resuming
                rename_files_from_map(save_dir_path)

            # Move feature files to done_dir_path and delete .dat files
            os.makedirs(done_dir_path, exist_ok=True)
            for item in os.listdir(save_dir_path):
                if not item.endswith(".dat"):
                    shutil.move(os.path.join(save_dir_path, item), done_dir_path)
                else:
                    os.remove(os.path.join(save_dir_path, item))
            # should be empty now, otherwise, os.rmdir will raise an error that the directory is not empty
            os.rmdir(save_dir_path)
            print(f"Resuming feature extraction. Existing features moved to {done_dir_path}")
        else:
            raise ValueError(f"Directory save_dir_path={save_dir_path} already exists, but neither --overwrite nor --resume are enabled.")

    # Check if there are any slides already processed, this will only be the case if --resume is enabled. We checked for this earlier.
    processed_slides = set()
    if os.path.exists(done_dir_path):
        # Get all .npy files in done_dir_path
        processed_positions = {f[:-len(".positions.npy")] for f in os.listdir(done_dir_path) if f.endswith(".positions.npy")}
        processed_features = {f[:-len(".features.npy")] for f in os.listdir(done_dir_path) if f.endswith(".features.npy")}
        # In case they were somehow already converted, add .pt files to processed_features and processed_positions
        processed_positions.update({f[:-len(".positions.pt")] for f in os.listdir(done_dir_path) if f.endswith(".positions.pt")})
        processed_features.update({f[:-len(".features.pt")] for f in os.listdir(done_dir_path) if f.endswith(".features.pt")})

        processed_slides = processed_positions.intersection(processed_features)
        print(f"Found {len(processed_slides)} slides already processed in {done_dir_path}")

    # Filter out already processed slides
    # Path("slide.1.id.ext").stem -> "slide.1.id"
    valid_slide_paths = sorted([slide_path for slide_path in valid_slide_paths if Path(slide_path).stem not in processed_slides])
    valid_mask_paths = sorted([mask_path for mask_path in valid_mask_paths if Path(mask_path).stem.replace("_mask", "") not in processed_slides])
    # check if slides match the masks
    assert (
        len(valid_slide_paths) == len(valid_mask_paths)
    ), f"Number of valid slides ({len(valid_slide_paths)}) and masks ({len(valid_mask_paths)}) do not match after filtering out processed slides."
    assert all(
        Path(slide_path).stem == Path(mask_path).stem.replace("_mask", "")
        for slide_path, mask_path in zip(valid_slide_paths, valid_mask_paths)
    ), "Slide names do not match mask names."
    assert len(valid_slide_paths) > 0, "No valid slides found after filtering out processed slides."
    print(f"Processing {len(valid_slide_paths)} slides that have not been processed yet.\n")

    # extract deep features from WSIs
    output_list = extract_deep_features(
        wsi_paths=valid_slide_paths,
        mask_paths=valid_mask_paths,
        metadata=metadata,
        save_dir=save_dir_path,
        mode="wsi",  # this script is for WSI feature extraction
        feature_extractor=args.feature_extractor,
        preproc_func=args.preproc_func,
        patch_size=args.patch_size,
        patch_resolution=args.patch_resolution,
        patch_units=args.patch_units,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose,
        crash_on_exception=args.crash_on_exception,
        save_format=args.save_format,
        device=device,
    )

    print(f"{args.dataset} dataset had {len(valid_slide_paths)} slides with masks - extracted features for all of them.")
    print(f"{args.dataset} dataset had {len(all_slide_paths)} slides in total.")

    # Move all slides from _done directory back to the normal directory and delete _done directory
    if os.path.exists(done_dir_path):
        # Move features and position files back to save_dir_path
        for item in os.listdir(done_dir_path):
            shutil.move(os.path.join(done_dir_path, item), save_dir_path)
        os.rmdir(done_dir_path)
        print(f"All slides processed. Moved features back to {save_dir_path} and deleted {done_dir_path}")
        # remove file_map.dat and file_map_old.dat since they do not have information about the moved files
        for map_file in ["file_map.dat", "file_map_old.dat"]:
            map_file_path = Path(save_dir_path) / map_file
            if map_file_path.exists():
                os.remove(map_file_path)

    # Convert all .npy files to .pt if required
    if args.save_format == "pt":
        convert_all_npy_to_pt(save_dir_path)
        print(f"Converted all .npy files to .pt in {save_dir_path}")
