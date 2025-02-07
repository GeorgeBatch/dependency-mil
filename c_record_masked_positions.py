"""

"""

# built-in imports
import argparse
import os
import sys
import glob
import importlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path

# standard imports
import numpy as np

# get tiatoolbox>=1.6
import tiatoolbox
assert tiatoolbox.__version__ >= "1.6", (
    f"tiatoolbox version {tiatoolbox.__version__} is installed, but version >= 1.6 is required."
)
# tiatoolbox imports
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore.wsireader import WSIReader


def process_slide(
    slide_path, mask_path, patch_size, patch_resolution, patch_units, min_mask_ratio, save_dir_path, metadata,
):
    slide_id = slide_path.stem

    # get metadata mpp and power if available for the slide
    mpp = None
    power = None
    if metadata:
        if slide_id not in metadata:
            raise ValueError(f"Metadata for slide {slide_id} not found in provided metadata file.")
        slide_metadata = metadata[slide_id]
        mpp = slide_metadata["mpp"]
        power = slide_metadata["objective_power"]

    wsi_reader = WSIReader.open(slide_path, mpp=mpp, power=power)
    extractor = SlidingWindowPatchExtractor(
        input_img=wsi_reader, # could use the path, but would not be able to provide the mpp and power
        input_mask=mask_path,
        resolution=patch_resolution,
        units=patch_units,
        patch_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        min_mask_ratio=min_mask_ratio,
    )

    # Get all coordinates and save them to an npy file
    all_coordinates = extractor.coordinate_list
    np.save(f"{save_dir_path}/{slide_id}.positions.npy", all_coordinates)


def process_slide_wrapper(args):
    process_slide(*args)


# Example commands for different datasets at thresholds 0, 0.1, 0.5
#
# python c_record_masked_positions.py --dataset DHMC_20x --slide_format tif --min_mask_ratio 0.0 --tiatoolbox_metadata_path wsi_metadata/DHMC_20x.json --overwrite
# python c_record_masked_positions.py --dataset DHMC_20x --slide_format tif --min_mask_ratio 0.1 --tiatoolbox_metadata_path wsi_metadata/DHMC_20x.json --overwrite
# python c_record_masked_positions.py --dataset DHMC_20x --slide_format tif --min_mask_ratio 0.5 --tiatoolbox_metadata_path wsi_metadata/DHMC_20x.json --overwrite
# python c_record_masked_positions.py --dataset DHMC_40x --slide_format tif --min_mask_ratio 0.0 --tiatoolbox_metadata_path wsi_metadata/DHMC_40x.json --overwrite
# python c_record_masked_positions.py --dataset DHMC_40x --slide_format tif --min_mask_ratio 0.1 --tiatoolbox_metadata_path wsi_metadata/DHMC_40x.json --overwrite
# python c_record_masked_positions.py --dataset DHMC_40x --slide_format tif --min_mask_ratio 0.5 --tiatoolbox_metadata_path wsi_metadata/DHMC_40x.json --overwrite
#
# python c_record_masked_positions.py --dataset TCGA-lung --slide_format svs --min_mask_ratio 0.0
# python c_record_masked_positions.py --dataset TCGA-lung --slide_format svs --min_mask_ratio 0.1
# python c_record_masked_positions.py --dataset TCGA-lung --slide_format svs --min_mask_ratio 0.5
#
# python c_record_masked_positions.py --dataset TCIA-CPTAC --slide_format svs --min_mask_ratio 0.0
# python c_record_masked_positions.py --dataset TCIA-CPTAC --slide_format svs --min_mask_ratio 0.1
# python c_record_masked_positions.py --dataset TCIA-CPTAC --slide_format svs --min_mask_ratio 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save patches from WSI slides.")
    parser.add_argument("--dataset", type=str, default="TCGA-lung", help="Path to the dataset directory containing WSI slides.")
    parser.add_argument("--slide_format", type=str, default="svs", help="Format of the slide files (e.g., ndpi, svs).")
    parser.add_argument("--tiatoolbox_metadata_path", type=str, help="Path to the tiatoolbox metadata file")
    parser.add_argument("--masks_dir", default="wsi_thumbnails_and_masks/", type=str, help="Directory where masks have been saved.")
    parser.add_argument("--save_dir_root", default="wsi_masked_positions/", type=str, help="Root directory where patches will be saved.")
    parser.add_argument("--min_mask_ratio", type=float, default=0.0, help="Minimum ratio of mask pixels in a patch for it to be saved.")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patches to extract.")
    parser.add_argument("--patch_resolution", type=float, default=0.5, help="Resolution of the patches to extract.")
    parser.add_argument("--patch_units", type=str, default="mpp", help="Units of the patch resolution.")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of worker processes to use.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Delete the dataset directory if it exists.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output", default=False)
    args = parser.parse_args()

    print("\n", "-" * 48, "\n")
    print("Arguments passed to the script:")
    print("\nargs - raw:")
    print(sys.argv)
    print("\nargs:")
    print(args)
    print("\n", "-" * 48, "\n")
    
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
    valid_slide_paths = [Path(slide_path) for slide_path, mask_path in zip(all_slide_paths, all_mask_paths) if os.path.exists(mask_path)]
    valid_mask_paths = [Path(mask_path) for mask_path in all_mask_paths if os.path.exists(mask_path)]
    assert len(valid_slide_paths) == len(valid_mask_paths), "Number of valid slides and masks do not match."
    assert len(valid_slide_paths) > 0, "No valid slides found."
    assert all(slide_path.stem == mask_path.stem.replace("_mask", "") for slide_path, mask_path in zip(valid_slide_paths, valid_mask_paths)), "Slide and mask names do not match."
    print(f"Found {len(valid_slide_paths)} valid slides and masks.")

    save_dir_path = os.path.join(
        args.save_dir_root,
        args.dataset,
        f"min_mask_ratio_{args.min_mask_ratio}",
        f"patch_{args.patch_size}_{args.patch_resolution}_{args.patch_units}",
    )

    # Create directory if it doesn't exist
    if os.path.exists(save_dir_path):
        if args.overwrite:
            shutil.rmtree(save_dir_path)
            print(f"Overwrite invoked. Deleted existing directory: {save_dir_path}")
        else:
            raise FileExistsError(
                f"Directory {save_dir_path} already exists. Use --overwrite to delete it."
            )
    os.makedirs(save_dir_path, exist_ok=True)

    args_list = [
        (
            slide_path,
            mask_path,
            args.patch_size,
            args.patch_resolution,
            args.patch_units,
            args.min_mask_ratio,
            save_dir_path,
            metadata,
        )
        for slide_path, mask_path in zip(valid_slide_paths, valid_mask_paths)
        if os.path.exists(slide_path) and os.path.exists(mask_path)
    ]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        executor.map(process_slide_wrapper, args_list)

    output_file_paths = sorted(Path(save_dir_path).glob("*.positions.npy"))
    print(f"Saved {len(output_file_paths)} files to {save_dir_path}")
    if args.verbose:
        for npy_file in output_file_paths:
            positions = np.load(npy_file)
            print(f"File: {npy_file.name}, Shape: {positions.shape}")

