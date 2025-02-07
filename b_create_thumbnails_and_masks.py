"""
Produces thumbnails and masks  for WSIs in a dataset.

The thumbnail is a downsampled version of the WSI.
The mask is produced by thresholding the tissue regions in the WSI.
Thumbail and mask for the WSI have the same size and are saved in the same directory.
"""

# built-in imports
import argparse
import os
import sys
import glob
import json
from pprint import pprint

from pathlib import Path

# standard imports
import numpy as np

# get tiatoolbox>=1.6
import tiatoolbox
assert tiatoolbox.__version__ >= "1.6", (
    f"tiatoolbox version {tiatoolbox.__version__} is installed, but version >= 1.6 is required."
)
# tiatoolbox imports
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imwrite

# python b_create_thumbnails_and_masks.py --dataset ouh_batch1_20x --slide_format ndpi
# python b_create_thumbnails_and_masks.py --dataset ouh_batch1_40x --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset ouh_batch2_20x --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset ouh_batch2_40x --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset ouh_batch3_40x --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset DHMC_20x --slide_format tif --tiatoolbox_metadata_path wsi_metadata/DHMC_20x.json
# python b_create_thumbnails_and_masks.py --dataset DHMC_40x --slide_format tif --tiatoolbox_metadata_path wsi_metadata/DHMC_40x.json
# python b_create_thumbnails_and_masks.py --dataset TCGA-lung --slide_format svs
# python b_create_thumbnails_and_masks.py --dataset TCIA-CPTAC_test --slide_format svs
# python b_create_thumbnails_and_masks.py --dataset TCIA-CPTAC --slide_format svs
# python b_create_thumbnails_and_masks.py --dataset DART_001 --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset DART_002 --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset DART_003 --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset DART_004 --slide_format tif
# python b_create_thumbnails_and_masks.py --dataset CAMELYON16 --slide_format tif
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thumbnail and Mask creation for WSI')
    parser.add_argument('--dataset', type=str, default='TCIA-CPTAC_test', help='Dataset name')
    parser.add_argument('--slide_format', type=str, default='svs',
                        help='Image format for tiles.', choices=['svs', 'tif', 'tiff', 'ndpi'])
    # save directory for masks and thumbnails
    parser.add_argument("--save_dir", default="wsi_thumbnails_and_masks/", type=str)
    # masking strategy
    parser.add_argument("--mask_method", default="otsu", type=str)
    # mask and thumbnail resolution
    parser.add_argument("--mask_units", default="mpp", type=str)
    parser.add_argument("--mask_resolution", default=8, type=float)
    # tiatoolbox metadata path
    parser.add_argument("--tiatoolbox_metadata_path", type=str, help="Path to the tiatoolbox metadata file")
    # flags
    parser.add_argument("--verbose", action="store_true", help="Print verbose output", default=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing thumbnails and masks", default=False)
    # all arguments
    args = parser.parse_args()

    # load metadata if provided
    metadata = None
    if args.tiatoolbox_metadata_path:
        with open(args.tiatoolbox_metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata for {len(metadata)} slides loaded from {args.tiatoolbox_metadata_path}", end="\n\n")

    # get all slide paths in the dataset
    path_base = os.path.join('WSI', args.dataset)
    dataset_output_save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(dataset_output_save_dir, exist_ok=True)

    all_slide_paths = sorted(
        glob.glob(os.path.join(path_base, "*/*." + args.slide_format))
        + glob.glob(os.path.join(path_base, "*/*/*." + args.slide_format))
    )

    total_failed = 0
    slide_2_failure_details = {}
    # iterate over all slides in the dataset
    for idx, slide_path in enumerate(all_slide_paths):
        slide_id = Path(slide_path).stem
        print(f"Process slide {idx + 1}/{len(all_slide_paths)}: {slide_id}. Total failed until now: {total_failed}")

        wsi_thumbnail_path = os.path.join(dataset_output_save_dir, f"{slide_id}_thumbnail.jpg")
        mask_path = os.path.join(dataset_output_save_dir, f"{slide_id}_mask.png")
        if (
            not args.overwrite
            and os.path.exists(wsi_thumbnail_path)
            and os.path.exists(mask_path)
            and os.path.getsize(wsi_thumbnail_path) > 0
            and os.path.getsize(mask_path) > 0
        ):
            print(f"Slide {slide_id} already processed. Skipping to next slide.")
            continue

        # get metadata for the slide if available
        mpp = None
        power = None
        if metadata:
            if slide_id not in metadata:
                raise ValueError(f"Metadata for slide {slide_id} not found in provided metadata file.")
            slide_metadata = metadata[slide_id]
            mpp = slide_metadata["mpp"]
            power = slide_metadata["objective_power"]
        
        # open the slide, handle exceptions and skip to next slide if error
        try:
            wsi = WSIReader.open(slide_path, mpp=mpp, power=power)
        except Exception as e:
            total_failed += 1
            slide_2_failure_details[slide_id] = {
                "error": str(e),
                "slide_path": slide_path,
            }
            print(f"Error: {e} for slide {slide_id}. Skipping to next slide.  Total failed: {total_failed}")
            continue

        if args.verbose:
            pprint(wsi.info.as_dict())

        # get thumbnail and mask of wsi (mask and thumbnail size are the same) and save it
        try:
            wsi_thumbnail = wsi.slide_thumbnail(resolution=args.mask_resolution, units=args.mask_units)
            imwrite(wsi_thumbnail_path, wsi_thumbnail)

            mask = wsi.tissue_mask(method=args.mask_method, resolution=args.mask_resolution, units=args.mask_units)
            mask_thumbnail = mask.slide_thumbnail(resolution=args.mask_resolution, units=args.mask_units)
            imwrite(mask_path, np.uint8(mask_thumbnail * 255))
        except (OSError, ValueError) as e:
            total_failed += 1
            slide_2_failure_details[slide_id] = {"error": str(e), "slide_path": slide_path}
            print(f"Error: {e} for slide {slide_id}. Skipping to next slide.  Total failed: {total_failed}")
            continue

    print(f"DONE!!! Creating thumbnails and Masking done for {len(all_slide_paths)-total_failed} / {len(all_slide_paths)} slides of {args.dataset} dataset.")

    # save details of failed slides to a file
    with open(os.path.join(args.save_dir, f"{args.dataset}_failed_slides.json"), "w") as f:
        json.dump(slide_2_failure_details, f, indent=4)
