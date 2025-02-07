"""
Saves metadata for all WSIs in a dataset.

For DHMC dataset, the metadata is saved in a CSV file. The metadata includes `mpp` and `power` values for each slide.
"""

# built-in imports
import argparse
import os
import sys
import glob
import importlib
import json
from pprint import pprint
import concurrent.futures

from pathlib import Path

# standard imports
import numpy as np
import pandas as pd

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
from tiatoolbox.wsicore.wsireader import WSIReader


DHMC_metadata_path = "/well/rittscher-dart/shared/datasets/lung/DHMC/MetaData_Release_1.0.csv"

class NumpyPosixEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return super(NumpyPosixEncoder, self).default(obj)


def process_slide(slide_path, dataset, metadata_df, verbose):
    slide_basename = Path(slide_path).name
    slide_id = Path(slide_path).stem
    
    if dataset.startswith("DHMC"):
        slide_metadata = metadata_df.loc[slide_basename]
        mpp = slide_metadata["Microns Per Pixel"]
        power = slide_metadata["Magnification"]
    else:
        mpp = None
        power = None

    # open the slide
    try:
        reader = WSIReader.open(slide_path, mpp=mpp, power=power)
    except Exception as e:
        return slide_id, None, str(e)
    
    if reader is None:
        return slide_id, None, "reader is None"

    info_dict = reader.info.as_dict()
    if verbose:
        pprint(reader.info.as_dict())

    slide_metadata = {}
    slide_metadata["stain"] = "EVG" if "_EVG" in slide_path else "H&E"
    for key, value in info_dict.items():
        slide_metadata[key] = value

    return slide_id, slide_metadata, None


# python a_save_slide_metadata.py --dataset ouh_batch1_20x --slide_format ndpi
# python a_save_slide_metadata.py --dataset ouh_batch1_40x --slide_format tif
# python a_save_slide_metadata.py --dataset ouh_batch2_20x --slide_format tif
# python a_save_slide_metadata.py --dataset ouh_batch2_40x --slide_format tif
# python a_save_slide_metadata.py --dataset ouh_batch3_40x --slide_format tif

# python a_save_slide_metadata.py --dataset DHMC_20x --slide_format tif
# python a_save_slide_metadata.py --dataset DHMC_40x --slide_format tif
# python a_save_slide_metadata.py --dataset TCGA-lung --slide_format svs
# python a_save_slide_metadata.py --dataset TCIA_CPTAC_test --slide_format svs
# python a_save_slide_metadata.py --dataset TCIA-CPTAC --slide_format svs

# python a_save_slide_metadata.py --dataset DART_001 --slide_format tif
# python a_save_slide_metadata.py --dataset DART_002 --slide_format tif
# python a_save_slide_metadata.py --dataset DART_003 --slide_format tif
# python a_save_slide_metadata.py --dataset DART_004 --slide_format tif

# python a_save_slide_metadata.py --dataset CAMELYON16 --slide_format tif

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saving metadata for all WSIs in a dataset")
    parser.add_argument("--dataset", type=str, default="TCIA-CPTAC_test", help="Dataset name")
    parser.add_argument("--slide_format", type=str, default="svs", help="Image format for tiles.", choices=["svs", "tif", "tiff", "ndpi"])
    parser.add_argument("--save_dir", default="wsi_metadata/", type=str, help="Directory to save metadata")
    # compute arguments
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes to use")
    # flags
    parser.add_argument("--verbose", action="store_true", help="Print verbose output", default=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata file", default=False)
    # all arguments
    args = parser.parse_args()

    # get all slide paths in the dataset
    path_base = os.path.join("WSI", args.dataset)
    dataset_metadata_save_path = os.path.join(args.save_dir, f"{args.dataset}.json")

    if os.path.exists(dataset_metadata_save_path) and not args.overwrite:
        raise FileExistsError(f"Metadata for {args.dataset} already exists. Skipping.")

    all_slide_paths = sorted(
        glob.glob(os.path.join(path_base, "*/*." + args.slide_format))
        + glob.glob(os.path.join(path_base, "*/*/*." + args.slide_format))
    )

    if args.dataset.startswith("DHMC"):
        metadata_df = pd.read_csv(DHMC_metadata_path, index_col="File Name")
    else:
        metadata_df = None

    total_failed = 0
    slide_2_metadata = {}
    slide_2_failure_details = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_slide, slide_path, args.dataset, metadata_df, args.verbose)
            for slide_path in all_slide_paths
        ]
        for future in concurrent.futures.as_completed(futures):
            slide_id, slide_metadata, error = future.result()
            if error:
                total_failed += 1
                slide_2_failure_details[slide_id] = error
            else:
                assert slide_id not in slide_2_metadata, f"Duplicate slide: {slide_id}"
                slide_2_metadata[slide_id] = slide_metadata

    print(f"\nDONE!!! Extracting metadata done for {len(all_slide_paths)-total_failed} / {len(all_slide_paths)} slides of {args.dataset} dataset.")

    # save metadata to a JSON file
    with open(dataset_metadata_save_path, "w") as f:
        json.dump(slide_2_metadata, f, cls=NumpyPosixEncoder, indent=2)

    # save details of failed slides to a file
    with open(os.path.join(args.save_dir, f"{args.dataset}_failed_slides.json"), "w") as f:
        json.dump(slide_2_failure_details, f, indent=2)
