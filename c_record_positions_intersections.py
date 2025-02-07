# built-in imports
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# standard imports
import numpy as np
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description="Record positions intersections.")
parser.add_argument("--save_positions", action="store_true", default=False, help="Save the intersection positions.")
parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files.")
parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed information.")
parser.add_argument("--num_workers", type=int, default=24, help="Number of worker processes to use.")
args = parser.parse_args()

def process_slide(slide_id, slide_ids_path, dataset, feature_extractor, img_norm, patch_extraction, min_mask_ratio):
    # Construct paths
    features_positions_path = slide_ids_path / f"{slide_id}.positions.pt"
    min_mask_ratio_path = Path(f"wsi_masked_positions/{dataset}/{min_mask_ratio}/{patch_extraction}/{slide_id}.positions.npy")
    
    # Check if both paths exist
    if features_positions_path.exists() and min_mask_ratio_path.exists():
        # where to save
        intersection_indexes_save_path = Path(
            f"wsi_masked_positions_intersections/{dataset}/{feature_extractor}/{img_norm}/{patch_extraction}/{min_mask_ratio}/{slide_id}.indexes.npy"
        )
        intersection_positions_save_path = intersection_indexes_save_path.with_suffix(".positions.npy")
        
        # create directories if they don't exist
        intersection_indexes_save_path.parent.mkdir(parents=True, exist_ok=True)
        intersection_positions_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if the intersection indexes file already exists
        if intersection_indexes_save_path.exists() and not args.overwrite:
            print(f"File {intersection_indexes_save_path} already exists. Skipping...")
            return
        
        # Load positions
        features_positions = torch.load(features_positions_path, weights_only=True).numpy()  # bag size x 4
        min_mask_ratio_positions = np.load(min_mask_ratio_path)
        
        # Convert rows to strings
        features_positions_as_strings = np.array([" ".join(map(str, row)) for row in features_positions])
        min_mask_ratio_positions_as_strings = np.array([" ".join(map(str, row)) for row in min_mask_ratio_positions])
        
        # Find intersection
        intersection_strings = np.intersect1d(features_positions_as_strings, min_mask_ratio_positions_as_strings)
        
        # Get indexes of features-corresponding positions that are in the intersection
        intersection_indexes = np.nonzero(np.isin(features_positions_as_strings, intersection_strings))[0]
        
        # Subset the features_positions array
        features_positions_subset = features_positions[intersection_indexes]
        
        if args.verbose:
            print("Proportion of intersection:", round(intersection_strings.shape[0] / features_positions.shape[0], 2))
            print("Indexes of intersection:", intersection_indexes.shape, intersection_indexes, "\n")
            print("Subset of features_positions:", features_positions_subset.shape, features_positions_subset, "\n")
            print("="*80)
        
        # Save the intersection indexes
        np.save(intersection_indexes_save_path, intersection_indexes)
        # Save the intersection positions only if --save_positions is invoked
        if args.save_positions:
            np.save(intersection_positions_save_path, features_positions_subset)

# Get all datasets in the datasets/ directory
datasets_path = Path("datasets/")
datasets = [d.name for d in datasets_path.iterdir() if d.is_dir()]

for dataset in datasets:
    # Get all feature extractors in the datasets/{dataset}/features/ directory
    feature_extractors_path = datasets_path / dataset / "features"
    if not feature_extractors_path.exists():
        print(f"Directory {feature_extractors_path} does not exist. Skipping...")
        continue
    feature_extractors = [f.name for f in feature_extractors_path.iterdir() if f.is_dir()]
    if args.verbose:
        print("\tfeature_extractors:", feature_extractors)
    
    for feature_extractor in feature_extractors:
        # Get all image norms in the datasets/{dataset}/features/{feature_extractor}/ directory
        img_norms_path = feature_extractors_path / feature_extractor
        if not img_norms_path.exists():
            print(f"Directory {img_norms_path} does not exist. Skipping...")
            continue
        img_norms = [f.name for f in img_norms_path.iterdir() if f.is_dir()]
        if args.verbose:
            print("\t\timg_norms:", img_norms)
        
        for img_norm in img_norms:
            # Get all patch extractions in the datasets/{dataset}/features/{feature_extractor}/{img_norm}/ directory
            patch_extractions_path = img_norms_path / img_norm
            if not patch_extractions_path.exists():
                print(f"Directory {patch_extractions_path} does not exist. Skipping...")
                continue
            patch_extractions = [f.name for f in patch_extractions_path.iterdir() if f.is_dir()]
            if args.verbose:
                print("\t\t\tpatch_extractions:", patch_extractions)
            
            for patch_extraction in patch_extractions:
                # Get all min mask ratios in the wsi_masked_positions/{dataset}/ directory
                min_mask_ratios_path = Path(f"wsi_masked_positions/{dataset}/")
                if not min_mask_ratios_path.exists():
                    print(f"Directory {min_mask_ratios_path} does not exist. Skipping...")
                    continue
                min_mask_ratios = [f.name for f in min_mask_ratios_path.iterdir() if f.is_dir()]
                if args.verbose:
                    print("\t\t\t\tmin_mask_ratios:", min_mask_ratios)
                
                for min_mask_ratio in min_mask_ratios:
                    # Get all slide IDs in the directory
                    slide_ids_path = patch_extractions_path / patch_extraction
                    if args.verbose:
                        print("\t\t\t\t\t", slide_ids_path)
                    if not slide_ids_path.exists():
                        print(f"Directory {slide_ids_path} does not exist. Skipping...")
                        continue
                    slide_ids = [f.name.replace(".positions.pt", "") for f in slide_ids_path.glob("*.positions.pt")]
                    
                    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                        for slide_id in slide_ids:
                            executor.submit(process_slide, slide_id, slide_ids_path, dataset, feature_extractor, img_norm, patch_extraction, min_mask_ratio)