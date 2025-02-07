"""
Use pre-trained PRISM slide aggregator to compute slide-level embeddings.
Takes Virchow v1 concatenated (cls + mean of patch tokens) embeddings as inputs.

Page: https://huggingface.co/paige-ai/Prism

Needed to add to conda-medical-pytorch-cuda12.1 environment:
> conda install transformers environs sacremoses

transformers==4.42.4
torch==2.2.2
einops==0.8.0
environs==11.0.0
sacremoses==0.1.1

# # install xformers to use memory-efficient attention
# # set env `PERCEIVER_MEM_EFF_ATTN=true` to enable
# xformers==0.0.26
"""

import os
import argparse
import pandas as pd

import torch
from transformers import AutoModel

# Define the captions for each class
# Inspiration: https://github.com/mahmoodlab/TITAN/blob/main/datasets/config_tcga-ot.yaml
CAPTIONS = {
    "LUAD": [
        "lung adenocarcinoma",
        "adenocarcinoma of the lung",
        "pulmonary adenocarcinoma",
        "peripheral lung adenocarcinoma",
        "LUAD",
    ],
    "LUSC": [
        "squamous cell carcinoma",
        "lung squamous cell carcinoma",
        "squamous carcinoma of the lung",
        "LUSC",
    ],
    "TC": [
        "typical carcinoid",
        "typical carcinoid of the lung",
        "lung typical carcinoid",
    ]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute slide-level embeddings using PRISM model")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to run the model on")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load PRISM model.
    model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
    model.eval()
    model = model.to(device)

    # Create a dictionary to store results for each class and dataset
    results = {class_name: {dataset: {} for dataset in ["ouh_batch1_20x", "ouh_batch2_20x", "ouh_batch3_40x"]} for class_name in CAPTIONS.keys()}

    for class_name, captions in CAPTIONS.items():
        for dataset in results[class_name].keys():
            embedding_data_dir = f"datasets/{dataset}/features/VirchowFeatureExtractor_v1_concat/imagenet/patch_224_0.5_mpp"
            feature_files = sorted([f for f in os.listdir(embedding_data_dir) if f.endswith(".features.pt")])

            for feature_file in feature_files:
                slide_id = feature_file.replace(".features.pt", "")
                embedding_data_path = os.path.join(embedding_data_dir, feature_file)
                embedding_data = torch.load(embedding_data_path)
                tile_embeddings = embedding_data.unsqueeze(0).to(device)
                if args.verbose:
                    print("tile_embeddings.shape", tile_embeddings.shape)

                with torch.autocast("cuda", torch.float16), torch.inference_mode():
                    # Compute similarity with each caption
                    similarities = []
                    for caption_text in captions:
                        caption = model.tokenize([caption_text]).to(device)
                        output = model(input_ids=caption, tile_embeddings=tile_embeddings)
                        similarity = output["sim"].item()
                        similarities.append(similarity)
                        if args.verbose:
                            print(f"Similarity of {slide_id} with caption '{caption_text}': {similarity}")

                    # Store results
                    results[class_name][dataset][slide_id] = similarities

                    if args.verbose:
                        print("\n", "="*80, "\n")

    # Save results to CSV files
    # output_dir = "results/slides-active-data-enrichment/prism-captions"
    output_dir = "results/slides-active-data-enrichment/prism-multicaptions"
    os.makedirs(output_dir, exist_ok=True)
    for class_name, datasets in results.items():
        for dataset, slides in datasets.items():
            df = pd.DataFrame.from_dict(slides, orient="index", columns=CAPTIONS[class_name])
            csv_path = f"{output_dir}/{class_name}_{dataset}.csv"
            df.to_csv(csv_path)
            print(f"Results saved to {csv_path}")