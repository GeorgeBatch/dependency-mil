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

import torch
from transformers import AutoModel

# python c_compute_prism_slide_level_feats.py --embedding_data_dir datasets/DHMC_40x/features/VirchowFeatureExtractor_v1_concat/imagenet/patch_224_0.5_mpp
# python c_compute_prism_slide_level_feats.py --embedding_data_dir datasets/DHMC_20x/features/VirchowFeatureExtractor_v1_concat/imagenet/patch_224_0.5_mpp
# python c_compute_prism_slide_level_feats.py --embedding_data_dir datasets/TCGA-lung/features/VirchowFeatureExtractor_v1_concat/imagenet/patch_224_0.5_mpp --device cuda:1
# python c_compute_prism_slide_level_feats.py --embedding_data_dir datasets/TCIA-CPTAC/features/VirchowFeatureExtractor_v1_concat/imagenet/patch_224_0.5_mpp --device cuda:1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute slide-level embeddings using PRISM model")
    parser.add_argument("--embedding_data_dir", type=str, required=True, help="Directory containing tile-level embedding data")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument("--neg_prompts", type=str, nargs='+', default=["LUAD"], help="Negative prompts for zero-shot classification")
    parser.add_argument("--pos_prompts", type=str, nargs="+", default=["LUSC"], help="Positive prompts for zero-shot classification")
    parser.add_argument("--caption", type=str, default="Lung adenocarcinoma", help="Caption for the model to tokenize and process")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load PRISM model.
    model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
    model.eval()
    model = model.to(device)

    embedding_data_dir = args.embedding_data_dir
    feature_files = sorted([f for f in os.listdir(embedding_data_dir) if f.endswith(".features.pt")])

    for feature_file in feature_files:
        embedding_data_path = os.path.join(embedding_data_dir, feature_file)
        embedding_data = torch.load(embedding_data_path)
        tile_embeddings = embedding_data.unsqueeze(0).to(device)
        if args.verbose:
            print("tile_embeddings.shape", tile_embeddings.shape)

        with torch.autocast("cuda", torch.float16), torch.inference_mode():
            reprs = model.slide_representations(tile_embeddings)
            if args.verbose:
                print(reprs["image_embedding"].shape)
                print(reprs["image_latents"].shape)

            # Save slide-level "image_embedding"
            slide_embedding_path = embedding_data_path.replace(".features.pt", ".image_embedding.pt")
            slide_embedding_path = slide_embedding_path.replace("/features/", "/slide_features/")
            slide_embedding_path = slide_embedding_path.replace("/VirchowFeatureExtractor_v1_concat/", "/PRISM/")
            os.makedirs(os.path.dirname(slide_embedding_path), exist_ok=True)

            torch.save(reprs["image_embedding"].cpu(), slide_embedding_path)
            print(f"Slide-level image_embedding saved to {slide_embedding_path}")

            # # probabilities for negative and positive prompts
            # scores = model.zero_shot(
            #     reprs["image_embedding"],
            #     neg_prompts=args.neg_prompts,
            #     pos_prompts=args.pos_prompts,
            # )
            # print("Scores for negative and positive prompts:", scores)

            # # generated caption for the slide
            # genned_ids = model.generate(
            #     key_value_states=reprs["image_latents"],
            #     do_sample=False,
            #     num_beams=5,
            #     num_beam_groups=1,
            # )
            # genned_caption = model.untokenize(genned_ids)
            # if args.verbose:
            #     print(genned_caption)

            # # similarity with a given caption
            # caption = model.tokenize([args.caption]).to(device)
            # output = model(input_ids=caption, tile_embeddings=tile_embeddings)
            # if args.verbose:
            #     for key in output.keys():
            #         print(key, output[key])
            #         print()

        if args.verbose:
            print("\n", "="*80, "\n")