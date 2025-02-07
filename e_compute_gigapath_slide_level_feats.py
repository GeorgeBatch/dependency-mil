"""
Use pre-trained prov-gigapath slide aggregator to compute slide-level embeddings.
Takes prov-gigapath concatenated embeddings as inputs.

Page: https://huggingface.co/prov-gigapath/prov-gigapath
"""

import os
import argparse

import torch
import glob

from pathlib import Path

import source.feature_aggregation.gigapath.slide_encoder as provgigapath_slide_encoder

# python c_compute_gigapath_slide_level_feats.py --embedding_data_dir datasets/DHMC_40x/features/prov-gigapath/imagenet/patch_224_0.5_mpp
# python c_compute_gigapath_slide_level_feats.py --embedding_data_dir datasets/DHMC_20x/features/prov-gigapath/imagenet/patch_224_0.5_mpp
# python c_compute_gigapath_slide_level_feats.py --embedding_data_dir datasets/TCGA-lung/features/prov-gigapath/imagenet/patch_224_0.5_mpp
# python c_compute_gigapath_slide_level_feats.py --embedding_data_dir datasets/TCIA-CPTAC/features/prov-gigapath/imagenet/patch_224_0.5_mpp
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute slide-level embeddings using prov-gigapath model")
    parser.add_argument("--embedding_data_dir", type=str,
                        default="datasets/TCIA-CPTAC_test/features/prov-gigapath/imagenet/patch_224_1.0_mpp",
                        help="Directory containing tile-level embedding data")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to run the model on")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    parser.add_argument("--save_all_layer_embed", action="store_true", help="Save embeddings from all layers")
    args = parser.parse_args()

    assert os.path.exists(args.embedding_data_dir), f"Directory {args.embedding_data_dir} does not exist"

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load prov-gigapath model, might need to use huggingface `login()` to access the gated model 
    model = provgigapath_slide_encoder.create_model(
        model_arch="gigapath_slide_enc12l768d",
        # pretrained="hf_hub:prov-gigapath/prov-gigapath",  # first time, afterwards can use local cache
        pretrained=os.path.join(
            os.path.expanduser("~"),
            ".cache/huggingface/hub/models--prov-gigapath--prov-gigapath/blobs/slide_encoder.pth",
        ),
        in_chans=1536,
        global_pool=True,  # like in https://github.com/prov-gigapath/prov-gigapath/blob/main/demo/run_gigapath.ipynb
    )
    model.eval()
    model = model.to(device)

    embedding_data_paths = sorted(glob.glob(os.path.join(args.embedding_data_dir, "*.features.pt")))

    for i, embedding_data_path in enumerate(embedding_data_paths):
        embedding_data = torch.load(embedding_data_path)
        coords_data_path = embedding_data_path.replace(".features.pt", ".positions.pt")
        coords = torch.load(coords_data_path)[:,:2] # get all rows, only first 2 columns

        # expand batch dimension
        tile_embeddings = embedding_data.unsqueeze(0).to(device)
        tile_coords = coords.unsqueeze(0).to(device)

        if args.verbose:
            print("tile_embeddings.shape", tile_embeddings.shape)
            print("tile_coords.shape", tile_coords.shape)

        with torch.autocast("cuda", torch.float16), torch.inference_mode():
            # TODO: check subsetting if increasing batch size to > 1
            slide_embeddings = model(tile_embeddings, tile_coords, all_layer_embed=True)[0]
        
        outputs_dict = {"layer_{}_embed".format(i): slide_embeddings[i].cpu() for i in range(len(slide_embeddings))}
        outputs_dict["last_layer_embed"] = slide_embeddings[-1].cpu()
        if args.verbose:
            print("outputs_dict.keys()", outputs_dict.keys())

        assert torch.allclose(
            outputs_dict["layer_12_embed"],
            outputs_dict["last_layer_embed"],
        ), "The last layer embedding should be the same as the layer 12 embedding"

        if args.verbose:
            print(Path(embedding_data_path).name, "aggregated into:", outputs_dict["last_layer_embed"].shape, outputs_dict["last_layer_embed"][:,:5])

        # Save slide embeddings from all layers
        for layer, embedding in outputs_dict.items():
            if not args.save_all_layer_embed and layer != "last_layer_embed":
                continue
            slide_embedding_path = embedding_data_path.replace(".features.pt", f".{layer}.pt")
            slide_embedding_path = slide_embedding_path.replace("/features/", "/slide_features/")
            slide_embedding_path = slide_embedding_path.replace("/prov-gigapath/", "/prov-gigapath/") # stays the same for prov-gigapath
            os.makedirs(os.path.dirname(slide_embedding_path), exist_ok=True)
            torch.save(embedding, slide_embedding_path)
            if args.verbose:
                print(f"Slide-level {layer} saved to {slide_embedding_path}")
        
        print(f"{i+1}/{len(embedding_data_paths)} | Processed {Path(embedding_data_path).name} \n")