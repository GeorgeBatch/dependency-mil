# Dependency-MIL

## Accurate Subtyping of Lung Cancers by Modelling Class Dependencies

[[`Paper`](https://ieeexplore.ieee.org/document/10635232)] [[`Pre-print`](https://ora.ox.ac.uk/objects/uuid:4966840e-ccef-4fbf-b5fb-6cf0376d9aaa)] [[`Code`](https://github.com/GeorgeBatch/dependency-mil)] [[`BibTeX`](#Citation)]

**Has been presented in 2024 at the 21th International Symposium on Biomedical Imaging (ISBI-2024).**

Authors: George Batchkala, Bin Li, Mengran Fan, Mark McCole, Cecilia Brambilla, Fergus Gleeson, Jens Rittscher.

## Creation of the Multi-label Dataset

### Source files used to make the labels

* [DHMC_MetaData_Release_1.0.csv](labels/source_copies_for_label_files/DHMC_MetaData_Release_1.0.csv) - downloaded from https://bmirds.github.io/LungCancer/; gives predominant LUAD pattern

* [tcga_classes_extended_info.csv](labels/source_copies_for_label_files/tcga_classes_extended_info.csv) - see https://github.com/GeorgeBatch/TCGA-lung-histology-download/
* [tcga_dsmil_test_ids.csv](labels/source_copies_for_label_files/tcga_dsmil_test_ids.csv) - see https://github.com/GeorgeBatch/TCGA-lung-histology-download/

* [tcia_cptac_md5sum_hashes.txt](labels/source_copies_for_label_files/tcia_cptac_md5sum_hashes.txt) - see https://github.com/GeorgeBatch/TCIA-CPTAC-lung-histology-download
* [tcia_cptac_luad_lusc_cohort.csv](labels/source_copies_for_label_files/tcia_cptac_luad_lusc_cohort.csv) - see https://github.com/GeorgeBatch/TCIA-CPTAC-lung-histology-download
* [tcia_cptac_string_2_ouh_labels.csv](labels/source_copies_for_label_files/tcia_cptac_string_2_ouh_labels.csv) - took unique values from [tcia_cptac_luad_lusc_cohort.csv](labels/source_copies_for_label_files/tcia_cptac_luad_lusc_cohort.csv) and manually mapped to labels inspired by OUH (Oxford University Hospitals) reports


### Dummy label files

Columns include the `label` (LUAD vs LUSC) and paths to features:
* `features_csv_file_path`
* `h5_file_path`
* `pt_file_path`

```
mapping = {
    "LUAD": 0,
    "LUSC": 1,
}
```

DHMC has only LUAD slides, so all entries in the `label` field are 0:
* [DHMC_20x.csv](labels/dummy-label-files/DHMC_20x.csv)
* [DHMC_40x.csv](labels/dummy-label-files/DHMC_40x.csv)

TCGA has both LUAD and LUSC so entries in the `label` field include 0 and 1:
* [TCGA-lung-default.csv](labels/dummy-label-files/TCGA-lung-default.csv)
* [TCGA-lung-ms.csv](labels/dummy-label-files/TCGA-lung-ms.csv)

### Run the creation code

Run the labels creation code [notebook](labels_creation_code/make_detailed_labels_for_dhmc_tcga_tcia.ipynb).
The code will create the files in [labels/experiment-label-files/](labels/experiment-label-files/).

**Note, the combined dataset for training/validation is not the same as in the paper since the in-house DART dataset is not publicly available.**
The test set, however, is the same as in the paper and is fully available in the [8-label task](labels/experiment-label-files/DETAILED_COMBINED_HARD_TEST_LUAD_LUSC_BENIGN.csv) and [5-label task](labels/experiment-label-files/DETAILED_COMBINED_HARD_TEST_LUAD_LUSC_BENIGN_AT_LEAST_ONE_KNOWN_PATTERN.csv).


## Running Scripts

- [a_save_slide_metadata.py](./a_save_slide_metadata.py): Saves metadata for all WSIs in a dataset.  
  Example: `python a_save_slide_metadata.py --dataset TCGA-lung --slide_format svs`

- [b_create_thumbnails_and_masks.py](./b_create_thumbnails_and_masks.py): Produces thumbnails and masks for WSIs.  
  Example: `python b_create_thumbnails_and_masks.py --dataset TCGA-lung --slide_format svs`

- [c_compute_tiatoolbox_feats.py](./c_compute_tiatoolbox_feats.py): Extracts patch features for WSIs using tiatoolbox.  
  Example: `python c_compute_tiatoolbox_feats.py --dataset TCGA-lung --slide_format svs`

- [c_record_masked_positions.py](./c_record_masked_positions.py): Records masked positions for WSIs by comparing feature positions with mask intersections.  
  Example: `python c_record_masked_positions.py --dataset TCGA-lung --slide_format svs --min_mask_ratio 0.1`

- [c_record_positions_intersections.py](./c_record_positions_intersections.py): Records intersections between slide feature positions and mask intersection positions.  
  Example: `python c_record_positions_intersections.py --num_workers 24`

- [d_train_classifier.py](./d_train_classifier.py): Trains a MIL classifier on patch features.  
  * Example 1 (like in the paper): `python d_train_classifier.py --base_config_path ./configs/base_config.yaml --config_path ./configs/combined-configs-sota/simclr-tcga-lung_resnet18-10x_COMBINED-ALL-8-dsmil-wo_subsampling.yaml`
  * Example 2 (subsampling patches): `python d_train_classifier.py --base_config_path ./configs/base_config.yaml --config_path ./configs/combined-configs-sota/simclr-tcga-lung_resnet18-10x_COMBINED-ALL-8-dsmil_config.yaml`
  * Example 3 (with mixed supervision - used in-house data): `python d_train_classifier.py --base_config_path ./configs/base_config.yaml --config_path ./configs/combined-configs-mixed-supervision/simclr-tcga-lung_resnet18-10x_COMBINED-ALL-8-dsmil_config.yaml`

- [e_compute_gigapath_slide_level_feats.py](./e_compute_gigapath_slide_level_feats.py): Computes slide-level embeddings using the prov-gigapath model.  
  Example: `python e_compute_gigapath_slide_level_feats.py --embedding_data_dir datasets/TCGA-lung/features/prov-gigapath/imagenet/patch_224_0.5_mpp`

- [e_compute_prism_slide_caption_similarities.py](./e_compute_prism_slide_caption_similarities.py): Computes slide-level caption similarities using the PRISM model.  
  Example: `python e_compute_prism_slide_caption_similarities.py --embedding_data_dir datasets/TCGA-lung/features/VirchowFeatureExtractor_v1_concat/imagenet/patch_224_0.5_mpp`

- [e_compute_prism_slide_level_feats.py](./e_compute_prism_slide_level_feats.py): Computes slide-level embeddings using the PRISM model.  
  Example: `python e_compute_prism_slide_level_feats.py --embedding_data_dir datasets/TCGA-lung/features/VirchowFeatureExtractor_v1_concat/imagenet/patch_224_0.5_mpp`

- [f_train_linear_probing_classifier.py](./f_train_linear_probing_classifier.py): Trains a linear probing classifier on slide features.  
  Example: `python f_train_linear_probing_classifier.py --base_config_path ./configs/base_config.yaml --config_path ./configs/combined-configs-slide-linear-probing/PRISM_COMBINED-ALL-8-linear_config.yaml`


## Source Contents

### PyTorch Datasets and Data Loaders

The data loading pipeline is implemented using custom PyTorch Datasets and PyTorch Lightning DataModules. Specifically:

- Datasets: [source.data.dataset_detailed](./source/data/dataset_detailed.py) (`LungSubtypingDataset` and `LungSubtypingSlideEmbeddingDataset`) load precomputed features, positional data, and label masks from .pt and .npy files. They also perform on-the-fly subsampling and compute weight masks for instances with unknown labels.
- DataModules: [source.data.datamodule_detailed](./source/data/datamodule_detailed.py) (`LungSubtypingDM` and `LungSubtypingSlideEmbeddingDM`) handle the creation and splitting of datasets based on patient IDs, reading CSV descriptions that reference pre-extracted patch features.

### Dependency Modelling architecture

Dependency-MIL model can be created using `get_model()` function from [source.feature_aggregation.models.combined_model](./source/feature_aggregation/combined_model.py)

It uses the following components:

- Instance Embedders: `IdentityEmbedder`, `AdaptiveAvgPoolingEmbedder`, `LinearEmbedder`, `SliceEmbedder` from [source.feature_aggregation.instance_embedders](./source/feature_aggregation/instance_embedders.py)
- Bag Aggregators: `AbmilBagClassifier`, `DsmilBagClassifier` from [source.feature_aggregation.combined_model](./source/feature_aggregation/combined_model.py)
- Class Connectors: `BahdanauSelfAttention`, `TransformerSelfAttention` from [source.feature_aggregation.class_connectors](./source/feature_aggregation/class_connectors.py)
- Classifier Heads: `LinearClassifier`, `DSConvClassifier`, `CommunicatingConvClassifier` from [source.feature_aggregation.classifier_heads](./source/feature_aggregation/classifier_heads.py)

## Acknowledgements

George Batchkala is supported by Fergus Gleeson and the EPSRC Center for Doctoral Training in Health Data Science (EP/S02428X/1).
The work was done as part of DART Lung Health Program (UKRI grant 40255).

The computational aspects of this research were supported by the Wellcome
Trust Core Award Grant Number 203141/Z/16/Z and the NIHR Oxford BRC. The views
expressed are those of the author(s) and not necessarily those of the NHS, the
NIHR or the Department of Health.

## Citation

If you find Dependency-MIL useful for your your research and applications, please cite using this BibTeX:

```
@INPROCEEDINGS{batchkala2024dependency-mil,
  author={Batchkala, George and Li, Bin and Fan, Mengran and McCole, Mark and Brambilla, Cecilia and Gleeson, Fergus and Rittscher, Jens},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)}, 
  title={Accurate Subtyping of Lung Cancers by Modelling Class Dependencies}, 
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Accuracy;Convolution;Annotations;Histopathology;Lung cancer;Lung;Predictive models;lung cancer;computational pathology;multi-label classification;multiple-instance learning},
  doi={10.1109/ISBI56570.2024.10635232}
}
```
