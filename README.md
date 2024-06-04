# Dependency-MIL

## Accurate Subtyping of Lung Cancers by Modelling Class Dependencies - Accepted to ISBI 2024

[[`Pre-print`](https://ora.ox.ac.uk/objects/uuid:4966840e-ccef-4fbf-b5fb-6cf0376d9aaa)] [[`Code`](https://github.com/GeorgeBatch/dependency-mil)] [[`BibTeX`](#Citation)]

### Creation of the Multi-label Dataset

#### Source files used to make the labels

* [DHMC_MetaData_Release_1.0.csv](labels/source_copies_for_label_files/DHMC_MetaData_Release_1.0.csv) - downloaded from https://bmirds.github.io/LungCancer/; gives predominant LUAD pattern

* [tcga_classes_extended_info.csv](labels/source_copies_for_label_files/tcga_classes_extended_info.csv) - see https://github.com/GeorgeBatch/TCGA-lung-histology-download/
* [tcga_dsmil_test_ids.csv](labels/source_copies_for_label_files/tcga_dsmil_test_ids.csv) - see https://github.com/GeorgeBatch/TCGA-lung-histology-download/

* [tcia_cptac_md5sum_hashes.txt](labels/source_copies_for_label_files/tcia_cptac_md5sum_hashes.txt) - see https://github.com/GeorgeBatch/TCIA-CPTAC-lung-histology-download
* [tcia_cptac_luad_lusc_cohort.csv](labels/source_copies_for_label_files/tcia_cptac_luad_lusc_cohort.csv) - see https://github.com/GeorgeBatch/TCIA-CPTAC-lung-histology-download
* [tcia_cptac_string_2_ouh_labels.csv](labels/source_copies_for_label_files/tcia_cptac_string_2_ouh_labels.csv) - took unique values from [tcia_cptac_luad_lusc_cohort.csv](labels/source_copies_for_label_files/tcia_cptac_luad_lusc_cohort.csv) and manually mapped to labels inspired by OUH (Oxford University Hospitals) reports


#### Dummy label files

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

#### Run the creation code

Run the labels creation code [notebook](labels_creation_code/make_detailed_labels_for_dhmc_tcga_tcia.ipynb).
The code will create the files in [labels/experiment-label-files/](labels/experiment-label-files/).

**Note, the combined dataset for training/validation is not the same as in the paper since the in-house DART dataset is not publicly available.**
The test set, however, is the same as in the paper and is fully available in the [8-label task](labels/experiment-label-files/DETAILED_COMBINED_HARD_TEST_LUAD_LUSC_BENIGN.csv) and [5-label task](labels/experiment-label-files/DETAILED_COMBINED_HARD_TEST_LUAD_LUSC_BENIGN_AT_LEAST_ONE_KNOWN_PATTERN.csv).

#### Dataset and Data Loaders

Code for creating
* PyTorch dataset: [dataset_detailed.py](./source/datasets/dataset_detailed.py).
* PyTorch data loaders using PyTorch Ligtning Datamodule : [datamodule_detailed.py](./source/datasets/datamodule_detailed.py).

### Tiling, Feature Extraction, and Training - Improvements In Progress (last updated: June 4th, 2024)

For publication, I used the tiling and feature extraction pipeline from https://github.com/binli123/dsmil-wsi repository.
For faster computation, the csv features should be converted into `hdf5` and `pt` files like in https://github.com/mahmoodlab/CLAM.
I am currently working on standardising the tiling and feature extraction pipeline for the Dependency-MIL model using [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox).

For training I used the code from https://github.com/binli123/dsmil-wsi modified to accomodate for partial labels using `custom_binary_cross_entropy_with_logits` function from [source.losses](./source/losses.py)

I will release the code once I finish improving it. If you need the code urgently, please contact me.

### Dependency Modelling architecture

Dependency-MIL model can be created using `get_model()` function from [source.models.combined_model](./source/models/combined_model.py)

## Acknowledgements

George Batchkala is supported by Fergus Gleeson and the EPSRC Center for Doctoral Training in Health Data Science (EP/S02428X/1).
The work was done as part of DART Lung Health Program (UKRI grant 40255).

The computational aspects of this research were supported by the Wellcome
Trust Core Award Grant Number 203141/Z/16/Z and the NIHR Oxford BRC. The views
expressed are those of the author(s) and not necessarily those of the NHS, the
NIHR or the Department of Health.

## Citation

If you find Dependency-MIL useful for your your research and applications, please cite using this BibTeX (will be updated once the paper is published by IEEE in ISBI 2024 proceedings):

```
@INPROCEEDINGS{batchkala2024dependency-mil,
  author={Batchkala, George and Li, Bin and Fan, Mengran and McCole, Mark and Brambilla, Cecilia and Gleeson, Fergus and Rittscher, Jens},
  booktitle={2024 IEEE 21th International Symposium on Biomedical Imaging (ISBI)}, 
  title={Accurate Subtyping of Lung Cancers by Modelling Class Dependencies}, 
  year={2024},
  volume={},
  number={},
  pages={...},
  keywords={lung cancer;computational pathology;multi-label classification;multiple-instance learning},
  doi={...}
}
```
