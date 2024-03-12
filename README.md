# Accurate Subtyping of Lung Cancers by Modelling Class Dependencies - Accepted to ISBI 2024

This code was built upon the https://github.com/binli123/dsmil-wsi repository. So the organisation structure is largely 
of the datasets folder is inherited. For faster computation, the csv features were converted into `hdf5`
and `pt` files like in https://github.com/mahmoodlab/CLAM.  

## Creation of the Multi-label Dataset

### Source files used to make the labels

* [DHMC_MetaData_Release_1.0.csv](labels/source_copies_for_label_files/DHMC_MetaData_Release_1.0.csv) - downloaded from https://bmirds.github.io/LungCancer/; gives predominant LUAD pattern

* [tcga_classes_extended_info.csv](labels/source_copies_for_label_files/tcga_classes_extended_info.csv) - see https://github.com/GeorgeBatch/TCGA_lung/
* [tcga_dsmil_test_ids.csv](labels/source_copies_for_label_files/tcga_dsmil_test_ids.csv) - see https://github.com/GeorgeBatch/TCGA_lung/

* [tcia_cptac_md5sum_hashes.txt](labels/source_copies_for_label_files/tcia_cptac_md5sum_hashes.txt) - see https://github.com/GeorgeBatch/TCIA-CPTAC-download-instructions
* [tcia_cptac_luad_lusc_cohort.csv](labels/source_copies_for_label_files/tcia_cptac_luad_lusc_cohort.csv) - see https://github.com/GeorgeBatch/TCIA-CPTAC-download-instructions
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
