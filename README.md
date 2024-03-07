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
* [tcia_cptac_string_2_ouh_labels.csv](labels/source_copies_for_label_files/tcia_cptac_string_2_ouh_labels.csv) - took unique values from [tcia_cptac_luad_lusc_cohort.csv](labels/source_copies_for_label_files/tcia_cptac_luad_lusc_cohort.csv) and manually mapped to labels


