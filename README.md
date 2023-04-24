# 2023_TF01
Code to support paper: "Automatic Inspecting of Seal Integrity in Sterile Barrier Packaging: a Deep Learning Approach"

Some of the folders required to run scripts are missing from repository. Please run "00_SetUp_Project_first_Use.py" first to create missing folders.

If you wish to re-train all models as per paper, then run "02a_Build_Model.py". This script will build a training plan with all the serach files in "1\models\00_search_files\00_cue". This project contains all the scrit files used in the paper, that is a total of 432!

Bayesian seraches required TensorBoard.

If you want to test trained models against the Test dataset, add the model.h5 and the model.json to folder "\models\01_trained" run "02b_Test_Model.py".

Dependencies to other repositories:

- Image Dataset: While this repository contains the csv files with the sub-splits, images need to be imported from https://doi.org/10.5281/zenodo.7834077
- Trained Models: If you want to test the pre-trained models, files can be placed in  "\models\01_trained". These files are in the following repositories:
          - No Augmentation: https://doi.org/10.5281/zenodo.7859697
          - Augmentation:  https://doi.org/10.5281/zenodo.7858003
