# Directory
This module contains functions used to manipulate files and directories. These functions include functions to copy, 
rename, and move files as well as functions to take very large raw images (1 GB +) and split them into smaller tiles 
to make them compatible with the annotation and generator pipelines.

## file_utils.py

This script contains functions used match files with matching names but different extensions (e.g. test.bmp and test
.png). Functions to move batches of files based on these criteria are also included.

## general_utils.py

This script contains functions used to extract file lists by name, extension, or other attributes from target 
directories. This script also contains functions that take very large raw images (1 GB +) and split them into smaller tiles 
to make them compatible with the annotation and generator pipelines.


# Examples

A few examples of image splitting operations are shown in the examples below:
 
```python
# Import required libraries and set data location
import bscai
import numpy as np
import os
import shutil
import PIL
data_location = 'examples/data'

############################################################
# Find and split images into smaller tiles:
############################################################

# Get a list of all png images
image_list = bscai.directory.general_utils.get_dir_contents(data_location, '*.png', return_relative_filepaths=False)

# Load the first image as an array and get the dimensions
image_1 = PIL.Image.open(image_list[0])
np.shape(image_1)

# Split the image in half into a list of two images
split_image_1 = bscai.directory.general_utils.split_img_by_dim(image_1, (600, 128))

# Split the directory of images in half and write them to a temporary directory in the data_location
out_directory = os.path.join(data_location, 'out_temp')
os.mkdir(out_directory)
bscai.directory.general_utils.split_images_in_dir(data_location, out_directory, columns=1, rows=2, pattern='*.png')

# Examine the split files
# Get a list of all png images
image_list = bscai.directory.general_utils.get_dir_contents(out_directory, '*.png', return_relative_filepaths=False)

# Load the first image as an array and get the dimensions
image_1 = PIL.Image.open(image_list[0])
np.shape(image_1)

# Remove the split files
shutil.rmtree(out_directory)
```