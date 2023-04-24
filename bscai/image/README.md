# Image
This module contains functions used to manipulate image data in array or raw image form. This module is broken down 
into the following source scripts:

## augmentation.py

#### deprecated

This script contains functions to perform image augmentation on existing images using methods that do not change the 
size or shape of the image (e.g. sharpening, modifying the brightness, etc.). This script relies on the imgaug 
package. The functions in this module are not recommended for use as they are simply aliases for functionality 
already available in the imgaug package.
**This script will not be updated or officially supported in future releases.**

## image_array_functions.py

#### deprecated

This script contains functions to modify crop images and draw label overlays on the images using x y coordinate 
values for the label locations.
**This script will not be updated or officially supported in future releases.**

## merge_depth_channel.py

This script contains very specific functions used to stack the top half of an image onto the blue channel of the 
bottom half of an image. The images must be monochrome images for this function to work properly. These functions are
specifically designed to work with images produced by the GDO inspection system which contains a monochrome depth map
on the top half of the image.

## processors.py

This script contains image processing classes that are designed for use with the data generators in the bscai
.generators.data module. Each class in this script is designed to be compatible with the image_processor argument of
 bscai.generators.data.image_label_from_json generator

## reshape.py

This script contains functions that change the shape of an image or segmentation array. This includes stretching the 
width and height, rotating, or any other transformations that modify image data dimensions.

## sample.py

This script contains functions used to randomly or semi-randomly sample tiles from an image or array that meet 
certain required conditions (e.g. yield a 100 x 200 pixel tile that contains at least 15% of the total pixels 
labeled with a given defect type)


# Examples

These functions can be used to modify and augment image data as shown in the examples below:
 
```python
# Import required libraries and set data location
import bscai
import numpy as np
import os
from PIL import Image

# Set data location
data_location = 'examples/data'


# Load an image and corresponding segmentation array
image_PIL = Image.open(os.path.join(data_location, '118-2 Bild000040_bottom.png'))
image_array = np.array(image_PIL)
json_data = bscai.annotation.load.get_json_data(os.path.join(data_location, '118-2 Bild000040_bottom.json'))
seg_array = bscai.annotation.load.regions_to_array(json_data['annotations_list'][0], ['Missing Material Partial Width'], 
tuple(json_data['image_shape']))

# View the original image
image_PIL.show()

############################################################
# Reshape the image:
############################################################

# Rotate the image and expand the canvas to include the whole image
image_rotated = bscai.image.reshape.rotate_stretch_img(image_array, new_angle = 15, remove_border=False)
Image.fromarray(image_rotated).show()

# Perform the same rotation on the segmentation array (uses nearest neighbor interpolation to preserve binary nature 
# of label data as shown by result of np.unique command)
seg_rotated = bscai.image.reshape.rotate_stretch_arr(seg_array, new_angle = 15, remove_border=False)
np.unique(seg_rotated)

# Rotate the image but resize the canvas to the largest rectangle that does not contain any black border region that 
# was outside of the original image canvas
image_rotated = bscai.image.reshape.rotate_stretch_img(image_array, new_angle = 15)
Image.fromarray(image_rotated).show()

# Rotate and stretch the image at the same time (stretching is applied 
image_rotated = bscai.image.reshape.rotate_stretch_img(image_array, new_angle = 15, new_width = 1200)
Image.fromarray(image_rotated).show()

############################################################
# Sample sub-sections of the image:
############################################################

# Try to pull an image tile at random that contains all of the defective pixels in the image:
tile_start = bscai.image.sample.random_tile(seg_array, tile_shape=(100, 50), max_tries = 100000, min_labeled=seg_array.sum())

# Check to see if the tile pull was successful
tile_start['Success']

# See how many tries it took to find a tile that contained all of the pixels
tile_start['NumberOfTries']

# Create a slicing object from the tile_start object
slicer = bscai.image.sample.tile_slice(tile_start['StartPoint'], (100, 50))

# Extract the tile from the image and corresponding segmentation array using the slicer
image_tile = image_array[slicer]
Image.fromarray(image_tile).show()
seg_tile = seg_array[slicer]

# Show that the segmentation tile contains the entire defect
seg_tile.sum() - seg_array.sum()

```