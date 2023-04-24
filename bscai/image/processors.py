"""
This module contains image processor classes to ingest a variety of image data formats. All of the classes in this
module are designed to be used as an image_processor by bscai.generators.data.image_label_from_json function.
"""

from PIL import Image
import numpy as np


class PillowProcessor:
    """
    This class uses the Pillow package to open and process images. It is designed to be used as the image_processor
    argument for bscai.generators.data.image_label_from_json function
    """

    def open(self, path):
        """
        Open an image as a PIL image from a given filepath.

        :param path: The path to the image

        :return: A PIL image object
        """

        return Image.open(path)

    def crop(self, image, crop_dims):
        """
        Crops an image using supplied dimensional tuple by calling the PIL.Image.crop method

        :param image: A PIL image object
        :param crop_dims: A dimensional tuple giving the cropping rectangle dimensions.
            Order of arguments must be: (min height, min width, max height, max width) where the min and max refer
            to the minimum and maximum pixel integer values

        :return: A PIL image object cropped to the dimensions given by crop_dims
        """

        # Convert the crop tuple to the expected PIL.Image.crop ordering of
        # (min width, min height, max width, max height) and crop the image
        return image.crop((crop_dims[1], crop_dims[0], crop_dims[3], crop_dims[2]))

    def np_array(self, image):
        """
        Converts a PIL image object to a numpy array in (height, width, channels) order. The resulting array
        will be forced to have 3 channels, even if it is a monochromatic image.

        :param image: A PIL image object

        :return: A numpy array of the image in (height, width, channels) order.
        """

        img_arr = np.array(image)

        # Compensate for poor PIL decision to force number of channels on opening image with no
        # option to change by copying monochrome channel data to 3 channels if array has only 2 axes
        # (thus a monochrome image)
        if len(np.shape(img_arr)) == 2:
            img_arr = np.stack((img_arr,) * 3, axis=-1)

        return np.array(img_arr)
