"""
This module contains functions that change the shape of an image or segmentation array.
This includes stretching the width and height, rotating, or any other transformations that modify data dimensions.
"""

import numpy as np
import math
from PIL import Image


def find_largest_inscribed_rect(width: int, height: int, rot_angle: float):
    """
    Function to find the with and height of the largest (by area) inscribed rectangle that fits within a rotated
    rectangle. Based on code found here:
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders/16761710#16761710

    Args:
        :param width: The width of the rectangle being rotated (before rotation)
        :param height: The height of the rectangle being rotated (before rotation)
        :param rot_angle: The counter-clockwise rotation angle applied to the rectangle (in degrees).

    Returns:
        :return dictionary containing the width and height of the largest inscribed rectangle that fits entirely
        within the rotated rectangle.

    """

    # Convert angle to radians
    angle = math.radians(rot_angle)

    if width <= 0 or height <= 0:
        return 0, 0

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a

    return {'Width': wr,
            'Height': hr}


def rotate_stretch_img(img: np.ndarray or Image.Image, new_height: int = None, new_width: int = None,
                       new_angle: float = 0, nearest_neighbor: bool = False, expand_to_fit: bool = True,
                       remove_border: bool = True):
    """
    Function to rotate and stretch an image. Height and width transformations are applied before rotation
    transformation. Note that using rotation with expand_to_fit or remove_border arguments as True will produce an
    image canvas that does not match the new_height and new_width dimensions.

    Args:
        :param img: A 2-D (monochrome) or 3-D (3-channel color) numpy array (H x W x C) or PIL Image object
        :param new_height: The new height of the image for height stretching
        :param new_width: The new width of the image for width stretching
        :param new_angle: The counter-clockwise rotation angle to apply (in degrees).
        :param nearest_neighbor: Whether nearest neighbor interpolation should be used. This should almost always be
        set to True for binary label masks and False for image data or other non-binary arrays.
        :param expand_to_fit: Whether the image canvas should be expanded to fit the entire image after rotation or
        whether the canvas size should remain with the non-rotated dimensions, resulting in the corners of the
        rotated image being clipped off.
        :param remove_border: Whether the image should be cropped to eliminate any region that will show black
        border space after rotation.

    Returns:
        :return The transformed numpy array or PIL Image (matches input type)

    """

    # If a numpy array was provided, convert to PIL image for transformation
    if isinstance(img, np.ndarray):
        img_out = Image.fromarray(img)

    else:
        img_out = img

    # If no argument was provided for new_height or new_width, set them to the current image dimensions
    if new_height is None:
        new_height = img_out.height
    if new_width is None:
        new_width = img_out.width

    # Stretch image if applicable
    if new_height != img_out.height or new_width != img_out.width:
        # NOTE: PIL counter-intuitively expects resize dimensions to be provided as(width, height)
        if nearest_neighbor:
            img_out = img_out.resize((new_width, new_height), resample=Image.NEAREST)
        else:
            img_out = img_out.resize((new_width, new_height), resample=Image.LANCZOS)

    # Rotate image if applicable
    if new_angle != 0:

        # Get height and width of image before rotation:
        pre_rot_height = img_out.height
        pre_rot_width = img_out.width

        # Perform rotation with requested border control (always expand to fit if removing borders)
        if nearest_neighbor:
            img_out = img_out.rotate(angle=new_angle, resample=Image.NEAREST, expand=expand_to_fit or remove_border)
        else:
            img_out = img_out.rotate(angle=new_angle, resample=Image.BICUBIC, expand=expand_to_fit or remove_border)

        # Crop border regions that would include blank spaces if requested
        if remove_border:

            # Find width and height of largest bounding rectangle that can be inscribed in rotated image
            rect = find_largest_inscribed_rect(width=pre_rot_width, height=pre_rot_height, rot_angle=new_angle)

            # Crop image to inscribed rectangle (always around center point)
            img_out = img_out.crop((int((img_out.width - rect['Width']) * 0.5),
                                    int((img_out.height - rect['Height']) * 0.5),
                                    int((img_out.width + rect['Width']) * 0.5),
                                    int((img_out.height + rect['Height']) * 0.5)))

    # If image was provided as numpy array, convert back to numpy array with correct number of channels:
    if isinstance(img, np.ndarray):
        # Convert back to image array
        img_out = np.array(img_out)

        # Compensate for poor PIL decision to force number of channels on opening image with no option to change
        # Copy monochrome channel data to 3 channels if array has only 2 axes (thus monochrome image) and input image
        # had 3 channels
        if len(np.shape(img_out)) == 2 and len(np.shape(img)) == 3:
            img_out = np.stack((img_out,) * 3, axis=-1)

    return img_out


def rotate_stretch_arr(seg_arr: np.ndarray, **kwargs):
    """
    Function to rotate and stretch a 3 dimensional array with an arbitrary number of channels by passing each
    individual channel to rotate_stretch_img.

    Args:
        :param seg_arr: A 3-D numpy array (H x W x C) with any number of channels. Single channel array should still
        be 3-D (e.g. 2 x 4 x 1). Data type must be supported by PIL.Image.fromarray
        :param kwargs: Additional arguments passed to rotate_stretch_img

    Returns:
        :return The transformed numpy array

    """

    # Create list to hold output arrays
    seg_list = []

    # Transform each channel of segmentation array one by one
    for i in range(0, np.shape(seg_arr)[2]):

        # Transform current channel and append to list
        seg_temp = rotate_stretch_img(img=seg_arr[:, :, i], **kwargs)
        seg_list.append(seg_temp)

    # Convert list back to numpy array of correct dimensionality
    seg_out = np.stack(seg_list, axis=-1)

    return seg_out
