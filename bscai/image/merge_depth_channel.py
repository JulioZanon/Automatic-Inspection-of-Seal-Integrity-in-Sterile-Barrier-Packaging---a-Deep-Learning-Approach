# -*- coding: utf-8 -*-
"""
@author: Varun
"""
import cv2
import os
from bscai.directory.general_utils import exist_file


def add_filename_suffix(filename: str, end_string: str = '_bottom') -> str:
    """
    Attaches end_string to end of output image filename
    Args
        :param filename: Name of file to be augmented
        :param end_string: String to append to filename (will be placed before file extension).
    Returns:
        :return: String of filename with appended end_string before the file extension
    """

    name, ext = os.path.splitext(filename)
    return "{name}{uid}{ext}".format(name=name, uid=end_string, ext=ext)


def merge_depth_channel(input_img_path: str, output_img_path) -> bool:
    """
    Adds top half of image to the bottom half as depth channel and returns as
    single composite image.
    Args
        :param input_img_path: The path to image to be split and have depth channel added.
        :param output_img_path: The desired path of the output image
    Returns:
        :return: True if the output file was produced, False otherwise
    """
    if not exist_file(input_img_path):
        return False

    # Read in original image
    image = cv2.imread(input_img_path, 0)
    height, width = image.shape[:2]

    start_row, start_col = int(0), int(0)
    end_row, end_col = 1200, 128

    # Mark top half of image
    cropped_top = image[start_row:end_row, start_col:end_col]

    start_row, start_col = 1200, int(0)
    end_row, end_col = int(height), int(width)

    # Mark bottom half of image
    cropped_bot = image[start_row:end_row, start_col:end_col]

    # Convert to 3 channel image
    bottom_rgb = cv2.cvtColor(cropped_bot, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(bottom_rgb)

    # Set blue channel to top half image array
    b = cropped_top
    final_img_with_depth = cv2.merge((b, g, r))

    # Check if have permission for writing to file and file can be created or not.
    try:
        cv2.imwrite(output_img_path, final_img_with_depth)
    except FileNotFoundError:
        return False
    except PermissionError:
        return False

    return True


def convert_image_directory(input_directory: str, output_directory: str) -> bool:
    """ 
    Converts a directory of images to include a depth channel . 
    
    Args:
        :param input_directory: Name of the annotation file from which the annotation data should be loaded.
        :param output_directory: The path to the subdirectory where the annotation file is present.
    Returns:
        :return: True if successful, False otherwise
    """
    if not os.path.isdir(input_directory):
        return False

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    files = os.listdir(input_directory)
    for filename in files[:]:  # files[:] makes a copy of filelist.
        if not(filename.endswith(".png")):
            files.remove(filename)
    if len(files) <= 0:
        print("Could not find image files. Check directory name and try again.")
        return False

    i = 0
    for filename in files:
        
        output_path = os.path.join(output_directory, filename)
        output_path = add_filename_suffix(output_path)

        input_path = os.path.join(input_directory, filename)
        
        try:
            if not merge_depth_channel(input_path, output_path):
                print("Failed on: ")
                print(input_path)
            i += 1
        except ValueError as e:
            print(e)

    print("Sucessfully merged depth channel for image files.")

    return True
