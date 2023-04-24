# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:55:24 2019

@author: klappec and varun and munjalk
"""

import os
import ntpath
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from bscai.directory.general_utils import sha256_checksum, get_shape_dict, get_name_filepath_dict
from bscai.annotation.raw_annotation_data_loader import get_all_json_data_from_annotation_file, \
    get_regions_dictionary_from_annotation_file_data, format_coordinates_list, bin_array_2_rle,\
    bypass_image_name_from_annotation_data

Image.MAX_IMAGE_PIXELS = None


# Encoder to pass in numpy objects into file, allows compatibility with json.dump() function
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def json_format_upgrader(input_file_or_dir: str, output_directory: str,
                         image_directory: str = None, image_shape=None) -> bool:
    """
    Detects the correct format and converts to the current bscai supported JSON format as appropriate.
    Can be slow, but you should only have to run it once. Makes mulitple passes
    over the images, be patient with lots of large images.
    If an image directory or image_shape tuple is not supplied, will only convert JSON files that contain image
    shape information in the JSON file. If image_directory is not supplied, sha-256 can't be added, and a warning is
    thrown.

    **For now, assumes images have unique filenames.** Works recursively.

    If -01 annotations without sha256 are passed, and image_dirctory is given,
    sha256 will be added.

    Image names must match the json 'fileref' attribute (with extension stripped off)

    Args:
        :param input_file_or_dir: The path to the json file or subdirectory containing the JSON data to be converted.
        :param output_directory: The path to the subdirectory where the new JSON data will be placed.
        :param image_directory: The path to the directory where the images are stored.
        :param image_shape: tuple of (width, height) If provided and image_directory is not passed,
            this will be used as as the image shape. All images must be this shape.

    Returns:
        :return: True if JSON files are converted, False otherwise
    """

    if not (os.path.isdir(input_file_or_dir) or os.path.isfile(input_file_or_dir)):
        print(f'{input_file_or_dir} could not be found.')
        return False

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    if os.path.isdir(input_file_or_dir):
        files = os.listdir(input_file_or_dir)
    else:
        files = [input_file_or_dir]

    for file in files[:]:  # files[:] makes a copy of filelist.
        if not (file.endswith(".json")):
            files.remove(file)
    if len(files) <= 0:
        print(f"No JSON files found at {input_file_or_dir}.")
        return False

    shape_dict = None
    name_filepath_dict = None

    if image_directory:
        shape_dict = get_shape_dict(image_directory)
        name_filepath_dict = get_name_filepath_dict(image_directory)

    for filename in files:

        # Initialize file paths for this json file
        input_path = os.path.join(input_file_or_dir, filename)
        output_path = os.path.join(output_directory, filename)

        # Initialize loop variables for this file
        shape = None

        with open(input_path, 'rt') as f:
            input_data = json.load(f)

        # check if it's in coco format, move to 001 format. Done, nothing else to do.
        if input_data.get('categories'):
            try:
                if not convert_json_schema_coco_01(input_data, output_directory):
                    print("Failed on: ")
                    print(input_path)
            except ValueError as e:
                print("Failed converting coco to 01 on: ")
                print(input_path)
                print(e)

        # check if already 01 and sha256 needs to be added
        elif input_data.get('annotation'):
            if input_data.get('annotation')[0].get('annotation_version')=='0.1.0.0':
                if not image_directory:
                    print("WARNING: 01 annotation file provided with no image_directory.")
                    print("Failed on ", filename)
                else:
                    # create filewriter
                    output_file_path = os.path.join(output_directory,filename)
                    # Check if have permission for writing to file and file can be created or not.
                    try:
                        annotation_file_writer = open(output_file_path, 'w')

                    except FileNotFoundError:
                        print("File not found error.")
                        return False
                    except PermissionError:
                        print("Permissions error when closing annotation_file_writer.")
                        annotation_file_writer.close()
                        return False

                    # get checksum
                    name = input_data['file_data']['file_ref']
                    name = ntpath.basename(name)
                    file_ref_checksum = sha256_checksum(name_filepath_dict[name])
                    input_data['annotation'][0]['checksum'] = file_ref_checksum

                    # dump data
                    json.dump(input_data, annotation_file_writer, cls=MyEncoder)

                # if there's no image directory and it's already 0.1.0.0, there's nothing else to do
        else:
            annotation_dict = bypass_image_name_from_annotation_data(input_data)
            name = annotation_dict['fileRef']
            name = ntpath.basename(os.path.splitext(name)[0])

            # If the annotation file contains the image shape, use that
            if 'img_height' in list(annotation_dict.keys()) and 'img_width' in list(annotation_dict.keys()):
                shape = (annotation_dict['img_height'], annotation_dict['img_width'])

            # Otherwise, use the shape from the image directory if provided
            elif image_directory:
                shape = shape_dict.get(name)

            # Lastly, use the image_shape argument if provided
            else:
                shape = image_shape

            if not shape:
                print("Failed on: ", input_path)
                print(f"Could not find associated image file for {filename}")
                continue

            # check if we need to convert to RLE from x-y coordinate data
            try:
                annotation_data = bypass_image_name_from_annotation_data(input_data)
                key = list(annotation_data['regions'].keys())[0]
                all_points_x = annotation_data['regions'][key]['shape_attributes'].get('all_points_x')
            except:
                all_points_x = None
            # if there is data in all_points_x, and we need to convert to RLE, then:
            if all_points_x is not None:

                try:
                    retval = convert_json_xy_to_rle(input_path, output_path, shape)

                    if not retval:
                        print("Failed failed converting xy to rle on: ", input_path)
                except Exception as e:
                    print("Failed failed converting xy to rle on: ", input_path)
                    print(e)
                    continue

                # now that we converted it, re-read the data in again
                with open(output_path, 'rt') as f:
                    input_data = json.load(f)
                input_path=output_path # needed so so it can re-convert from 003 to 01
            # otherwise, then convert 0.0.3 to 001
            try:
                if not convert_json_schema_003_to_01(input_path, output_path, name_filepath_dict):
                    print("Failed converting 003 to 01 on: ")
                    print(input_path)

            except ValueError as e:
                print("Failed converting 003 to 01 on: ")
                print(input_path)
                print(e)
                continue

    print("Successfully converted JSON files.")
    return True


def xy_pairs_to_rle(x_coordinate_list: list, y_coordinate_list: list, img_shape: tuple) -> list:
    """
    Converts arrays of point-wise xy coordinate data into an array of RLE-encoded data

    Args:
       :param x_coordinate_list: a list of x coordinates loaded from the coordinates section corresponding to a
       particular defect from the annotation file
       :param y_coordinate_list: a list of y coordinates loaded from the coordinates section corresponding to a
       particular defect from the annotation file.
       :param img_shape: Tuple defining shape of binary mask (height x width x channels)
    Returns:
       :return: List containing converted RLE-encoded data
    """
    if len(x_coordinate_list) == 0 and len(y_coordinate_list) == 0:
        return []
    # Create binary mask from coordinate data
    out_arr = np.zeros(shape=img_shape)
    out_arr[y_coordinate_list, x_coordinate_list] = 1

    # Convert binary mask to rle format
    rle_array = bin_array_2_rle(out_arr)

    return rle_array


def regions_dict_xy_to_rle(regions_dict: dict, img_shape: list) -> bool:
    """
    Converts each region in a regions dictionary from xy coordinate format to RLE format

    Args:
        :param regions_dict:A dictionary that contains the mappings from region number to region information
        :param img_shape: Tuples defining shape of the binary mask for the image
        (height x width x channels)
     Returns:
        :return: True if successful, False otherwise
    """
    # Check whether regions_dict is valid and not empty:
    if not regions_dict:
        print("Region dict empty or missing.")
        return False
    for region in regions_dict.values():

        region_shape_attribs = region['shape_attributes']

        # Load the x and y coordinates list from point-wise encoded (uncompressed) data
        defect_region_x_coordinates_list = format_coordinates_list(region_shape_attribs['all_points_x'])
        defect_region_y_coordinates_list = format_coordinates_list(region_shape_attribs['all_points_y'])
        RLE = xy_pairs_to_rle(defect_region_x_coordinates_list, defect_region_y_coordinates_list, img_shape)

        region_shape_attribs['name'] = 'run_length'

        if len(RLE) == 0:
            region_shape_attribs['all_points'] = [img_shape[0] * img_shape[1]]
        else:
            region_shape_attribs['all_points'] = RLE

        del region_shape_attribs['all_points_x']
        del region_shape_attribs['all_points_y']

    return True


def convert_json_xy_to_rle(input_data: str, output_file_path: str, img_shape) -> bool:
    """
    Converts a single JSON file from the original xy pointwise JSON schema version < 0.0.3.0 to JSON schema version
    0.0.3.0

    Args:
        :param input_data: The JSON data loaded as a dict or the file path to the non-RLE encoded annotation file
        :param output_file_path: The desired path for the new RLE-encoded annotation file
        :param img_shape: Tuple defining shape of binary mask (height x width x channels)

    Returns:
        :return: True if the output file was produced, False otherwise
    """

    # Check if have permission for writing to file and file can be created or not.
    try:
        annotation_file_writer = open(output_file_path, 'w')

    except FileNotFoundError:
        print("File not found error.")
        return False
    except PermissionError:
        print("Permissions error when closing annotation_file_writer.")
        annotation_file_writer.close()
        return False

    # Load the Annotation Data Dictionary if input_data is a file path
    if isinstance(input_data, str):
        input_data = get_all_json_data_from_annotation_file(input_data)

    # Unpacking the first element to annotation_metadata
    # Original JSON schema has only 1 element
    annotation_dict = input_data[list(input_data.keys())[0]]

    # add missing metadata
    annotation_dict['img_height'] = img_shape[0]
    annotation_dict['img_width'] = img_shape[1]
    annotation_dict['annotation_version'] = '0.0.3.0'

    # Load only the regions part of the annotation data
    regions_dict = get_regions_dictionary_from_annotation_file_data(input_data)

    # Change the regions dict to RLE
    if not regions_dict_xy_to_rle(regions_dict, img_shape):
        print("regions_dict_xy_to_rle failure.")
        return False

    # Write the data to the new annotation file
    json.dump(input_data, annotation_file_writer, cls=MyEncoder)
    annotation_file_writer.close()

    return True


def convert_json_schema_003_to_01(input_data: str, output_file_path: str, file_ref_dict: list = None) -> bool:
    """
    This function is responsible for converting JSON files from JSON schema version 0.0.3.0 to JSON schema version
    0.1.0.0

    Args:
        :param input_data: The JSON data loaded as a dict or the file path to the non-RLE encoded annotation file
        :param output_file_path: The desired path for the new JSON data file
        :param file_ref_dict: A list of files to attempt to match using the file_ref property of the input json file.
        If this argument is provided, the conversion process will include the SHA-256 checksum of this file in the
        output JSON.

    Returns:
        :return: True if the output file was produced, False otherwise
    """
    # Extract top level key to access old data in new schema spec
    if isinstance(input_data, str):
        with open(input_data, 'rt') as f:
            input_data = json.load(f)
    else:
        input_data = input_data
    top_level = next(iter(input_data))
    regions_dict = get_regions_dictionary_from_annotation_file_data(input_data)
    values = list(regions_dict.values())

    file_ref_checksum = ""

    if file_ref_dict is not None:
        file_ref_checksum = sha256_checksum(file_ref_dict[ntpath.basename(input_data[top_level]["fileRef"])])

    spec = {
        "file_data": {
            "file_ref": input_data[top_level]["fileRef"],
            "user_name": input_data[top_level]["userName"],
            "created_timestamp": "",
            "size": input_data[top_level]["size"],
            "shape": ""
        },
        "annotation": [
            {
                "run_number": 0,
                "checksum": file_ref_checksum,
                "annotation_version": "0.1.0.0",
                "regions": ""
            }
        ]

    }
    # Populate height and width for RLE JSON data
    if ("img_height" in input_data[top_level] and "img_width" in input_data[top_level]):
        spec["file_data"]["shape"] = [
            input_data[top_level]["img_height"],
            input_data[top_level]["img_width"]]

    # Add regions dictionary into new file
    spec["annotation"][0]["regions"] = values

    try:
        annotation_file_writer = open(output_file_path, 'w')
    except FileNotFoundError:
        return False
    except PermissionError:
        return False
    json.dump(spec, annotation_file_writer)

    return True


def replaceMultiple(mainString, toBeReplaces, newString):
    """
    Replace a set of multiple sub strings with a new string in main string.

    Args:
        :param mainString: It takes in the string in which replacement needs done
        :param toBeReplaces: It contains the substring inside mainString to be replaced
        :param newString: It contains the replacement string to be used.

    Returns:
        :returns mainString with string replaced.
    """
    # Iterate over the strings to be replaced
    for elem in toBeReplaces:
        # Check if string is in the main string
        if elem in mainString:
            # Replace the string
            mainString = mainString.replace(elem, newString)

    return mainString


def convert_json_schema_coco_01(input_data: str or dict, output_directory: str, stop_on_error: bool = True) -> bool:
    """
    This function is responsible for converting JSON files from the COCO segmentation schema into
    JSON schema version 0.1.0.0

    Args:
        :param input_data: A file path to a COCO formatted JSON or a python dictionary representing a coco .json
            annotation file, usually read in with json.load(file)
        :param output_directory: The path to the directory where the new JSON files with JSON schema version 0.1.0.0
            will be created
        :param stop_on_error: Whether the conversion program should stop if an error is encountered when converting
            an entry in the input_data JSON file.
            CAUTION: If stop_on_error is False, will return json objects with empty "regions" entries for every
            annotation object that produces an error.

    Returns:
        :return: True if the output file(s) were produced, False otherwise
    """

    # If input_data is a file path, open the JSON at that target
    if isinstance(input_data, str):
        with open(input_data, 'rt') as f:
            input_data = json.load(f)

    # Creating 3 lists for iterating through the json input data
    images_list = input_data.get('images')
    anno_list = input_data.get('annotations')
    cato_list = input_data.get('categories')
    counter = 0

    # Iterate through the images first:
    for image in images_list:
        # Skeleton of the 0.1.0.0 json format
        coco = {
            'file_data': {},
            'annotation': []
        }

        # Importing the file_data attributes from the COCO json file to 0.1.0.0
        file_data = {'file_ref': image['path']}
        file_path_string = image['file_name']
        # Append unique counter id to ensure duplicate image names on different file paths do not cause file overwrites
        file_path_string = ntpath.splitext(file_path_string)[0] + str(counter) + '.json'
        counter += 1
        # file_path_string = file_path_string.replace(".bmp",".json")
        final_output_file_path = os.path.join(output_directory, file_path_string)
        # image_file_path = images_list[index]['path']
        image_height = image['height']
        image_width = image['width']
        shape_list = [image_height, image_width]
        file_data['shape'] = shape_list
        # coco.get('file_data').update(file_data)

        # run_number simply counts the number of annotations in the json for a single image
        run_num = 1

        # Defining dictionaries for the annotation part of 0.1.0.0
        anno_dict = {}
        regions = []
        defect_dict = {}

            # Iterating through the annotation list:
        for anno in anno_list:
            try:
                region_attributes = {}
                shape_attributes = {}
                if anno['image_id'] == image['id']:

                    # Getting the username
                    if 'creator' in anno:
                        user_name = anno['creator']
                    else:
                        user_name = 'coco_conversion'
                    file_data['user_name'] = user_name

                    anno_dict['run_number'] = run_num
                    # Checksum left out for now
                    anno_dict['annotation_version'] = '0.1.0.0'
                    region_attributes['color'] = anno['color']
                    region_attributes['defectType'] = anno['category_id']

                    segment_list = anno['segmentation']
                    # Defining image array from H x W
                    array = np.zeros((image_height, image_width)).astype(np.uint8)

                    # Drawing Polygons using the segmentation list in COCO
                    # And then converting it to RLE format for 0.1.0.0
                    # Adding functionality for nested polygons using XORing
                    for segment_arr in segment_list:
                        try:
                            arr_new = np.zeros((image_height, image_width)).astype(np.uint8)
                            bin_mask_new = Image.fromarray(arr_new)
                            ImageDraw.Draw(bin_mask_new).polygon(segment_arr, outline=1, fill=1)
                            array_new = np.asarray(bin_mask_new)
                            array = np.bitwise_xor(array, array_new)

                        # Exception handler to prevent errors due to malformed entry in segment_list (usually a
                        # single point entry (list with only two values) that isn't compatible with ImageDraw
                        except Exception as e:

                            # Print annotation detail
                            print('Error converting segmentation object', str(segment_arr), 'in', str(anno))
                            # Print exception details
                            print('Error encountered was:')
                            print(e)

                            if stop_on_error:
                                raise ValueError(
                                    'Error encountered during conversion of segmentation array. '
                                    'Single point segmnetation may have been encountered. ',
                                    'See error message printed above for details.')

                    # Converting binary mask to RLE format
                    rle_list = bin_array_2_rle(array)
                    shape_attributes['all_points'] = rle_list
                    shape_attributes['name'] = "run_length"

                    # Extracting timestamp at which annotation was done
                    if 'events' in anno:
                        if(len(anno['events']) != 0):
                            anno_time = anno['events'][0]['created_at']['$date']
                            anno_time = int(anno_time / 1000.0)
                            normal_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(anno_time))

                        else:
                            normal_time = 'unknown_time'

                    else:
                        normal_time = 'unknown_time'
                    file_data['created_timestamp'] = normal_time

                    file_data['size'] = 0

                    # Iterating the category list
                    for cato in cato_list:
                        if anno['category_id'] == cato['id']:
                            region_attributes['name'] = cato['name']

                    # Putting them in the defect_dictionary
                    defect_dict['region_attributes'] = region_attributes
                    defect_dict['shape_attributes'] = shape_attributes

                    regions.append(defect_dict.copy())

            # Error handling if an entity in anno_list cannot be converted correctly
            except Exception as e:

                # Print annotation detail
                print('Error converting JSON dictionary:', str(anno))
                # Print exception details
                print('Error encountered was:')
                print(e)

                if stop_on_error:
                    raise ValueError('Error encountered during conversion of annotation region. '
                                     'Malformed annotation object may have been encountered. '
                                     'See error message printed above for details.')

        anno_dict['regions'] = regions
        run_num += 1

        # Now finally write the JSON file per bscai version 0.1.0.0 to disk
        coco.get('file_data').update(file_data)

        coco.get('annotation').append(anno_dict.copy())

        # File writing operations
        try:
            annotation_file_writer = open(final_output_file_path, 'w')
        except FileNotFoundError:
            return False
        except PermissionError:
            return False
        json.dump(coco, annotation_file_writer, cls=MyEncoder)

    return True


