"""
This module is used to load data from the JSON files in legacy pointwise format. Several JSON loading functions in
this library will eventually be deprecated.
The module assumes that the file for the Annotations contains the same format that was used until mid 2018. The
sample JSON files look something like this:

JSON Format for Annotation:
    {
        "C:\\Images\\Image_Name": {
            "annotation_version": "0.0.2.12",
            "base64_img_data": "",
            "fileName": "C:\\Images\\Finished\\JSON_Name",
            "fileRef": "C:\\Images\\Image_Name",
            "file_attributes": "{}",
            "regions": {
                "1": {
                    "region_attributes": {
                        "color": "#7F8F3740",
                        "defectType": 1,
                        "name": "Defect_1_Name"
                    },
                    "shape_attributes": {
                        "all_points_x": [],
                        "all_points_y": [],
                        "name": "pixelByPixel"
                    }
                },
                "2": {
                    "region_attributes": {
                        "color": "#7F0AEB8D",
                        "defectType": 2,
                        "name": "Defect_2_Name"
                    },
                    "shape_attributes": {
                        "all_points_x": [],
                        "all_points_y": [],
                        "name": "pixelByPixel"
                    }
                },
                .
                .
                .
                .
                .
                .
                "k": {
                    "region_attributes": {
                        "color": "#7FE6295B",
                        "defectType": 17,
                        "name": "Defect_k_Name"
                    },
                    "shape_attributes": {
                        "all_points_x": [],
                        "all_points_y": [],
                        "name": "pixelByPixel"
                    }
                }
            },
            "size": 0,
            "userName": "BSCI-SWAUTO-SVC"
        }
    }

This module will have many sample small helper functions that can be used to load data from the json file and also
format that data appropriately.
"""
import ast
import json
from typing import Any
import numpy as np

from bscai.directory.general_utils import *

def get_all_json_data_from_annotation_file(annotation_file_path: str) -> dict:
    """
    Returns a dictionary containing all the data from an annotation file (JSON file).

    Args:
        :param annotation_file_path: The path to a Annotation JSON file.

    Returns:
        :return: A dictionary containing all the data that is present inside the JSON file representing the
                 annotation for a particular image file.
    """
    if exist_file(annotation_file_path):
        try:
            with open(annotation_file_path, 'r') as json_reader:
                return json.load(json_reader)
        except PermissionError as _:
            raise ValueError(f"Permission error faced when loading data from file: {annotation_file_path}")
        except Exception as _:
            raise ValueError(f"Unknown error faced when loading data from file: {annotation_file_path}")
    else:
        raise ValueError(f"Invalid File Path: {annotation_file_path}")


def bypass_image_name_from_annotation_data(annotation_data: dict) -> dict:
    """
    This function takes all the data which was loaded from an Annotation file and returns the only meaningful data
    by bypassing the image name present as the key in the annotation data. The returned data looks something like:

        {
            "annotation_version": "0.0.2.12",
            "base64_img_data": "",
            "fileName": "C:\\Images\\Finished\\JSON_Name",
            "fileRef": "C:\\Images\\Image_Name",
            "file_attributes": "{}",
            "regions": {
                "1": {
                    "region_attributes": {
                        "color": "#7F8F3740",
                        "defectType": 1,
                        "name": "Defect_1_Name"
                    },
                    "shape_attributes": {
                        "all_points_x": [],
                        "all_points_y": [],
                        "name": "pixelByPixel"
                    }
                },
                ...
                "k": {
                    "region_attributes": {
                        "color": "#7FE6295B",
                        "defectType": 17,
                        "name": "Defect_k_Name"
                    },
                    "shape_attributes": {
                        "all_points_x": [],
                        "all_points_y": [],
                        "name": "pixelByPixel"
                    }
                }
            },
            "size": 0,
            "userName": "BSCI-SWAUTO-SVC"
        }

    Args:
        :param annotation_data: The dictionary representing all the data present in the annotation file that was
                                loaded using the get_all_json_data_from_annotation_file.

    Returns:
        :return: A dictionary that contains the details about annotation data after bypassing the image name key
    """
    # First bypass the first key that is the Image Path used by the D3AI tool
    return list(annotation_data.items())[-1][-1]


def validate_annotation_data(annotation_data: dict) -> bool:
    """
    This function takes all the data which was loaded from an Annotation file and validates that the file follows the
    expected formatting semantics.

    Args:
        :param annotation_data: The dictionary representing all the data present in the annotation file that was
                                loaded using the get_all_json_data_from_annotation_file.

    Returns:
        :return: True if the annotation data is in the expected format False, otherwise
    """
    all_annotation_information = bypass_image_name_from_annotation_data(annotation_data)
    # Check for the metadata keys
    metadata_check = {"annotation_version", "base64_img_data", "fileName", "fileRef", "file_attributes", "size",
                      "userName", "regions"}.issubset(all_annotation_information.keys())
    if metadata_check:
        region_check = True
        # Get the dictionary that contains the annotation pixel for different defect region.
        all_defect_regions_dictionary = get_regions_dictionary_from_annotation_file_data(annotation_data)
        # Iterate through each defect region
        for _, current_defect_region_dict in all_defect_regions_dictionary.items():
            region_check = region_check and {"region_attributes", "shape_attributes"}.issubset(
                current_defect_region_dict.keys())
            # Because of short circuit evaluation key error will not be thrown if keys are not present in the defect
            # region dictionary
            region_check = region_check and {"color", "defectType", "name"}.issubset(
                current_defect_region_dict['region_attributes'].keys())
            region_check = region_check and ({"all_points_x", "all_points_y", "name"}.issubset(
                current_defect_region_dict['shape_attributes'].keys()) or {"all_points", "name"}.issubset(
                current_defect_region_dict['shape_attributes'].keys()))
        return region_check
    else:
        return False


def get_annotation_file_metadata(annotation_data: dict) -> dict:
    """
    This function takes all the data which was loaded from an Annotation file and returns the only meaningful data
    related to the metadata that the annotation file stores. The metadata returned includes information about the
    image file reference, and the annotation tool. The format of the returned dictionary is as follows:

        {
            "annotation_version": "0.0.2.12",
            "base64_img_data": "",
            "fileName": "C:\\Images\\Finished\\JSON_Name",
            "fileRef": "C:\\Images\\Image_Name",
            "file_attributes": "{}",
            "img_height" : 0,
            "img_width" : 0,
            "size": 0,
            "userName": "BSCI-SWAUTO-SVC"
        }

    Args:
        :param annotation_data: The dictionary representing all the data present in the annotation file that was
                                loaded using the get_all_json_data_from_annotation_file.

    Returns:
        :return: A dictionary that contains the details about the metadata contained within the annotation file
    """
    # First bypass the first key that is the Image Path used by the D3AI tool
    all_annotation_information = bypass_image_name_from_annotation_data(annotation_data)
    return {key: value for key, value in all_annotation_information.items() if key != 'regions'}


def get_regions_dictionary_from_annotation_file_data(annotation_data: dict) -> dict:
    """
    This function takes all the data which was loaded from an Annotation file and returns the only meaningful data
    that is the data containing information about all the regions. For reference the data returned from the original
    dictionary looks something like this:

        {
            "1": {"region_attributes": {
                    "color": "#7F8F3740",
                    "defectType": 1,
                    "name": "Defect_1_Name"},
                "shape_attributes": {
                    "all_points_x": [],
                    "all_points_y": [],
                    "name": "pixelByPixel"}
            }, .....
        }

    Args:
        :param annotation_data: The dictionary representing all the data present in the annotation file that was
                                loaded using the get_all_json_data_from_annotation_file.

    Returns:
        :return: A dictionary that contains the mappings from region number to region information
    """
    # First bypass the first key that is the Image Path used by the D3AI tool
    all_annotation_information = bypass_image_name_from_annotation_data(annotation_data)
    try:
        # Return only the information needed
        return all_annotation_information['regions']
    except KeyError as _:
        raise ValueError(f"Invalid key 'region' when trying to load the information about "
                         f"regions from: {all_annotation_information}")


def str_to_int_lst(input_str: str) -> List[int]:
    """
    Converts a list representation in string format to actual list object corresponding to that list representation.

    Args:
        :param input_str: str type object which represents the string version (representation) of a list.

    Returns:
        :return: list type object representing the input string.

    Examples:
        >>> print(str_to_int_lst('[1, 2, 3, 4]'))
        [1, 2, 3, 4]

        >>> ls = str_to_int_lst('[1, 2, 3, 4]')
        >>> ls.append(5)
        >>> print(ls)
        [1, 2, 3, 4, 5]
    """
    return ast.literal_eval(input_str)


def format_coordinates_list(coordinates_data_from_annotation_data: Any) -> List[int]:
    """
    Returns the list of coordinates after processing them in the correct format. If the input data from the file is
    actual string then converts the data into the list of integers. If the data is of the correct format,
    then returns the data as it is. This function is actually useful to escape the handling of exception when loading
    the data from the file and make the final function a little less cluttered.

    Args:
        :param coordinates_data_from_annotation_data: The data about the x and y coordinates that is loaded from the
                                                      annotation file.

    Returns:
        :return: A list of integer coordinates of the pixels corresponding to coordinates. Helps in making the
                 script standardized.

    Examples:
        >>> print(format_coordinates_list('[1, 2, 3, 4]'))
        [1, 2, 3, 4]
        >>> ls = format_coordinates_list('[1, 2, 3, 4]')
        >>> ls.append(5)
        >>> print(ls)
        [1, 2, 3, 4, 5]

        >>> ls = format_coordinates_list([1, 2, 3, 4])
        >>> ls.append(5)
        >>> print(ls)
        [1, 2, 3, 4, 5]
    """
    try:
        # Convert the data assuming it is type string.
        return str_to_int_lst(coordinates_data_from_annotation_data)
    except ValueError as _:
        # Data already in the correct format
        return coordinates_data_from_annotation_data


def rle_2_bin_array(rle: list, shape: tuple = None, bin_array: np.ndarray = None, fill_dir='C') -> np.ndarray:
    """
    This function returns the binary mask as a numpy.ndarray from a run length encoded list.

    Args:
        :param rle: The list containing the run length encoded information.
                    The list always starts with the number of zeros.
        :param shape: The shape of the decompressed binary mask. Must be provided if bin_array is not provided.
        :param bin_array: An existing binary label array to add positive labels to. Result will be the same as the
            union of the labeled pixels in bin_array and the rle list. If bin_array and shape arguments are provided,
            shape will be ignored.
        :param fill_dir: The direction that the target array should be filled in.
                         Passed to np.reshape as the 'order' parameter.
                         'C' indicates filling along column first
                         'R' indicates filling along row first

    Returns:
        :return: The shape x 1 binary array representing the all the annotated coordinates corresponding to the
                 rle encoded data.

    Examples:
        print(rle_2_bin_array(rle = [0,1,2,1,4], shape = (2,4)))
        [[1 0 0 1]
         [0 0 0 0]]
    """

    if bin_array is None:
        # Initialize existing array as all zeros if it was not provided
        bin_array = np.zeros(shape=np.prod(shape), dtype=np.int)
    else:
        # Flatten existing array to 1-d using fill_dir
        shape = np.shape(bin_array)
        bin_array = bin_array.flatten(order=fill_dir)

    # Define skipping variable to skip every other element in rle (since every odd element corresponds to zeroes)
    do_this = False
    cum_sum = 0

    for i in rle:

        # If this is an odd step, set range items to 1
        if do_this:
            bin_array[cum_sum:(cum_sum + i)] = 1

        # Add to cumulative sum
        cum_sum += i
        # Reverse logical
        do_this = not do_this

    bin_array = np.reshape(bin_array, newshape=shape, order=fill_dir)
    return bin_array


def rle_2_xy_array(rle: list, shape: tuple, fill_dir='C') -> np.ndarray:
    """
    This function returns the xy array as a numpy.ndarray from a run length encoded list.

    Args:
        :param rle: The list containing the run length encoded information.
                    The list always starts with the number of zeros.
        :param shape: The shape of the binary array represented by the rle list.
        :param fill_dir: The direction that the target array should be filled in.
                         Passed to np.reshape as the 'order' parameter.
                         'C' indicates filling along column first
                         'R' indicates filling along row first

    Returns:
        :return: A 2-Dimensional Array that contains the x, y coordinates merged together

    Examples:
        >>> print(rle_2_xy_array(rle = [0,1,2,1,4], shape = (2,4)))
        [[0 0]
         [3 0]]
    """

    #Get binary mask
    bin_mask = rle_2_bin_array(rle=rle, shape=shape, fill_dir=fill_dir)

    #Convert to x-y coordinates
    xy_coords = np.where(bin_mask == 1)

    #Convert to final array format
    return np.array(list(zip(xy_coords[1], xy_coords[0])))


def bin_array_2_rle(inp_arr: np.ndarray, rle_dir='C') -> list:
    """
    This function compresses a binary mask numpy.ndarray to a run length encoded list

    Args:
        :param inp_arr: numpy array containing binary mask data with format of:
                        0 for item not present, 1 for item present at that pixel
        :param rle_dir: The direction that the target array should be filled in.
                        Passed to np.reshape as the 'order' parameter when flattening array to 1-d before RLE
                        processing.
                        'C' indicates filling along column first
                        'R' indicates filling along row first
    Returns:
        :return: A list containing the binary mask as run length encoded data. Always starts with number of zeros.

    Examples:
        >>> print(bin_array_2_rle(inp_arr = np.array([[1,0,0,1], [1,0,1,0]])))
        [0, 1, 2, 2, 1, 1, 1]
    """

    # Flatten array per rle_dir argument
    flat_arr = np.reshape(inp_arr, newshape=np.size(inp_arr), order=rle_dir)

    # Perform RLE conversion
    rle_out = np.diff(flat_arr)
    rle_out = np.where(rle_out != 0)[0]
    rle_out = np.insert(rle_out, 0, -1)
    rle_out = np.append(rle_out, len(flat_arr) - 1)
    rle_out = list(np.diff(rle_out))

    # If flat_arr starts with a 1, prepend 0 to the RLE list since the RLE data always starts with the number of zeros
    if flat_arr[0] == 1:
        rle_out.insert(0, 0)

    return rle_out

#TODO: Rename and refactor this function to more reflect fact that x-y coordinates are returned not binary array
def get_bin_array_of_defect_region(defect_region_dict: dict, img_shape: tuple) -> np.ndarray:
    """
    This function returns the 2-Dimensional numpy.ndarray containing the data about particular defect
    label/type/region from the data loaded from the annotation file. This function only works with point-wise encoded
    data.

    Args:
        :param defect_region_dict: The dictionary containing the data about the current defect region that is being
                                   investigated. All the details should correspond to the same defect
                                   type/label/class/region.
        :param img_shape: Tuple defining shape of binary mask

    Returns:
        :return: The img_shape - Dimensional binary mask array representing the all the annotated pixels
        corresponding to the current region marked by the given input dictionary containing data about that
        particular defect region.
    """
    # Loading the shape data for the current defect region
    if 'shape_attributes' in defect_region_dict:
        region_shape_data = defect_region_dict['shape_attributes']
    else:
        raise ValueError(f"Invalid key when loading shape information: 'shape_attributes'")

    if 'all_points_x' in region_shape_data and 'all_points_y' in region_shape_data:
        # Load the x and y coordinates list from point-wise encoded (uncompressed) data
        defect_region_x_coordinates_list = format_coordinates_list(region_shape_data['all_points_x'])
        defect_region_y_coordinates_list = format_coordinates_list(region_shape_data['all_points_y'])
        
        #Translate the x-y coordinate list to a binary mask
        out_arr = np.zeros(shape = img_shape)
        out_arr[defect_region_y_coordinates_list, defect_region_x_coordinates_list,0] = 1
        return out_arr

    elif 'all_points' in region_shape_data: # load RLE compressed data
        return rle_2_xy_array(rle=region_shape_data['all_points'], shape = img_shape)
    else:
        raise ValueError("Invalid key when finding coordinates list: ['all_points_x', 'all_points_y'] or 'all_points'")


def merge_x_y_coordinates_lists_to_array(x_coordinates_list: List[int], y_coordinates_list: List[int]) -> np.ndarray:
    """
    Merges two individual x and y coordinates lists into one array such that the corresponding coordinates from each
    list are grouped together to form an array representing coordinates in 2 Dimensions.

    Args:
        :param x_coordinates_list: a list of x coordinates loaded from the coordinates section corresponding to a
        particular defect from the annotation file.
        :param y_coordinates_list: a list of y coordinates loaded from the coordinates section corresponding to a
        particular defect from the annotation file.

    Returns:
        :return: A 2-Dimensional Array that contains the x, y coordinates merged together

    Examples:
        >>> print(merge_x_y_coordinates_lists_to_array([1, 2, 3], [30, 20, 10]))
        [[ 1 30]
         [ 2 20]
         [ 3 10]]
    """
    return np.array(list(zip(x_coordinates_list, y_coordinates_list)))


def get_x_y_coordinate_array_of_defect_region(defect_region_dict: dict) -> np.ndarray:
    """
    This function returns the 2-Dimensional numpy.ndarray containing the data about particular defect
    label/type/region from the data loaded from the annotation file. This function only works with point-wise encoded
    data.

    Args:
        :param defect_region_dict: The dictionary containing the data about the current defect region that is being
                                   investigated. All the details should correspond to the same defect
                                   type/label/class/region.

    Returns:
        :return: The 2-Dimensional array representing the all the annotated coordinates corresponding to the current
                 region marked by the given input dictionary containing data about that particular defect region.
    """
    # Loading the shape data for the current defect region
    if 'shape_attributes' in defect_region_dict:
        region_shape_data = defect_region_dict['shape_attributes']
    else:
        raise ValueError(f"Invalid key when loading shape information: 'shape_attributes'")

    if 'all_points_x' in region_shape_data and 'all_points_y' in region_shape_data:
        # Load the x and y coordinates list from point-wise encoded (uncompressed) data
        defect_region_x_coordinates_list = format_coordinates_list(region_shape_data['all_points_x'])
        defect_region_y_coordinates_list = format_coordinates_list(region_shape_data['all_points_y'])
        return merge_x_y_coordinates_lists_to_array(defect_region_x_coordinates_list, defect_region_y_coordinates_list)
    #NOTE: Must implement xy array return using image shape from JSON file for RLE encoded data but it is not passed to
    # this function. To preserve compatibility, recommend leaving this function as is and writing new function to get
    # binary masks as they are more directly useful anyways
    #elif 'all_points' in region_shape_data: # load RLE compressed data
    #    return rle_2_xy_array(rle=region_shape_data['all_points'], shape = ())
    else:
        raise ValueError("Invalid key when finding coordinates list: ['all_points_x', 'all_points_y'] or 'all_points'")


def get_region_name(defect_region_dict: dict) -> str:
    """
    Takes in a dictionary representing the information of a particular defect region and return the name of the
    current defect region. The dictionary looks something like:

        {"region_attributes": {
                "color": "#7F8F3740",
                "defectType": 1,
                "name": "Defect_Name"
            },
            "shape_attributes": {
                "all_points_x": [],
                "all_points_y": [],
                "name": "pixelByPixel"
            }
        }
    Args:
        :param defect_region_dict: The dictionary containing the data about the current defect region that is being
                                   investigated. All the details should correspond to the same defect
                                   type/label/class/region.

    Returns:
        :return: The name of the defect region as loaded from the information represented by the defect region
                 dictionary.
    """
    if 'region_attributes' in defect_region_dict and 'name' in defect_region_dict['region_attributes']:
        return defect_region_dict['region_attributes']['name']
    else:
        raise ValueError(f'Invalid keys: "region_attributes", "name"')


def get_region_configuration_dictionary(configuration_dictionary: dict, defect_region_name: str) -> dict:
    """
    Return the configuration for the particular defect region from the whole configuration dictionary of the
    current part that is being investigated.

    Args:
        :param configuration_dictionary: The dictionary containing the various configurations for the different
                                         defect regions that could be present in the part.
        :param defect_region_name: The name of the defect region for which the tiles are being made.

    Returns:
        :return: A dictionary containing information about the configurations for the particular defect region if
                 found in the configuration dictionary. If not then return an empty dictionary.
    """
    if defect_region_name in configuration_dictionary:
        return configuration_dictionary[defect_region_name]
    else:
        return {}


def get_annotated_regions_name(annotation_data: dict) -> set:
    """
    This function returns a set of the names of the defect regions for which there exist some annotations.

    Args:
        :param annotation_data: The data loaded from the annotation file (.json file)

    Returns:
        :return: A set of the names of the defects for which annotations exist.
    """
    annotated_regions_name = []
    regions_dictionary = get_regions_dictionary_from_annotation_file_data(annotation_data)
    # Iterate through each region
    for _, defect_region_dictionary in regions_dictionary.items():
        defect_region_name = get_region_name(defect_region_dictionary)
        annotated_area_array = get_x_y_coordinate_array_of_defect_region(defect_region_dictionary)
        if isinstance(annotated_area_array, np.ndarray) and annotated_area_array.shape != (0,):
            annotated_regions_name.append(defect_region_name)
    return set(annotated_regions_name)


def get_good_image_annotation_data() -> dict:
    """
    This is a function that is just used to get the fake annotation data for the good images.
    :return:
    """
    return {
        "image_file_name": {
            "annotation_version": "0.0.2.12",
            "base64_img_data": "",
            "fileName": "annotation_file_name",
            "fileRef": "annotation_file_path",
            "file_attributes": {},
            "regions": {},
            "size": 0,
            "userName": "BSCI-SWAUTO-SVC"
        }
    }


def annotation_for_good_image(annotation_data: dict) -> bool:
    # Get the regions dictionary
    regions_dictionary = get_regions_dictionary_from_annotation_file_data(annotation_data)
    # Iterate through each region
    for _, defect_region_dictionary in regions_dictionary.items():
        defect_region_name = get_region_name(defect_region_dictionary)
        annotated_area_array = get_x_y_coordinate_array_of_defect_region(defect_region_dictionary)
        if isinstance(annotated_area_array, np.ndarray) and annotated_area_array.shape != (0,):
            # If at least one annotation exist than not a good image
            return False
    # No annotation exist
    return True
