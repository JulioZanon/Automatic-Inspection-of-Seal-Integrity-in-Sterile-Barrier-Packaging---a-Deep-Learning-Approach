"""
This module is used to load data from the JSON files in standardized format per the JSON schema found here:
https://bitbucket.org/bostonscientificit/json-schema/
"""

import json
import os
import ntpath
import numpy as np
import glob
from bscai.annotation.raw_annotation_data_loader import rle_2_bin_array
from bscai.image.sample import tile_slice


def get_json_data(path: str, include_extension: bool = True, annotation_indices: list = None) -> dict:
    """
    Function to extract image size, image filename, and regions dictionary from annotation JSON file.

    Args:
        :param path: The file path to the JSON file to be opened
        :param include_extension: Whether the returned filename for the image should include the extension or not (
        set to False if matching a filename but the image format may have been changed)
        :param annotation_indices: List giving the annotation indices to extract the regions objects from.
        If no list is provided, the output will be a list of all regions objects in the JSON file.

    Returns:
        :return A dictionary of attributes:
            image_name: the filename of the image as a string
            image_shape: the image shape as a list
            annotations_list: a list of individual dictionaries of all annotated regions from each annotation in
            annotation_indices

    Example:
    get_json_data("C:/images/json_example.json", include_extension=False, annotation_indices = [0])

    """

    try:
        # Parse json file from filepath
        with open(path) as json_file:
            json_data = json.load(json_file)

        # Get image name from file
        image_name = ntpath.basename(json_data['file_data']['file_ref'])

        if not include_extension:
            image_name = os.path.splitext(image_name)[0]

        # Get image shape
        image_shape = json_data['file_data']['shape']

        # Set annotation_indices equal to the length of all annotations if it is None
        if annotation_indices is None:
            annotation_indices = range(len(json_data['annotation']))

        # Combine all regions from each annotation into a single dictionary
        annotations_list = []
        for i in annotation_indices:
            annotations_list.append(json_data['annotation'][i]['regions'])

    except Exception as e:
        print(e)

        # Print associated json filename and either stop the generator or pass to the next iterate
        raise Exception(f'Error loading JSON: {path}')

    return {'image_name': image_name, 'image_shape': image_shape, 'annotations_list': annotations_list}


def region_empty(region: dict):
    """
    This function checks a single region to determine if there are any positively labeled pixels.
    Compatible with run length encoded and pointwise JSON files using the currently supported JSON schema.

    Args:
        :param region: an individual entry in each entry in 'annotations_list' in the dictionary returned by
        get_json_data. (e.g. test['annotations_list'][0][0] would return a single region)

    Returns:
        :return True if the region is empty, False if it contains at least one positively labeled pixel

    Example:
        test = get_json_data("C:/images/json_example.json", include_extension=False, annotation_indices = [0])
        region_empty(test['annotations_list'][0][0])

    """

    empty = True

    # Find pixelwise annotations (if length is 0, there are no pixels labeled in that region)
    if region['shape_attributes']['name'] == 'pixelByPixel':
        if (len(region['shape_attributes']['all_points_x'])) > 0:
            empty = False

    # Find run-length-encoded annotations (length must be greater than 1 otherwise all pixels are zeroes)
    elif region['shape_attributes']['name'] == 'run_length':
        if (len(region['shape_attributes']['all_points'])) > 1:
            empty = False

    else:
        raise ValueError(f"Unsupported pixel encoding format for label data: {region['shape_attributes']['name']}. "
                         "Supported formats are 'run_length' and 'pixelByPixel'. "
                         "This is stored in the 'name' property of the 'shape_attributes' of the region data.")

    return empty


def remove_empty_regions(regions: list):
    """
    This function takes the 'regions' object from get_json_data and removes all regions that do not have a single
    positively labeled pixel value. Compatible with run length encoded and pointwise JSON files using the currently
    supported JSON schema.

    Args:
        :param regions: an individual entry in 'annotations_list' in the dictionary returned by get_json_data

    Returns:
        :return The input regions dictionary with all entries that do not have a single labeled pixel removed

    Example:
        test = get_json_data("C:/images/json_example.json", include_extension=False, annotation_indices = [0])
        remove_empty_regions(test['annotations_list'][0])

    """

    # Initialize list to hold cut down regions
    regions_out = []

    # Iterate through each region and remove entries where there are no annotated pixels
    for region in regions:

        if not region_empty(region):
            regions_out.append(region)

    return regions_out


def regions_to_array(regions: dict, class_name_list: list, shape: tuple, label_union: bool = False,
                     start_point: tuple = None, tile_shape: tuple = None, flatten: bool = False):
    """
    This function converts rle encoded regions label to a binary mask array with one channel per entry in
    class_name_list where 0 represents unlabeled pixels 1 represents a labeled pixel for that class.

    Args:
        :param regions: the 'regions' entry in the dictionary returned by get_json_data
        :param class_name_list: list of class names to use. Region names not in list will be ignored. These classes
        correspond to the last channel of the array in the order given - input type: list of strings or lists
        To create one class from the combination of two or more region name types simply use a nested list
        (e.g. [['one', 'two'], 'three'] will create two classes with the first one equal to the union of labels for
        class one and two.
        :param shape: the shape of the full binary mask not including the last dimension (which will be used to store
        the results for each class).
        (e.g. providing a shape of (2, 4) with a class_name_list with length 3 will yield an array shape of (2, 4, 3))
        :param label_union: If True, returns a union of all regions labels in class_name_list so that if the pixel is
        labeled as a 1, it indicates at least one region in class_name-list was positively labeled (value = 1)
        :param start_point: starting point to use for selecting a tile from the entire array (zero indexed)
        :param tile_shape: shape of the tile to return from the entire array (starts at start_point)
        :param flatten: If True, flattens the first n - 1 channels to a 1-dimensional array

    Returns:
        :return A binary mask numpy array with one channel per entry in class_name_list where 0 represents unlabeled
        pixels 1 represents a labeled pixel for that class

    Example:
        test = get_json_data("C:/images/json_example.json", include_extension=False, annotation_indices = [0])
        regions_to_array(test['annotations_list'][0], class_name_list=["Surface Finish", "Class2"], shape=(7, 3),
                        start_point = (1, 2), tile_shape = (3,4))

    """

    # Initialize array of shape x number of selected classes
    if label_union:
        seg_labels = np.zeros(shape=shape + (1,), dtype='uint8')
    else:
        seg_labels = np.zeros(shape=shape + (len(class_name_list), ), dtype='uint8')

    # Loop through each region dict in regions list
    for region in regions:
        label_name = region['region_attributes']['name']

        # Find the position(s) of the label_name in the class_name_list
        class_pos = np.where([label_name in x for x in class_name_list])[0]

        # If the region name exists in any position, pull data into segmentation array
        if len(class_pos) > 0:

            # If the region has any positively labeled pixels, update arrays and counts
            if (len(region['shape_attributes']['all_points'])) > 1:

                # Extract the rle data for this region
                loop_rle = region['shape_attributes']['all_points']

                # Add to existing segmentation array
                if label_union:
                    seg_labels[..., 0] = rle_2_bin_array(rle=loop_rle, bin_array=seg_labels[..., 0])
                else:
                    for i in class_pos:
                        seg_labels[..., i] = rle_2_bin_array(rle=loop_rle, bin_array=seg_labels[..., i])

    # Cut array to target tile size if desired
    if start_point is not None and tile_shape is not None:
        seg_labels = seg_labels[tile_slice(start_point + (None, ), tile_shape + (None, ))]

    # Flatten all axes except last one (corresponding to classes) if requested
    if flatten:
        seg_labels = seg_labels.reshape(-1, seg_labels.shape[-1])

    return seg_labels


def dict_from_jsons(json_path: str, images_path: str = None, recursive: bool = True,
                    json_extensions: tuple or list = ('.json', ),
                    image_extensions: tuple or list = ('.png', '.bmp', '.jpeg', '.jpg')):
    """
    Function that extracts a list of valid JSONs (JSONs with corresponding images in images_path if provided) and a
    dictionary of indices for each region corresponding to the JSON list position

    Args:
        :param json_path: The file path to the label files. Uses glob.glob for file search so all glob.glob
            conventions apply (e.g. 'target_path/**/' will search in target_path and all subfolders)
        :param images_path: The file path to the associated images. If provided, only JSON files that have file name
            references which match an image name in images_path will be added to the dictionary. Uses glob.glob for
            file search so all glob.glob conventions apply
            (e.g. 'target_path/**/' will search in target_path and all subfolders)
        :param recursive: Whether to search file paths recursively. Passed to glob.glob
        :param json_extensions: A list of file extensions that will be processed as JSON files if found in json_path
        :param image_extensions: A list of file extensions that will be processed as JSON files if found in images_path

    Returns:
        :return A dictionary of objects:
            JSONList: list of valid JSON filepaths. If images_path is provided, only JSON files with matching images
            are returned in the list
            DefectDict: dictionary of region names where each entry lists all of the indices in JSONList that
            contain the given region

    Example:
    dict_from_jsons("C:/images/jsons", "C:/images/images")

    """

    # Initialize count dictionary with null class
    DefectDict = {'NullClass': []}

    # Get a list of all JSON files in json_path and sort by name
    json_list = []
    for i in json_extensions:
        json_list = json_list + glob.glob(os.path.join(json_path, "*" + i), recursive=recursive)
    json_list.sort()

    # Check to make sure json_list is not empty.
    if len(json_list) == 0:
        raise ValueError(f'No jsons found with extensions {json_extensions} on path {json_path}.')

    # Initialize JSONList to hold returned list of json files
    JSONList = []

    # Get a list of all images in images_path if it was provided
    if images_path is not None:
        images = []
        for i in image_extensions:
            images = images + glob.glob(os.path.join(images_path, "**/*" + i), recursive=recursive)
        images.sort()

        # Check to make sure images is not empty.
        if len(images) == 0:
            raise ValueError(f'No images found with extensions {image_extensions} on path {images_path}.')

        # Remove directory names from image file names for comparison to json contents
        image_filenames = list(map(ntpath.basename, images))

    # Iterate through each JSON file
    for i in json_list:

        # Get image name and regions
        jsondat = get_json_data(i, include_extension=True)
        image_name = jsondat['image_name']
        regions = jsondat['annotations_list'][0]

        # If no images_path was provided or if JSON file contains image name that matches images, proceed with region
        # count
        if images_path is None or (image_name in image_filenames):

            # Initialize counter for any region type to detect null images
            defect_count = 0
            for region in regions:
                label_name = region['region_attributes']['name']
                # Check if the region name exists in the dictionary. If not, add it
                if not (label_name in DefectDict):
                    DefectDict[label_name] = []

                # Determine if any positively labeled pixels exist in this region and, if so add it to the dictionary
                if not region_empty(region):
                    defect_count += 1
                    DefectDict[label_name].append(len(JSONList))

            # If the regions object did not contain a single positively labeled pixel, it is a null class JSON
            if defect_count == 0:
                DefectDict['NullClass'].append(len(JSONList))

            # Add JSON filepath to JSONList
            JSONList.append(i)

    return {'DefectDict': DefectDict, 'JSONList': JSONList}
