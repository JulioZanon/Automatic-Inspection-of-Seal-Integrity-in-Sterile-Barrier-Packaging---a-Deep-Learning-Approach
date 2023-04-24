"""
This module is used to contain various higher level wrapper functions over the os and glob library which are related
to most of the file and directory related queries that can be expected for the AI stuff. The most common types of
queries/functionality included are ways to get common files by their file names, moving common files, copying common
files, writing list of files into a file etc.
"""

from shutil import copy2
from typing import Tuple

from bscai.directory.general_utils import *


def check_common_files_args(*arguments) -> List[Tuple[str, str]]:
    """
    This function is used to check whether the given arguments for functions related to finding common files are in
    the required format or not. If the arguments are in the required format, then returns the zipped list with
    directories and their corresponding file extensions in one tuple as an element of the list
    Args:
        :param arguments: This argument can either be a dictionary type object mapping the path to the directories from
                          where the input images need to be loaded to the extension of the files of the files that
                          should be loaded. The dictionary would look something like this:
                          {directory_1_path: directory_1_file_extension,
                           directory_2_path: directory_2_file_extension,
                           ...}
                           This argument can also be two lists where the first list is the list of paths of the
                           directories from where the input files should be loaded, while the second list should be
                           the list of the file extensions for the corresponding directory paths. The lists would
                           look something like this:
                           [directory_1_path, directory_2_path]
                           and
                           [directory_1_file_extension, directory_2_file_extension]

    Returns:
        :return: A list containing Tuple pairs of the form (directory_path, file_extension)
    """
    if not 0 < len(arguments) < 3:
        print(f'Invalid number of arguments provided: {len(arguments)}. Either 1 dictionary or 2 list type objects '
              f'expected')
    elif len(arguments) == 1 and type(arguments[0]) == dict:
        return list(arguments[0].items())
    elif len(arguments) == 2:
        if type(arguments[0]) == list and type(arguments[-1]) == list:
            if len(arguments[0]) == len(arguments[-1]):
                return list(zip(arguments[0], arguments[-1]))
            else:
                print(f'Length of lists not equal to each other. List 1 Length: {len(arguments[0])}, List 2 Length: '
                      f'{len(arguments[-1])}')
        else:
            print(f'Type of arguments is not list for both of the arguments provided. '
                  f'Type of Argument 1: {type(arguments[0])}, Type of Argument 2: {type(arguments[-1])}')
    else:
        print(f'Invalid arguments provided. Refer to the documentation of the function!')
    return list()


def find_common_files(*args, unionize: bool = False) -> Set[str]:
    """
    This function is responsible to find the list of names of the files that are common based on their file name
    without the extension. The function is a generic function which can take different arguments for the input
    directories and the extension of the files to be loaded from those directories. This function can be used
    extensively to find names of files (both image and annotation) for which annotations have been constructed and
    many other tasks.

    Args:
        :param args: This argument can either be a dictionary type object mapping the path to the directories from
                     where the input images need to be loaded to the extension of the files of the files that should
                     be loaded. The dictionary would look something like this:

                     {directory_1_path: directory_1_file_extension,
                      directory_2_path: directory_2_file_extension,
                      ...}

                     This argument can also be two lists where the first list is the list of paths of the directories
                     from where the input files should be loaded, while the second list should be the list of the
                     file extensions for the corresponding directory paths. The lists would look something like this:
                     [directory_1_path, directory_2_path]
                     and
                     [directory_1_file_extension, directory_2_file_extension]

        :param unionize: This argument specifies which type of aggregation should be done over the file names from
                         the different directories for which the input path has been given. If set to true then all
                         the file names are unionized however if set to false, intersection is calculated over the
                         whole set of files to find the common files. Default value of this parameter is set to
                         false, so that the function finds just the common files.

    Returns:
        :return: A set of files names without the extension which represents the aggregation over the different
                 directories

    Examples:
        >>> find_common_files({'./image_dir': '.bmp', './annotation_dir': '.json'})
        >>> find_common_files(['./image_dir', './annotation_dir'], ['.bmp', '.json'])

    """
    # Get the zipped directory and their respective file extension information
    directory_and_file_extensions = check_common_files_args(*args)
    if directory_and_file_extensions:
        # Get the file names without extension from the first directory
        common_file_nms_set = get_file_nms_set(directory_and_file_extensions[0][0],
                                               directory_and_file_extensions[0][-1])
        # Loop over other directories and aggregate the file names without extension
        for directory_path, file_extension in directory_and_file_extensions[1:]:
            file_nms = get_file_nms_set(directory_path, file_extension)
            # Check for the unionize parameter and apply the applicable aggregation (union or intersection)
            if unionize:
                common_file_nms_set = common_file_nms_set.union(file_nms)
            else:
                common_file_nms_set = common_file_nms_set.intersection(file_nms)
        return common_file_nms_set
    else:
        raise ValueError(f'Invalid arguments provided for finding common files: {args}')


def get_common_image_and_annotation_files(image_directory_path: str, image_files_extension: str,
                                          annotation_directory_path: str) -> Set[str]:
    """
    This function is responsible to return a set containing the names of the files (without extension) which are
    common in both the image and annotation directory. This function can be used to get a set of the common image and
    annotation files

    Args:
        :param image_directory_path: The path to the directory that contains the image files
        :param image_files_extension: The extension for the image files. The image files which have this particular
                                      extension are the only ones that will be loaded
        :param annotation_directory_path: The path to the directory which contain the annotation (.json) files

    Returns:
        :return: A set containing the names of the files which are common to both the image and annotation file
                 directories. The file names returned will not contain the extension.
    """
    return find_common_files({image_directory_path: image_files_extension,
                              annotation_directory_path: '.json'},
                             unionize=False)


def move_common_files(*args, output_directory_paths: List[str], mover_function=copy2):
    """
    This function is responsible to move the common files from their respective parent directories to the respective
    given  output directories. The function by defaults copy the files from the input directory to the output
    directory, but can be used to cut (move) the files as well

    Args:
        :param args: This argument can either be a dictionary type object mapping the path to the directories from
                     where the input images need to be loaded to the extension of the files of the files that should
                     be loaded. The dictionary would look something like this:

                     {directory_1_path: directory_1_file_extension,
                      directory_2_path: directory_2_file_extension,
                      ...}

                     This argument can also be two lists where the first list is the list of paths of the directories
                     from where the input files should be loaded, while the second list should be the list of the
                     file extensions for the corresponding directory paths. The lists would look something like this:
                     [directory_1_path, directory_2_path]
                     and
                     [directory_1_file_extension, directory_2_file_extension]

        :param output_directory_paths: A list of the same length for the number of input directory paths,
                                       with the same relative ordering. The list contains the paths to the output
                                       directories for common files loaded
                                       from the corresponding input directories
        :param mover_function: The function which defines what to do with the two files. Default values if to copy
                               from input directory to output directory. Can use the move function to cut files.

    Returns:
        :return: None
    """
    # Get the directory and their respective file extension information
    directory_and_file_extensions = check_common_files_args(*args)
    # Get the names of the common files
    common_file_nms = find_common_files(*args, unionize=False)
    # Iterate through each file whose name is common among all the directories
    for file_nm in common_file_nms:
        for (file_source_directory_path, file_extension), output_directory_path in \
                zip(directory_and_file_extensions, output_directory_paths):
            if check_output_directory(output_directory_path):
                file_path = os.path.join(file_source_directory_path, f'{file_nm}{file_extension}')
                mover_function(file_path, output_directory_path)
            else:
                raise ValueError(f'Invalid path to the output directory: {output_directory_path}')


def put_common_files_in_directory(*args, output_directory_path: str, mover_function=copy2):
    """
    This function moves common files from the given input directory paths to the same output directory path.

    Args:
        :param args: This argument can either be a dictionary type object mapping the path to the directories from
                     where the input images need to be loaded to the extension of the files of the files that should
                     be loaded. The dictionary would look something like this:

                     {directory_1_path: directory_1_file_extension,
                      directory_2_path: directory_2_file_extension,
                      ...}

                     This argument can also be two lists where the first list is the list of paths of the directories
                     from where the input files should be loaded, while the second list should be the list of the
                     file extensions for the corresponding directory paths. The lists would look something like this:
                     [directory_1_path, directory_2_path]
                     and
                     [directory_1_file_extension, directory_2_file_extension]

        :param output_directory_path: A string representing the common output directory path where all the input
                                      files should be dumped
        :param mover_function: The function which defines what to do with the two files. Default values if to copy
                               from input directory to output directory. Can use the move function to cut files.

    Returns:
        :return: None
    """
    # Get the directory and their respective file extension information
    directory_and_file_extensions = check_common_files_args(*args)
    # Call the mover function with the same output directory for all the input files
    move_common_files(*args, output_directory_paths=[output_directory_path] * len(directory_and_file_extensions),
                      mover_function=mover_function)


def move_common_image_and_annotation_files(image_source_directory_path: str, image_files_extension: str,
                                           annotation_source_directory_path: str, image_destination_directory_path: str,
                                           annotation_destination_directory_path: str, mover_function: copy2):
    """
    This function is responsible to just move the common image and annotation files from their input directories to
    the given output directories.

    Args:
        :param image_source_directory_path: The path to the input directory which contains the images.
        :param image_files_extension: The extension for the images. Only image files with that extension will be moved
        :param annotation_source_directory_path: The path to the input directory containing the annotation (.json) files
        :param image_destination_directory_path: The path to the output directory where the images will be dumped
        :param annotation_destination_directory_path: The path to the output directory where the annotation files
                                                      will be dumped
        :param mover_function: The function which defines what to do with the two files. Default values if to copy
                               from input directory to output directory. Can use the move function to cut files.

    Returns:
        :return: None
    """
    move_common_files({image_source_directory_path: image_files_extension,
                       annotation_source_directory_path: '.json'},
                      output_directory_paths=[image_destination_directory_path,
                                              annotation_destination_directory_path],
                      mover_function=mover_function)
