"""
This module is responsible to hold various functions that can be used to  do general file and directory related
functions. This module hopes to provide an wrapper over the os and glob library. The module currently only has
simple functions that tend to things like constructing directories and checking for existence of paths. The other
functioning include getting list of files and directories in a given directory.

Note: Is supposed to work on all the following OS: Windows, MacOS and Linux
"""

import glob
import os
import re
from typing import List, Set
from PIL import Image
import math
import errno
import hashlib
import ntpath
from datetime import datetime

def get_name_filepath_dict(file_directory):
    """
    :param file_directory: a directory containing files

    :return: dict of format {filename : full filepath}

    It is recursive: searches all files in the input directory
    """

    image_file_list = get_dir_contents(file_directory, recursive=True, return_relative_filepaths=False)
    image_file_list = [x for x in image_file_list if os.path.isfile(x)]
    image_file_names = [ntpath.basename(x) for x in image_file_list if os.path.isfile(x)]

    # Zip the file name lists together into a dictionary that uses the base name of the file
    # as the key to retrieve the full file path
    image_zip = zip(image_file_names, image_file_list)
    # Create a dictionary from zip object
    image_dict = dict(image_zip)
    return image_dict

def get_shape_dict(image_directory):
    """
    :param image_directory: a directory containing images

    :return: dict of of {extensionless-filename : shape}

    Is not recursive.
    """
    shape_dict = {}
    # only grab the image files
    img_files = get_dir_contents(os.path.join(image_directory),recursive=True)
    pattern = re.compile(r'\.bmp|\.jpg|\.png|\.jpeg|\.tiff', re.UNICODE)
    img_files = [a for a in img_files if pattern.search(a)]

    if len(img_files) < 0:
        print("Could not find image files. Check directory name and try again.")
        return False

    for file_name in img_files:
        with Image.open(os.path.join(image_directory, file_name)) as img:
            # height comes first then width for json format, but width x height for PIL
            shape = (img.size[1], img.size[0])

            # remove the extension
            name = ntpath.basename(os.path.splitext(file_name)[0])
            shape_dict[name] = shape

    return shape_dict

def exist_directory(directory_path: str) -> bool:
    """
    This function checks whether a particular directory exist or not in the file system.

    Args:
        :param directory_path: Represents a path to a directory. If a relative path is given it is resolved
                               with respect to the current working directory of the process.

    Returns:
        :return: (bool) True if the path points to an existing directory else False.
    """
    return os.path.isdir(directory_path)


def exist_file(file_path: str) -> bool:
    """
    This function checks whether a particular file exist of not in the file system.

    Args:
        :param file_path: Represents a path to a file. If a relative path is given it is resolved
                          with respect to the current working directory of the process.

    Returns:
        :return: (bool) True if the path points to an existing file else False.
    """
    return os.path.isfile(file_path)


def check_output_directory(output_directory_path: str) -> bool:
    """
    This function is responsible to check if the given path to a directory (output) exist. If it does not exist the
    function tries to construct a directory at the corresponding path. If the directory does not exist and the path
    cannot be resolved, then return false, else if the directory exist return true, else construct the directory and
    return true. Useful when trying to save stuff in output directory. Do not have to check for existence and then
    take the appropriate action. Using this function can take care of all cases.

    Args:
        :param output_directory_path: Represents a path to the output directory where the
                                      output images will be stored
    Returns:
        :return: (bool) True if the output directory exists, else tries to make one and returns
                 True if there is no error, otherwise returns False.

    Note: Think about performance based on catching exceptions.
    """
    try:
        if not exist_directory(output_directory_path):  # Make a new directory if does not exist already.
            os.makedirs(output_directory_path)
        return True
    except FileNotFoundError:  # If the path points to a directory that cannot be constructed.
        return False
    except PermissionError:
        return False


def get_dir_contents(directory_path: str, pattern: str = '*', recursive: bool = False,
                     return_relative_filepaths: bool = True) -> List[str]:
    """
    This function is responsible to return the contents of a particular directory based on the given pattern which
    will be matched using regular expression for the contents in the directory.

    Args:
        :param directory_path: Represents a path to a directory.
        :param pattern: Represents the end pattern to be used for the glob matching.
        :param recursive: Whether to search recursively through subfolders of directory_path
        :param return_relative_filepaths: Whether the resulting file list should contain filepaths relative to
        directory_path or absolute filepaths

    Returns:
        :return: (list) A list of the names of the contents present in the directory that match the
                 glob pattern matching
    """

    # Convert directory name to proper path
    dir_pathname = os.path.normpath(directory_path)

    if os.path.isdir(directory_path):

        try:

            # Get entire contents of directory
            dir_content_names = glob.glob(os.path.join(dir_pathname, '*' + pattern), recursive=recursive)
            if return_relative_filepaths:

                # Loop through each entry to convert it to a path relative to dir_pathname
                for i in range(len(dir_content_names)):
                    dir_content_names[i] = os.path.relpath(dir_content_names[i], dir_pathname)

        except PermissionError:
            return list()
        return dir_content_names
    else:
        return list()


def get_file_names_list(directory_path: str, file_extension: str = '*.*') -> List[str]:
    """
    This function returns the file names of all files present in a directory that match a specified extension. The
    file names are returned without the extension as a list.
    (e.g. 'test.png' becomes 'test' when the file_extension argument = '.png')

    Args:
        :param directory_path: (str) Represents a path to a directory.
        :param file_extension: (str) The extension of the files for which the file names are needed.
                         The extension should have a '.' (dot) sign in front of them.

    Returns:
        :return: (list) A list of the file names present in the directory which have the given file extension.
    """
    if file_extension:
        if file_extension[0] == '.':
            return get_dir_contents(directory_path, f'*{file_extension}')
        else:
            return get_dir_contents(directory_path, f'*.{file_extension}')
    else:
        return os.listdir(directory_path)


def get_file_nms_set(directory_path: str, file_extension: str = '*.*') -> Set[str]:
    """
    This function returns the file names of all files present in a directory that match a specified extension. The
    file names are returned without the extension as a set.
    (e.g. 'test.png' becomes 'test' when the file_extension argument = '.png')

    Args:
        :param directory_path: (str) Represents a path to a directory.
        :param file_extension: (str) The extension of the files for which the file names are needed.
                         The extension should have a '.' (dot) sign in front of them.

    Returns:
        :return: A list of the file names present in the directory which have the given file extension. The
                 returns file names without extension in a set form
    """

    def extension_replacer(file_name):
        return file_name.replace(file_extension, '')

    return set(map(extension_replacer, get_file_names_list(directory_path, file_extension)))


def get_subdirectory_names(directory_path: str) -> List[str]:
    """
    This function returns the names of the subdirectories present in the given directory. It is important to note that
    the names of the subdirectories have the forward or back slash appended to them based on the operating system being
    used.

    Args:
        :param directory_path: (str) Represents a path to a directory.

    Returns:
        :return: (list) A list of the names of the subdirectories present in the given directory.
    """
    return get_dir_contents(directory_path, '//')


def get_subdirectory_nms(directory_path: str) -> List[str]:
    """
    This function returns the names of the subdirectories present in the given directory.
    It is important to note that the name of the subdirectories returned is without the
    accompanying slashes.

    Args:
        :param directory_path: (str) Represents a path to a directory.

    Returns:
        :return: (list) A list of the names of the subdirectories present in the given directory.
    """
    return list(map(lambda subdir_name: re.sub(r'[//\\]', '', subdir_name),
                    get_dir_contents(directory_path, '//')))


def construct_path(directory_path: str, directory_content_name: str) -> str:
    """
    This function is used to construct paths by appropriately appending the directory content with the directory path
    based on the current operating system.

    Args:
        :param directory_path: The path to the parent directory
        :param directory_content_name: The name of the file or subdirectory contained in the parent directory for which
                                       you want to construct the path

    Returns:
        :return: A path signifying the path to the file or subdirectory present within the parent directory
    """
    return os.path.join(directory_path, directory_content_name)


def split_images_in_dir(input_dir: str, output_dir: str, dim: tuple = None, columns: int = 1, rows: int = 1,
                        append_dirname = False, append_datetime=False,append_string=None,
                        **kwargs) -> None:
    """
    Data pre-processing function for large images. Splits images that are
    too large to be handled by D3AI or other applications and writes them to
    the disk, prepended with a number. "test.png" called with columns=2, rows=2
    will output "0_test.png" "1_test.png" "2_test.png" "3_test.png"

    Args:
        :param input_dir: str directory images are pulled from to be split
        All images from this directory are pulled. Make sure the dir
        contains only images you're interested in splitting.

        :param output_dir: str directory split images are output to

        :param dim: dimension of each output image (height, width). there may small 'end slices'
        but most slices wil be of this dim

        :param columns: int number of columns to split the image into.
          only used if dim is not specified. Must also specify rows.

        :param rows: int number of rows to split image into.
          only used if dim is not specified. Must also specify columns.

        :param append_dirname: Optional. appends the name of the folder the input file resides in, to the output file

        :param append_datetime: Optional. Appends datetime of the file to the end

        :param append_string: appends this string, if given, to the start of each filename

        :param kwargs: Additional arguments passed to get_dir_contents

    Returns:
        :return: None

    Examples:
        # create a 3x3 grid out of the images
        split_images_in_dir(input_dir,output_dir,columns=3,rows=3)

        # split the images into a (200,200) squares
        split_images_in_dir(input_dir,output_dir,dim=(200,200))
    """

    Image.MAX_IMAGE_PIXELS = None

    if not os.path.exists(input_dir):
        raise ValueError("input directory does not exist. Double check your path.")

    filenames = [file for file in get_dir_contents(input_dir, **kwargs) if os.path.isfile(os.path.join(input_dir, file))]

    if len(filenames) == 0:
        raise ValueError("no image files found in input directory. Double check the directory.")

    for file in filenames:
        print("Splitting", file)
        try:
            img = Image.open(os.path.join(input_dir, file))
        except OSError as e:
            print(e)
            continue

        # if dim is specified, just use that. Make sure nothing is out of bounds
        # then just go ahead and call split_img_by_dim
        if dim:
            if len(img.size) != len(dim):
                raise ValueError("slice size must have same number of dimensions as image size")

            for a, b in list(zip(dim, img.size)):
                if a > b:
                    raise ValueError("Slice dimensions cannot exceed image dimensions.")

            imgs = split_img_by_dim(img, dim)

        # If col and row specified,
        # we have to calculate dim manually, THEN call split_img_by_dim
        elif columns>=1 and rows>=1:
            width,height=img.size
            slice_width=math.ceil(width/columns)
            slice_height=math.ceil(height/rows)
            calculated_dim=(slice_height, slice_width)
            imgs=split_img_by_dim(img,calculated_dim)
        else:
            raise ValueError("""
                 Invalid arguments supplied to function.
                 Check that you passed a 2-tuple or a column>=1 and rows>=1.
                 """)

        # Now that we have the images, save them to output_dir
        # First, split the filename into the path and the base filename

        basepath, base_name = os.path.split(file)

        # Now save each tile to the output_dir using the same file structure as input_dir if applicable
        for i, img in enumerate(imgs):
            name=''

            if append_dirname:
                if '/' in input_dir:
                    sep='/'
                else:
                    sep='\\'
                dirname = input_dir.split(sep)[-1]
                name = dirname + " " + name

            if append_datetime:
                t = datetime.now()
                s = t.strftime(format="%m-%d-%Y %H-%M-%S")
                name = name + " " + s

            if append_string!=None:
                name = str(append_string) + " " + name

            name = os.path.join(name + ' ' + str(i) + ' ' + base_name)
            path = os.path.join(output_dir, basepath, name)

            # Check the directory structure and create it if it does not exist
            if not os.path.exists(os.path.dirname(path)):
                try:
                    os.makedirs(os.path.dirname(path))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            img.save(path)


def split_img_by_dim(img, dim: tuple) -> list:
    """
    Splits a PIL img into slices of size dim or smaller

    Args:
        :param img: PIL img to split

        :param dim: max dim of output images (height, width)
    
    Returns:
        :return: A list of images of size dim or smaller
    """
    
    imgs=[]
    width,height=img.size
    slice_height, slice_width = dim
    
    # figure out number of cols/rows to split into
    x_iterations = math.ceil(width/slice_width)
    y_iterations = math.ceil(height/slice_height)

    for x in range(x_iterations):
        for y in range(y_iterations):

            # PIL requires 2 coordinates:
            # upper-left
            # lower-right coordinates
            left = x*slice_width
            upper = y*slice_height

            right = min(width, (x+1)*slice_width)
            lower = min(height, (y+1)*slice_height)

            bbox = (left, upper, right, lower)
            Slice = img.crop(bbox)

            imgs.append(Slice)

    return imgs


def sha256_checksum(file_path: str) -> str:
    """
    Calculates the checksum of a file using the SHA-256 hashing algorithm and returns the value as a hex string

    Args:
        :param file_path: The path to the file

    Returns:
        A hex string of the file hash
    """

    # Initialize hash
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K bytes in size
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    # Return the hash as a hex string
    return sha256_hash.hexdigest()
