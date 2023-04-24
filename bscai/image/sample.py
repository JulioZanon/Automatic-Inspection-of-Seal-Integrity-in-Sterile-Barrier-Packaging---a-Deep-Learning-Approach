"""
This module contains functions used to randomly or semi-randomly sample tiles from an image or array that meet
certain required conditions (e.g. they contain a given defect type)
"""

import numpy as np
import random
import itertools


def expand_grid(*itrs):
    """
    Function that returns all possible combinations of the iterators supplied as arguments to the function.
    Based on code found here:
    https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python
    Designed to mimic the functionality of the expand.grid function in R

    Args:
        :param itrs: two or more finite iterable objects (e.g. lists, generators, etc.)

    Returns:
        :return a list of lists with each entry corresponding to a unique combination of elements of the input
        iterators.

    """

    product = list(itertools.product(*itrs))
    combinations = [[x[i] for x in product] for i in range(len(itrs))]
    return np.transpose(combinations).tolist()


def tile_slice(start_point: list or tuple, tile_dims: list or tuple) -> tuple:
    """
    Function that creates a slicing object that can be used to cut a parent array down to a target rectangular tile

    Args:
        :param start_point: The coordinates of the starting point. Dimensions that are not to be sliced down should
        contain 'None' entries
        :param tile_dims: The shape of the tile. Dimensions that are not to be sliced down should contain 'None' entries

    Returns:
        :return a tuple slice object that can be passed to the parent array to yield the desired tile. Example:
        tile = parent_arr[output] where 'output' is the object returned by this function.

    """

    # Initialize the slicing object as a list of the appropriate length
    cut = [slice(None), ] * (len(tile_dims))

    # Loop through each dimension
    for i in range(len(tile_dims)):

        # If the dimension needs to be sliced to a smaller target size, create the slice and add it to the list
        if tile_dims[i] is not None:
            cut[i] = slice(start_point[i], (start_point[i] + tile_dims[i]), 1)

    # Convert the list to a tuple and return it
    return tuple(cut)


def check_array_counts(array: np.ndarray, summarized_axes: tuple, min_labeled: int = None,
                       max_labeled: int or list = None, boundary_label_gap: int or list = None) -> bool:
    """
    Function that checks a tile array to see if minimum and maximum labeled pixel count requirements are met.

    Args:
        :param array: The array to check for conditions
        :param summarized_axes: Argument passed to array.sum() axis argument for condition checking
        :param min_labeled: minimum value of sum of pixels in random tile for at least one channel for the random
        tile to be valid. May also be a list with one entry per channel of parent_arr
        :param max_labeled: maximum value of sum of pixels in random tile that cannot be exceeded in any channel for
        the random tile to be valid. May also be a list with one entry per channel of parent_arr.
        :param boundary_label_gap: The pixel width around the border of the tile that must be free of all labeled
        pixels for the tile to be valid. (e.g. a value of 10 will make sure that the first 10 pixels of the tile from
        any border will not contain a labeled pixel value of 1 for any entry in parent_arr)

    Returns:
        :return True if all requirements are satisfied. False if at least one condition is not satisfied.

    """

    # Calculate the sum of pixels in each last channel in the proposal region
    array_sum = array.sum(axis=summarized_axes)

    # Set success variable to True
    success = True

    # Check minimum label requirements
    if min_labeled is not None:

        # If minimum label requirements are not met, set success to False:
        if np.all(array_sum < min_labeled):
            success = False

    # Check maximum label requirements if nothing has failed yet
    if max_labeled is not None and success:

        # If maximum label requirements are not met, set success to False:
        if np.any(array_sum > max_labeled):
            success = False

    # Check boundary gap requirements if nothing has failed yet
    if boundary_label_gap is not None and success:

        # If boundary label gap is an integer, convert it to a list with identical entries
        if isinstance(boundary_label_gap, int):
            boundary_label_gap = [boundary_label_gap] * (1 + max(summarized_axes))

        # Get cut-points for inner tile (without borders)
        cut = [slice(None), ] * len(np.shape(array))

        for i in summarized_axes:
            cut[i] = slice(boundary_label_gap[i], -boundary_label_gap[i], 1)

        # If the full array sum is larger, there are labeled pixels in the border region
        if np.any(array_sum > np.sum(array[tuple(cut)], axis=summarized_axes)):
            success = False

    return success


def random_tile(parent_arr: np.ndarray, tile_shape: tuple, valid_range: list or np.ndarray = None,
                search_method: str = 'random', max_tries: int = 1000, **kwargs) -> dict:
    """
    Function that randomly samples a cropped array region of the given dimensions from a parent numpy array.

    Args:
        :param parent_arr: The parent numpy array to sample tile proposals from
        :param tile_shape: A tuple of the cropped tile shape to yield (should have 1 less dimension than parent_arr)
        :param valid_range: A len(tile_shape) x 2 numpy array or list giving the valid range that a random tile can be
        pulled from for each dimension of the parent_array. The range corresponds to valid starting locations of the
        tile corner with the smallest coordinate value across all axes.
        :param search_method: The search method to use to create proposal cropped regions
        :param max_tries: The maximum number of attempts to yield a valid image before the routine will exit and
        indicate that no tile could be successfully generated.
        :param kwargs: Additional parameters passed to check_array_counts

    Returns:
        :return a dictionary with the following entries:
            :return Success: boolean indicator if a successful tile was identified
            :return StartPoint: a list of the starting corner coordinates for the tile if a successful tile was
            :return NumberOfTries: the number of tries attempted before the function exited (either successfully or
            unsuccessfully if max_tries was exceeded)

    """

    # Create a matrix of the minimum and maximum possible starting points for a tile:
    max_range = np.zeros(shape=(len(tile_shape), 2))

    for i in range(len(tile_shape)):
        max_range[i, 1] = np.shape(parent_arr)[i] - tile_shape[i]

    max_range = np.array(max_range, dtype='int')

    # Set valid_range to max_range if no parameters were provided:
    if valid_range is None:
        valid_range = max_range

    # Convert valid_range to numpy array if it is a list
    if isinstance(valid_range, list):
        valid_range = np.array(valid_range)

    # Convert valid_range to integer if it is not already an integer type
    if valid_range.dtype != np.dtype('int'):
        valid_range = np.array(valid_range, dtype='int')

    # Create a tuple that contains all array indices but the last one from the parent array
    all_but_last = tuple(range(parent_arr.ndim - 1))

    # Grid based search method. Good for reducing number of runs when a high proportion of the defect must be in a tile
    if search_method == 'grid':

        i = 0
        success = False
        while not success and i < max_tries - 1:

            # Define search grid step size (proportion of tile length or width)
            grid_step = 1

            # Define grid definition starting point
            start = [random.randint(x[0], min(x[1], x[0] + y)) for x, y in zip(valid_range, tile_shape)]

            # Define valid search grid number of steps for each tile
            steps = [1 + int((x[1] - z) / (y * grid_step)) for x, y, z in zip(valid_range, tile_shape, start)]

            # Generate search grid starting points for each tile
            grid_starts = []
            for j in range(len(steps)):
                grid_starts.append([x * tile_shape[j] * grid_step + start[j] for x in range(steps[j])])

            # Translate to list of all grid starting points
            grid_starts = expand_grid(*grid_starts)

            # Randomize grid search starting coordinates
            random.shuffle(grid_starts)

            # Loop through search grid to attempt to yield tile
            for start_point in grid_starts:

                # Get candidate tile by constructing a tuple of slices for each dimension, then slicing the parent
                # array (much faster than np.take or other broadcast operations since array is not copied):
                cut = tile_slice(start_point + [None], list(tile_shape) + [None])
                candidate = parent_arr[cut]

                # Check label count requirements
                success = check_array_counts(array=candidate, summarized_axes=all_but_last, **kwargs)

                # If successful tile was generated, break out of while loop
                if success:
                    break

                # Increment max tries counter
                i += 1

                # Break out of loop if maximum number of tries exceeded
                if i >= max_tries - 1:
                    break

    # Completely random search method (good for ensuring no bias in sampling but potentially slower than grid)
    if search_method == 'random':

        for i in range(0, max_tries):

            # Randomly select tile starting point:
            start_point = [random.randint(x[0], x[1]) for x in valid_range]

            # Get candidate tile by constructing a tuple of slices for each dimension, then slicing the parent
            # array (much faster than np.take or other broadcast operations since array is not copied):
            cut = tile_slice(start_point + [None], list(tile_shape) + [None])
            candidate = parent_arr[cut]

            # Check label count requirements
            success = check_array_counts(array=candidate, summarized_axes=all_but_last, **kwargs)

            # If successful tile was generated, break out of while loop
            if success:
                break

    # Set start_point to None if a tile that met requirements was not found
    if not success:
        start_point = None

    return {'Success': success,
            'StartPoint': start_point,
            'NumberOfTries': i + 1}