"""
This module is responsible for managing functions that belong the manipulation of the image array loaded using the
cv2/PIL library.

TODO: Update this script and determine what should be deprecated and what should be kept and maintained
"""

import numpy as np
from image_ai.boxes.yolo_box import YOLOv3Box


def crop_image_array(image_array: np.ndarray,
                     upper_left_coordinates: np.array,
                     bottom_right_coordinates: np.array) -> np.ndarray:
    """
    Returns a Crop from the original image array. This crop is bound by the given arguments of the
    upper_left_coordinates and the bottom_right_coordinates.

    Args:
        :param image_array: A 3-Dimensional numpy.ndarray denoting the RGB representation of an image. This should
                            correspond to the array loaded form the image using the cv2.imread function
        :param upper_left_coordinates: A 1-Dimensional numpy.array that will denote the x and y coordinates of the
                                       upper left bound of the crop that needs to be generated.
        :param bottom_right_coordinates: A 1-Dimensional numpy.array that will denote the x and y coordinates of the
                                         bottom right bound of the crop that needs to be generated.

    Returns:
        :return: A 3-Dimensional numpy.ndarray that denotes the cropped region from the original image_array given as
                 the argument to the function.
    """
    if len(image_array.shape) != 3 or image_array.shape[-1] != 3:
        raise ValueError("Illegal Shape for the Input Image Array!")
    elif upper_left_coordinates.shape != (2,) or bottom_right_coordinates.shape != (2,):
        raise ValueError("Illegal Shape for the Crop Coordinates Array!")
    else:
        return np.array(image_array[int(upper_left_coordinates[-1]):int(bottom_right_coordinates[-1]),
                        int(upper_left_coordinates[0]):int(bottom_right_coordinates[0]), :])


def draw_annotated_defect_region(image_array: np.ndarray, defect_entity_coordinates_array: np.ndarray,
                                 color: tuple = (255, 0, 0)) -> None:
    """
    A function that shades the annotated region of the defect in the image array with the corresponding color
    specified in the function parameters

    Args:
        :param image_array: A 3-Dimensional numpy array representing the image
        :param defect_entity_coordinates_array: A 2-Dimensional array representing the coordinates which have been
                                                annotated as a defect entity
        :param color: The color of the box outline. A tuple of length 3 of the format: R, G, B

    Returns:
        :return: None
    """
    if len(defect_entity_coordinates_array.shape) != 2:
        raise ValueError(f'Expecting a 2 Dimensional Array representing the defect entity coordinates')
    image_array[defect_entity_coordinates_array[:, 0], defect_entity_coordinates_array[:, 1]] = color


def draw_bounding_box(image_array: np.ndarray, box_info: YOLOv3Box,
                      box_line_width: int = 5, color: tuple = (255, 0, 0)) -> None:
    """
    This function modifies the image array such that the box outline is drawn on the image itself with the color
    specified in the function parameter. The box drawn is of the box line width.

    Args:
        :param image_array: A 3-Dimensional numpy array representing the image
        :param box_info: The YOLOv3Box type object which contains details about the YOLOv3 box annotation
        :param box_line_width: The width of the box that needs to be drawn
        :param color: The color of the box outline. A tuple of length 3 of the format: R, G, B

    Returns:
        :return: None
    """
    # Top Horizontal Line
    image_array[box_info.get_top_left_x():box_info.get_bottom_right_x(),
    box_info.get_top_left_y():box_info.get_top_left_y() + box_line_width] = color
    # Bottom Horizontal Line
    image_array[box_info.get_top_left_x():box_info.get_bottom_right_x(),
    box_info.get_bottom_right_y():box_info.get_bottom_right_y() + box_line_width] = color
    # Left Vertical Line
    image_array[box_info.get_top_left_x():box_info.get_top_left_x() + box_line_width,
    box_info.get_top_left_y():box_info.get_bottom_right_y()] = color
    # Right Vertical Line
    image_array[box_info.get_bottom_right_x():box_info.get_bottom_right_x() + box_line_width,
    box_info.get_top_left_y():box_info.get_bottom_right_y()] = color
