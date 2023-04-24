import math

import matplotlib.pyplot as plt
import numpy as np
#from FastLine import Line
import math
import cv2
#from sewar import mse, rmse, psnr, ssim, uqi, msssim, ergas, scc,rase, sam, vifp
import os
import subprocess as sp
from typing import cast

def get_img_path_array(folder_path = '', file_name_filter = '', file_extension_filter = '', img_idx_array = []):
    # construct array with paths in folder
    if len(file_name_filter) > 0:
        array_paths = np.array(
            [os.path.join(path, name) for path, subdirs, files in os.walk(folder_path) for name in files if
             str.lower(os.path.splitext(name)[1]) == file_extension_filter and str.lower(file_name_filter) in str.lower(
                 os.path.splitext(name)[0])])
    else:
        array_paths = np.array(
            [os.path.join(path, name) for path, subdirs, files in os.walk(folder_path) for name in files if
             str.lower(os.path.splitext(name)[1]) == file_extension_filter])

    if len(img_idx_array) > 0:
        array_paths = array_paths[img_idx_array]

    return array_paths



def slop_to_deg(slope):
    return np.arctan(slope) * 180 / np.pi

def angle_dif(a1, a2):
    dif = abs(a1-a2)
    if dif > 90: dif = 180 - dif
    return dif

def line_length_angle(p1= (0, 0), p2=(0, 0) ):
    """
    Returns the length of the line as well as the angle in degrees based on a
    180 deg Clockwise with horizontal = 0 deg, clockwise from 0 deg = (+) Deg and
    vertical = 90 deg. Anticlockwise (after passing 90 deg) = (-)
    :param p1: x, y where x = 0 is in the top corner and x max is on the bottom
    :param p2: x, y where y = 0 is in the left corner and y max is on the right hand side
    :return: length, angle in degrees.
    """
    length = np.abs(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))
    h = (p2[0] - p1[0]) # x:
    d = (p2[1] - p1[1]) # y:
    if d != 0:
        angle = np.arctan(h / d) * 180 / np.pi
    else:
        angle = 90

    return length, angle


def line_length(p1= (0, 0), p2=(0, 0) ):
    return np.abs(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))


def extend_line_along_ROI(roi = [0, 0, 0, 0], line = [0, 0, 0, 0]):
    """
    Extend a line given by two points to intersect the sides of a ROI.
    :param roi: x_origin, y_origin, x_size, y_size
    :param line: x1, y1, x2, y2
    :return: points of a line that intersect in ROI ix1, iy1, ix2, iy2
    """
    intersection_pts = None
    x1 = np.rint(line[0])
    y1 = np.rint(line[1])
    x2 = np.rint(line[2])
    y2 = np.rint(line[3])

    line_segment = Line(p1=(x1, y1), p2=(x2, y2))
    roi = np.rint(roi)
    x_origin, y_origin, x_size, y_size = roi

    roi_lines = np.zeros(4 * 4, dtype=float).reshape([4, 4])
    roi_lines[0,0], roi_lines[0,1], roi_lines[0,2], roi_lines[0,3] = x_origin, y_origin, x_origin + x_size, y_origin # roi left side
    roi_lines[1,0], roi_lines[1,1], roi_lines[1,2], roi_lines[1,3]  = x_origin, y_origin, x_origin, y_origin + y_size # roi top side
    roi_lines[2,0], roi_lines[2,1], roi_lines[2,2], roi_lines[2,3]  = x_origin, y_origin + y_size, x_origin + x_size, y_origin + y_size # roi right side
    roi_lines[3,0], roi_lines[3,1], roi_lines[3,2], roi_lines[3,3]  = x_origin + x_size, y_origin, x_origin + x_size, y_origin + y_size # roi left side

    for i, roi_line in enumerate(roi_lines):
        x1, y1, x2, y2 = roi_line
        intersection = line_segment.intersection(Line(p1=(x1, y1), p2=(x2, y2)))
        if intersection is not None:
            if intersection_pts is None:
                intersection_pts = np.expand_dims(np.array(intersection), axis=0)
            else:
                intersection_pts = np.append(intersection_pts, np.expand_dims(np.array(intersection), axis=0), axis=0)

    # check the two pt of which its limits are within the ROI
    if intersection_pts.shape[0]  == 4:
        idx = np.zeros(4, dtype=bool)
        for i, pt in enumerate(intersection_pts):
            if np.rint(pt[0]) >= x_origin and np.rint(pt[0]) <= x_origin + x_size and np.rint(pt[1]) >= y_origin and np.rint(pt[1]) <= y_origin + y_size: idx[i] = True
        #check if 4 pts where give i.e. diagonal lines
        if np.sum(idx) == 4:
            idx[0] = False
            idx[2] = False
        intersection_pts = intersection_pts[idx, ...]

    intersection_pts = np.abs(intersection_pts)

    return intersection_pts


def merge_lines(lines, img = None, roi = [0, 0, 0, 0], pix_dist_step = [5], ang_dif_step = [10], transition = 'DarkToBright', min_length = 10, max_length = 4000):

    """
    Cluster lines based on proximity of the points and the angle of the line. Then clusters are converted into single lines
    TODO: Transition 'DarkToBright', currently np.polyfit is done with all points of close lines. These points should be
    replaced for interception points if the lines cross.
    :param lines:
    :param img:
    :param roi: x_origin, y_origin, x_size, y_size
    :param pix_dist:
    :param ang_diff:
    :param num_of_segments:
    :param transition:
    :param min_length:
    :param max_length:
    :return:
    """
    t = 0
    if len(pix_dist_step) != len(ang_dif_step): raise RuntimeError('pix_dist_step and ang_dif_step must have the same size')


    for pix_dist, ang_dif in zip(pix_dist_step, ang_dif_step):

        # get roi
        x_origin, y_origin, x_size, y_size = roi
        # initialise arrays

        # loop trhough the lines
        line_list = list()
        # convert points to lines and filter out those with points outside the roi
        for i, line in enumerate(lines):
            if (line[0] >= x_origin and line[1] >= y_origin) and (line[0] <= x_origin + x_size and line[1] <= y_origin + y_size) and (
                    line[2] >= x_origin and line[3] >= y_origin) and (line[2] <= x_origin + x_size and line[3] <= y_origin + y_size):
                line_list.append(Line(p1=(line[0], line[1]), p2=(line[2], line[3])))

        if len(line_list) == 0: raise RuntimeError('No lines where found within the roi. Increase ROI size')

        # create cluster of lines by extending on line at the time across the roi and checking the distance of the points from other lines
        cluster = np.zeros(len(line_list) * len(line_list), dtype=bool).reshape(len(line_list), len(line_list))
        distances_to_p1 = np.zeros(len(line_list) * len(line_list), dtype=int).reshape(len(line_list), len(line_list))
        distances_to_p2 = np.zeros(len(line_list) * len(line_list), dtype=int).reshape(len(line_list), len(line_list))
        angle_diff = np.zeros(len(line_list) * len(line_list), dtype=float).reshape(len(line_list), len(line_list))

        i = 0
        for line in line_list:
            extended_line = extend_line_along_ROI(roi = roi, line = [line.p1[0], line.p1[1], line.p2[0], line.p2[1]])
            extended_line = Line(p1=(extended_line[0, 0], extended_line[0, 1]), p2=(extended_line[1, 0], extended_line[1, 1]))
            j = 0
            for test_line in line_list:
                if i != j:
                    distances_to_p1[i, j] = np.rint(extended_line.distance_to(test_line.p1))
                    distances_to_p2[i, j] = np.rint(extended_line.distance_to(test_line.p2))
                    _, ang1 = line_length_angle(extended_line.p1, extended_line.p2)
                    _, ang2 = line_length_angle(test_line.p1, test_line.p2)
                    angle_diff[i, j] = angle_dif(ang1, ang2)

                    if (distances_to_p1[i, j] <= pix_dist or distances_to_p2[i, j] <= pix_dist) and angle_diff[i, j]  <= ang_dif:
                        cluster[i, j] = True
                else:
                    cluster[i, j] = True
                j += 1
            i += 1

        # # Delete rows in cluster array that did not have any proximity
        # loop = cluster.shape[0]
        # for i in range(loop):
        #     j = loop - 1 - i
        #     if sum(cluster[j]) <= 1:
        #         cluster = np.delete(cluster, j, 0)

        # delete repeated rows
        idx = list()
        for i in range(cluster.shape[0]):
            for j in range(i+1, cluster.shape[0]):
                if np.array_equal(cluster[i], cluster[j]): idx.append(j)
        idx = np.unique(np.array(idx))
        cluster = cluster[np.bitwise_not(np.isin(np.arange(cluster.shape[0]), idx)), ...]

        # # convert cluster to indices
        indices = list()
        for i in range(cluster.shape[0]):
            indices.append(np.where(cluster[i]))

        # construct lines with the points of the clustered lines as per indices
        merged_lines = list()
        for i in indices:
            list_of_points = list()
            for j in i[0]:
                list_of_points.append(line_list[j].p1)
                list_of_points.append(line_list[j].p2)

            x = np.array(list_of_points)[:, 0]
            y = np.array(list_of_points)[:, 1]
            if np.max(x) - np.min(x) < np.max(y) - np.min(y): # Horizontal line
                roi_extend = [x_origin, np.min(y), x_size, np.max(y) - np.min(y)]
            elif np.max(x) - np.min(x) > np.max(y) - np.min(y): # Vertical line
                roi_extend = [np.min(x), y_origin, np.max(x) - np.min(x), y_size]
            else: #squere roi
                roi_extend = [np.min(x), np.min(y), np.max(x) - np.min(x), np.max(y) - np.min(y)]


            theta = np.polyfit(x, y, 1)
            x = np.sort(x)
            fit_y_line = theta[1] + theta[0] * x
            fit_line_p1 =  (x[0], fit_y_line[0])
            fit_line_p2 = (x[-1], fit_y_line[-1])
            fit_line = Line(p1=fit_line_p1, p2=fit_line_p2)
            if not math.isnan(fit_line.m) and not math.isinf(fit_line.m):
                fit_line_extended_1 = extend_line_along_ROI(roi=roi_extend, line=[fit_line.p1[0], fit_line.p1[1], fit_line.p2[0], fit_line.p2[1]])
                fit_line_extended_p1 =  (fit_line_extended_1[0, 0], fit_line_extended_1[0, 1])
                fit_line_extended_p2 = (fit_line_extended_1[1, 0], fit_line_extended_1[1, 1])
                fit_line_extended = Line(p1=fit_line_extended_p1, p2=fit_line_extended_p2)
                length = line_length(fit_line_extended.p1, fit_line_extended.p2)
                if length >= min_length and length <= max_length:
                    #TODO: Check transition Dark to Bright and Bright to dark
                    merged_lines.append(fit_line_extended)

        lines = np.zeros(len(merged_lines) * 4, dtype=float).reshape(len(merged_lines), 4)
        angles = np.zeros(len(merged_lines) , dtype=float)
        lengths = np.zeros(len(merged_lines), dtype=float)

        # Convert to arrays
        for i, line in enumerate(merged_lines):
            lines[i, ...] = np.array([line.p1[0], line.p1[1], line.p2[0], line.p2[1]])
            #angles[i] = slop_to_deg(line.m)
            #lengths[i] = line_length(line.p1, line.p2)
            lengths[i], angles[i] = line_length_angle(line.p1, line.p2)
        #Stop clustering
        if len(merged_lines) == 1: break

    # order by length and build list

    idx = np.flip(np.argsort(lengths))
    merged_lines = list()
    for i in idx:
        line = {
            'p1': lines[i, 0: 2],
            'p2': lines[i, 2: 4],
            'length': np.abs(lengths[i]),
            'angle': angles[i]
        }
        merged_lines.append(line)


    return merged_lines


def find_line(img, t01):
    # read image
    img_raw = img
    if t01['print_functions']:
        plt.imshow(img_raw)
        plt.title('Original')
        plt.show()
    # crop one channel
    img = np.copy(img_raw[t01['crop_origin'][0]:t01['crop_origin'][0] + t01['crop_size'][0],
                  t01['crop_origin'][1]:t01['crop_origin'][1] + t01['crop_size'][1], 0])
    # img[-10:,...] = 0
    if t01['print_functions']:
        plt.imshow(img, cmap='gray')
        plt.title('Crop BW')
        plt.show()

    # gaussian filter to smooth image before finding edges
    img_gb = cv2.GaussianBlur(img, t01['gaussian_kernel'], t01['gaussian_sigma'])
    if t01['print_functions']:
        plt.imshow(img_gb, cmap='gray')
        plt.title('gaussian Blure')
        plt.show()
    # thresoling image to accentuate transitions
    ret, img_thr = cv2.threshold(img_gb, t01['threshol_min'], t01['threshol_max'], cv2.THRESH_TOZERO)
    if t01['print_functions']:
        plt.imshow(img_thr, cmap='gray')
        plt.title('Thresholding')
        plt.show()
    # edge detector
    img_c = cv2.Canny(img_thr, t01['canny_thr1'], t01['canny_thr2'])
    if t01['print_functions']:
        plt.imshow(img_c, cmap='gray')
        plt.title('Canny Operator')
        plt.show()

    # Detect points that form a line
    lines = cv2.HoughLinesP(img_c, rho=t01['HoughLinesP_rho'], theta=np.pi / 180, threshold=t01['HoughLinesP_thr'], minLineLength=10, maxLineGap=500)
    lines = lines[:, 0, :]  # remove dimension. [lineIdx, [y1, x1, y2, x2]]
    for i in range(lines.shape[0]):  # change x and y: [lineIdx, [x1, y1, x2, y2]]
        lines[i] = np.array([lines[i][1], lines[i][0], lines[i][3], lines[i][2]])

    # Draw lines on the image - Crop original image with 3 channels to be able to print countour in red
    img_fl = np.copy(img_raw[t01['crop_origin'][0]:t01['crop_origin'][0] + t01['crop_size'][0],
                     t01['crop_origin'][1]:t01['crop_origin'][1] + t01['crop_size'][1], :])

    for i, line in enumerate(lines):
        y1, x1, y2, x2 = line
        cv2.line(img_fl, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # draw search roi
    top_line_search_roi = [t01['top_line_search_ROI_x'][0], t01['top_line_search_ROI_y'][0],
                           t01['top_line_search_ROI_x'][1] - t01['top_line_search_ROI_x'][0],
                           t01['top_line_search_ROI_y'][1] - t01['top_line_search_ROI_y'][0]]
    cv2.rectangle(img_fl, (top_line_search_roi[1], top_line_search_roi[0]),
                  (top_line_search_roi[1] + top_line_search_roi[3], top_line_search_roi[0] + top_line_search_roi[2]),
                  (0, 0, 255), 3)

    left_line_search_roi = [t01['left_line_search_ROI_x'][0], t01['left_line_search_ROI_y'][0],
                            t01['left_line_search_ROI_x'][1] - t01['left_line_search_ROI_x'][0],
                            t01['left_line_search_ROI_y'][1] - t01['left_line_search_ROI_y'][0]]
    cv2.rectangle(img_fl, (left_line_search_roi[1], left_line_search_roi[0]), (
    left_line_search_roi[1] + left_line_search_roi[3], left_line_search_roi[0] + left_line_search_roi[2]), (0, 255, 0),
                  3)

    right_line_search_roi = [t01['right_line_search_ROI_x'][0], t01['right_line_search_ROI_y'][0],
                             t01['right_line_search_ROI_x'][1] - t01['right_line_search_ROI_x'][0],
                             t01['right_line_search_ROI_y'][1] - t01['right_line_search_ROI_y'][0]]
    cv2.rectangle(img_fl, (right_line_search_roi[1], right_line_search_roi[0]), (
    right_line_search_roi[1] + right_line_search_roi[3], right_line_search_roi[0] + right_line_search_roi[2]),
                  (0, 255, 0), 3)

    if t01['print_functions']:
        plt.imshow(img_fl)
        plt.title('Hugh Lines and Search ROI')
        plt.show()

    # merge TOP lines
    merged_lines_top = merge_lines(lines, roi=top_line_search_roi, pix_dist_step=t01['top_line_pix_dist'],
                                   ang_dif_step=t01['top_line_ang_dif'], transition='DarkToBright',
                                   min_length=t01['top_line_min_length'], max_length=t01['top_line_max_length'])

    img_mlt = np.copy(img_raw[t01['crop_origin'][0]:t01['crop_origin'][0] + t01['crop_size'][0],
                      t01['crop_origin'][1]:t01['crop_origin'][1] + t01['crop_size'][1], :])
    for line in merged_lines_top:
        y1, x1, y2, x2 = int(np.rint(line['p1'][1])), int(np.rint(line['p1'][0])), int(np.rint(line['p2'][1])), int(
            np.rint(line['p2'][0]))
        cv2.line(img_mlt, (y1, x1), (y2, x2), (255, 0, 0), 3)
    if t01['print_functions']:
        plt.imshow(img_mlt)
        plt.title('Top Fitted Lines -- Total: ' + str(len(merged_lines_top)))
        plt.show()

    # merge LEFT lines
    merged_lines_left = merge_lines(lines, roi=left_line_search_roi, pix_dist_step=t01['left_line_pix_dist'],
                                    ang_dif_step=t01['left_line_ang_dif'], transition='DarkToBright',
                                    min_length=t01['left_line_min_length'], max_length=t01['left_line_max_length'])

    img_mll = np.copy(img_raw[t01['crop_origin'][0]:t01['crop_origin'][0] + t01['crop_size'][0],
                      t01['crop_origin'][1]:t01['crop_origin'][1] + t01['crop_size'][1], :])
    for line in merged_lines_left:
        y1, x1, y2, x2 = int(np.rint(line['p1'][1])), int(np.rint(line['p1'][0])), int(np.rint(line['p2'][1])), int(
            np.rint(line['p2'][0]))
        cv2.line(img_mll, (y1, x1), (y2, x2), (255, 0, 0), 3)
    if t01['print_functions']:
        plt.imshow(img_mll)
        plt.title('Left Fitted Lines -- Total: ' + str(len(merged_lines_left)))
        plt.show()

    # merge RIGHT lines
    merged_lines_right = merge_lines(lines, roi=right_line_search_roi, pix_dist_step=t01['right_line_pix_dist'],
                                     ang_dif_step=t01['right_line_ang_dif'], transition='DarkToBright',
                                     min_length=t01['right_line_min_length'], max_length=t01['right_line_max_length'])

    img_mlr = np.copy(img_raw[t01['crop_origin'][0]:t01['crop_origin'][0] + t01['crop_size'][0],
                      t01['crop_origin'][1]:t01['crop_origin'][1] + t01['crop_size'][1], :])
    for line in merged_lines_right:
        y1, x1, y2, x2 = int(np.rint(line['p1'][1])), int(np.rint(line['p1'][0])), int(np.rint(line['p2'][1])), int(
            np.rint(line['p2'][0]))
        cv2.line(img_mlr, (y1, x1), (y2, x2), (255, 0, 0), 3)
    if t01['print_functions']:
        plt.imshow(img_mlr)
        plt.title('right Fitted Lines -- Total: ' + str(len(merged_lines_right)))
        plt.show()

    # Extend lines and get intersections to define the 2 points of the puch top edge.

    top_line = Line(p1=merged_lines_top[0]['p1'], p2=merged_lines_top[0]['p2'])
    left_line = Line(p1=merged_lines_left[0]['p1'], p2=merged_lines_left[0]['p2'])
    right_line = Line(p1=merged_lines_right[0]['p1'], p2=merged_lines_right[0]['p2'])

    p1 = top_line.intersection(left_line)
    p2 = top_line.intersection(right_line)
    l, a = line_length_angle(p1, p2)
    p12 = ((np.sin(a * np.pi / 180) * l / 2) + p1[0], (np.cos(a * np.pi / 180) * l / 2) + p1[1])

    line = {
        'p1x': int(np.rint(p1[0])),
        'p1y':  int(np.rint(p1[1])),
        'p2x': int(np.rint(p2[0])),
        'p2y': int(np.rint(p2[1])),
        'p12x': int(np.rint(p12[0])),
        'p12y': int(np.rint(p12[1])),
        'length': l,
        'angle': a
    }

    img_edge = np.copy(img_raw[t01['crop_origin'][0]:t01['crop_origin'][0] + t01['crop_size'][0],
                       t01['crop_origin'][1]:t01['crop_origin'][1] + t01['crop_size'][1], :])
    cv2.line(img_edge, (line['p1y'], line['p1x']), (line['p2y'], line['p2x']), (255, 0, 0), 3)
    cv2.circle(img_edge, (line['p12y'], line['p12x']), 10, (0, 255, 0), 5)
    if t01['print_result']:
        plt.imshow(img_edge)
        plt.title('pouch edge and centre')
        plt.show()
    # convert points to original image size (i.e. add ROI ofsets.)
    line['p1x'] = line['p1x'] + t01['crop_origin'][0]
    line['p1y'] = line['p1y'] + t01['crop_origin'][1]
    line['p2x'] = line['p2x'] + t01['crop_origin'][0]
    line['p2y'] = line['p2y'] + t01['crop_origin'][1]
    line['p12x'] = line['p12x'] + t01['crop_origin'][0]
    line['p12y'] = line['p12y'] + t01['crop_origin'][1]

    return line


def stats_dic(array = []):
    stats_dic = {
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'mean': np.mean(array),
        'median': np.median(array),
        'std': np.std(array)
    }
    return stats_dic

def normalise_array(array, min, max):
    return (array - min) / (max - min)

def get_intensity_stats(img, t02, origin = (0, 0)):
    # read image
    img_raw = img
    img_roi = np.copy(img)

    # Dark area
    dark_roi = [origin[0] + t02['crop_origin_dark'][0], origin[1] + t02['crop_origin_dark'][1],
                t02['crop_size_dark_bright'][0], t02['crop_size_dark_bright'][1]]
    dark_roi = np.array(dark_roi, dtype=int)
    dark_img = np.copy(img_raw[dark_roi[0]: dark_roi[0] + dark_roi[2], dark_roi[1]: dark_roi[1] + dark_roi[3], 0])
    # Bright area
    bright_roi = [origin[0] + t02['crop_origin_bright'][0], origin[1] + t02['crop_origin_bright'][1],
                t02['crop_size_dark_bright'][0], t02['crop_size_dark_bright'][1]]
    bright_roi = np.array(bright_roi, dtype=int)
    bright_img = np.copy(img_raw[bright_roi[0]: bright_roi[0] + bright_roi[2], bright_roi[1]: bright_roi[1] + bright_roi[3], 0])

    # Feature1 area
    feature1_roi = [origin[0] + t02['crop_origin_feature1'][0], origin[1] + t02['crop_origin_feature1'][1],
                t02['crop_size_feature1'][0], t02['crop_size_feature1'][1]]
    feature1_roi = np.array(feature1_roi, dtype=int)
    feature1_img = np.copy(img_raw[feature1_roi[0]: feature1_roi[0] + feature1_roi[2], feature1_roi[1]: feature1_roi[1] + feature1_roi[3], 0])

    # draw boxes to visualise ROIs
    if t02['print_functions']:
        cv2.rectangle(img_roi, (dark_roi[1], dark_roi[0]), (dark_roi[1] + dark_roi[3], dark_roi[0] + dark_roi[2]),
                      (0, 0, 255), 3)
        cv2.rectangle(img_roi, (bright_roi[1], bright_roi[0]),
                      (bright_roi[1] + bright_roi[3], bright_roi[0] + bright_roi[2]), (255, 0, 0), 3)
        cv2.rectangle(img_roi, (feature1_roi[1], feature1_roi[0]), ( feature1_roi[1] + feature1_roi[3], feature1_roi[0] + feature1_roi[2]), (0, 255, 0), 3)
        plt.imshow(img_roi)
        plt.title('Dark ROI = Blue, Bright ROI = Red, Feature1 ROI =  Green')
        plt.show()

    # create constrast img by deliting the dark img intensity from the bright image, then eliminate < 0 values and finallly normalise based on min and max.

    contrast_img = normalise_array(bright_img,np.min(dark_img),np.max(bright_img))  - normalise_array(dark_img,np.min(dark_img),np.max(bright_img))
    contrast_array = contrast_img.flatten()
    contrast_array = contrast_array[np.argwhere(contrast_array > 0)] * 100
    intensity_stats = {
        'dark_img_intensity_stats': stats_dic([np.min(dark_img), np.max(dark_img), np.mean(dark_img), np.median(dark_img), np.std(dark_img)]),
        'bright_img_intensity_stats': stats_dic([np.min(bright_img), np.max(bright_img), np.mean(bright_img), np.median(bright_img), np.std(bright_img)]),
        'contrast_normalised_bright_dark_stats': stats_dic([np.min(feature1_img), np.max(feature1_img), np.mean(feature1_img), np.median(feature1_img), np.std(feature1_img)]),
        'feature1_img_intensity_stats': stats_dic([np.min(contrast_array), np.max(contrast_array), np.mean(contrast_array), np.median(contrast_array), np.std(contrast_array)])
    }

    return intensity_stats


def img_compare_stats(base_img, sample_img, t03, base_img_origin = (0, 0), sample_img_origin = (0, 0)):

    # Feature1 area Base Image
    base_feature1_roi = [base_img_origin[0] + t03['crop_origin_feature1'][0], base_img_origin[1] + t03['crop_origin_feature1'][1],
                t03['crop_size_feature1'][0], t03['crop_size_feature1'][1]]
    base_feature1_roi = np.array(base_feature1_roi, dtype=int)
    feature1_base_img = np.copy(base_img[base_feature1_roi[0]: base_feature1_roi[0] + base_feature1_roi[2], base_feature1_roi[1]: base_feature1_roi[1] + base_feature1_roi[3], 0])
    if t03['distort_base_img']:
        # gaussian filter to smooth image before finding edges
        feature1_base_img = cv2.GaussianBlur(feature1_base_img, t03['gaussian_kernel'], t03['gaussian_sigma'])
        # thresoling image to accentuate transitions
        ret, feature1_base_img = cv2.threshold(feature1_base_img, t03['threshol_min'], t03['threshol_max'], cv2.THRESH_TOZERO)

    # Feature1 area Sample Image
    sample_feature1_roi = [sample_img_origin[0] + t03['crop_origin_feature1'][0], sample_img_origin[1] + t03['crop_origin_feature1'][1],
                t03['crop_size_feature1'][0], t03['crop_size_feature1'][1]]
    sample_feature1_roi = np.array(sample_feature1_roi, dtype=int)
    feature1_sample_img = np.copy(sample_img[sample_feature1_roi[0]: sample_feature1_roi[0] + sample_feature1_roi[2], sample_feature1_roi[1]: sample_feature1_roi[1] + sample_feature1_roi[3], 0])

    if t03['print_functions']:
        plt.figure()
        plt.subplot(211)
        plt.imshow(feature1_base_img, cmap='gray')
        plt.title('Based Image crop')
        plt.subplot(212)
        plt.imshow(feature1_sample_img, cmap='gray')
        plt.title('Sample Image crop')
        plt.show()


    # sift
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(feature1_base_img, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(feature1_sample_img, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    sift_matches = bf.match(descriptors_1, descriptors_2)
    sift_matches = sorted(sift_matches, key=lambda x: x.distance)

    if t03['print_functions']:
        if len(sift_matches) >= 50:
            num_to_draw = len(sift_matches)
        else:
            num_to_draw = -1
        sift_img = cv2.drawMatches(feature1_base_img, keypoints_1, feature1_sample_img, keypoints_2, sift_matches[:num_to_draw], feature1_sample_img, flags=2)
        plt.imshow(sift_img)
        plt.title('SIFT - number of matches ' + str(len(sift_matches)))
        plt.show()

    # surf
    surf = cv2.xfeatures2d.SURF_create()
    keypoints_1, descriptors_1 = surf.detectAndCompute(feature1_base_img, None)
    keypoints_2, descriptors_2 = surf.detectAndCompute(feature1_sample_img, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    surf_matches = bf.match(descriptors_1, descriptors_2)
    surf_matches = sorted(surf_matches, key=lambda x: x.distance)

    if t03['print_functions']:
        if len(surf_matches) >= 50:
            num_to_draw = len(surf_matches)
        else:
            num_to_draw = -1
        sift_img = cv2.drawMatches(feature1_base_img, keypoints_1, feature1_sample_img, keypoints_2, surf_matches[:num_to_draw], feature1_sample_img, flags=2)
        plt.imshow(sift_img)
        plt.title('SURF - number of matches ' + str(len(surf_matches)))
        plt.show()



    similarity_stats = {
    # https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6

        # 0 to inf where 0 denote perfect match
        'MeanSquaredError': mse(feature1_sample_img, feature1_base_img),

        # measures the amount of change per pixel due to the processing. RMSE values are non-negative 0 to inf and a value of 0 means the image or videos being compared are identical.
        'RootMeanSquaredError': rmse(feature1_sample_img, feature1_base_img),

        # 0 to inf where inf denote perfect match
        'PeakSignalNoiseRatio': psnr(feature1_sample_img, feature1_base_img),

        #quantifies image quality degradation caused by processing, such as data compression, or by losses in data transmission. 0 and 1 with 1 indicating perfect structural similarity. https://ieeexplore.ieee.org/abstract/document/1284395
        'StructuralSimilarityIndex': np.min(ssim(feature1_sample_img, feature1_base_img)),

        # is designed by modeling any image distortion as a combination of three factors: Loss of correlation, Luminance distortion, Contrast distortion. It take a range of -1 to 1 where
        # identical images score 1
        'UniversalQualityImageIndex': uqi(feature1_sample_img, feature1_base_img),

        # More flexibility than single ssim approach in incorporating the variations of image resolution and viewing conditions. https://www.cns.nyu.edu/pub/eero/wang03b.pdf
        'MultiScaleStructuralSimilarityIndex': msssim(feature1_sample_img, feature1_base_img).real,

        # 0 to inf where 0 denote perfect match
        'ErreurRelativeGlobaleAdimensionnelleSynthese': ergas(feature1_sample_img, feature1_base_img),

        # -1 to 1 where 1 denote perfect match
        'SpatialCorrelationCoefficient': scc(feature1_sample_img, feature1_base_img),

        # 0 to inf where 0 denote perfect match
        'RelativeAverageSpectralError': rase(feature1_sample_img, feature1_base_img),

        # is a physically-based spectral classification. The algorithm determines the spectral similarity between two spectra by calculating the angle between the spectra and treating them
        # as vectors in a space with dimensionality equal to the number of bands. Smaller angles represent closer matches to the reference spectrum.
        'SpectralAngleMapper': sam(feature1_sample_img, feature1_base_img),

        # 0 to inf where 1 denote perfect match
        'VisualInformationFidelity': vifp(feature1_sample_img, feature1_base_img),

        # SIFT - Feature Matching https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
        'SIFTmatch': len(sift_matches),

        # SURF - Feature Matching
        'SURFmatch': len(surf_matches)

    }

    return similarity_stats


def convert_batch_dic_to_arrays_nd_stats(batch_of_elements = []):
    num_of_items = len(batch_of_elements)

    single_value_arrays = dict()
    keys = []
    for k in batch_of_elements[0].keys():
        keys.append(k)
    for i, key in enumerate(keys):
        if type(batch_of_elements[0][key]) is dict:
            data_type = np.array(batch_of_elements[0][key]['mean']).dtype
        else:
            data_type = np.array(batch_of_elements[0][key]).dtype
        vector = np.zeros(num_of_items, dtype=data_type)
        for j, item in enumerate(batch_of_elements):
            if type(item[key]) is dict:
                vector[j] = np.array(item[key]['mean']).astype(data_type)
            else:
                vector[j] = np.array(item[key]).astype(data_type)
        vector_list =vector.tolist()
        single_value_arrays.update({key:vector_list})
    ###### generate stats dictionary
    combined_stats = dict()
    for key in keys:
        combined_stats.update({key:stats_dic(single_value_arrays[key])})
    return single_value_arrays, combined_stats

def find_divisible_nums(numarator: int, min = 1, max = 64):

    if numarator < max: max = numarator
    denominators = np.arange(start=min, stop=max+1, step=1, dtype=int)
    list_denominators = denominators.tolist()
    divisable_nums = list(filter(lambda x: ( numarator % x == 0), list_denominators))
    if len(divisable_nums)< 1:
        return np.array([1], dtype=int)
    else:
        return np.array(divisable_nums, dtype=int)



def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values