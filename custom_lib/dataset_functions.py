import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bscai.annotation.load import dict_from_jsons, get_json_data, regions_to_array
from .model_build_funtions import convert_seg_mask_to_classification_array
from custom_lib.json_function import read_from_json, dictionary_to_array
from custom_lib.data_log_functions import convert_cls_str_to_bool
dict_image_format = {
        'jpg': 1,
        'bmp': 2,
        'png': 3
    }

def get_dataset_size(dt):
    """
    Get size of dataset by itterating trhough dataset
    :param dt: dataset
    :return: count
    """
    count=0
    for element in dt:
        count+=1
    return count

def dataset_from_folder_seg(image_folder_path = '', img_params = [0, [0, 0], [0, 0], [0, 0], [0, 0]], notation_folder_path='', class_name_list= []):
    """
    Creates dataset from a folder of images and json masks
    :param folder_path: path with folder containing data set. Lables will be takeen by subfolder names.
    :param img_params: img_format[1], img_size[x,y], crop_origin=[x, y], crop_size= [x, y], img_resize= [x, y]
    :param notation_path_list: list of notation file paths
    :param class_name_list: list of classes to be included
    :return:
    """
    """
    # Build folder paths
    folder_path = os.path.join(image_folder_path, '*/*')
    # build datases from file names
    ds_files = tf.data.Dataset.list_files(str(folder_path))
    ds = ds_files.map(lambda x: process_path_seg(x, img_params, notation_dict, class_name_list), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    """
    ds_paths_masks = create_file_path_mask_dataset(image_folder_path, notation_folder_path, class_name_list, img_params)

    ds = ds_paths_masks.map(lambda x, y, z: process_path_seg((x, y, z), img_params, class_name_list),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


def dataset_from_folder(folder_path, img_params = [0, [0, 0], [0, 0], [0, 0], [0, 0]], class_name_list= []):
    """
    Creates dataset from a folder of images with classification lables.
    :param folder_path_: path with folder containing data set. Lables will be takeen by subfolder names.
    :param img_params: img_format[1], img_size[x,y], crop_origin=[x, y], crop_size= [x, y], img_resize= [x, y]
    :return: daset. List with the class names. count of records in dataset
    """
    # Extract class names from folder names
    class_name_list_from_folder = np.array(os.listdir(folder_path))

    if len(class_name_list) == 0:
        class_name_list = class_name_list_from_folder
    elif not any(i in class_name_list_from_folder for i in class_name_list):
        raise RuntimeError ('The provided class list does not match folder names. Folder names: ' + class_name_list_from_folder)
    else: class_name_list =  np.array(class_name_list)

    # Build folder paths
    folder_path = os.path.join(folder_path, '*/*')
    # build datases from file names
    ds_files = tf.data.Dataset.list_files(str(folder_path))
    ds = ds_files.map(lambda x: process_path(x, img_params, class_name_list ),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds, class_name_list.tolist()

def split_dataset(ds, split_ratio=0.7, shuffle = False):
    """
    TODO
    :param ds:
    :param split_ratio:
    :param shuffle:
    :return:
    """
    ds_size = get_dataset_size(ds)
    # calculate number of samples per set based on split ratio
    ds1_size = int(split_ratio * ds_size)
    buffer_size = ds_size #ds_size - ds1_size
    if shuffle:
        shuffled_ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        ds1 = shuffled_ds.take(ds1_size)
        ds2 = shuffled_ds.skip(ds1_size)
    else:
        ds1 = ds.take(ds1_size)
        ds2 = ds.skip(ds1_size)
    return ds1, ds2

def get_label_from_subfolders(folder_path, class_names):
    """
    TODO
    :param folder_path:
    :param class_names:
    :return:
    """
    # convert the path to a list of path components
    parts = tf.strings.split(folder_path, os.path.sep)

    # The second to last is the class-directory
    return parts[-2] == class_names


def crop_nd_size(img, img_size = [0, 0], crop_origin=[0, 0], crop_size= [150, 150], img_resize= [150,0]):
    """
    :param file_path:
    :param img_format: image format:  1 = 'jpg', 2 = 'bmp', 3 = 'png'
    :param img_resize: resize in pixels (heigth, width). if heigth == 0, then size = crop. if heigth <> 0 and width == 0, then it keeps aspect
    :param crop_origin: x1, y1 coordinates (x=vertical, y=horizontal)
    :param crop_size: size in pixels. (heigth, width). if 0, then no crop is applied
    :return: image
    """

    # Re-size image as per img_resize input
    # If Higth = 0 then image to crop size, or if no crop then keep original size
    # if if Higth > 0 and Width = 0, then resize width as per Hight by keeping same ratio
    size= [0,0]
    if img_resize[0] > 0:
        size[0] = int(2**(np.rint(np.log2(img_resize[0]))))
        if img_resize[1] > 0: size[1] = int(2**(np.rint(np.log2(img_resize[1]))))
        elif crop_size[0] == 0 or crop_size[1] == 0: size[1] = int(2**(np.rint(np.log2(int(( size[0]/img_size[0]) * img_size[1])))))
        else:  size[1] = int(2**(np.rint(np.log2(int((size[0]/crop_size[0]) * crop_size[1])))))
    elif crop_size[0] > 0 and crop_size[1] > 0: [int(2 ** (np.rint(np.log2(crop_size[0])))), int(2 ** (np.rint(np.log2(crop_size[1]))))]
    else: size = [int(2**(np.rint(np.log2(img_size[0])))), int(2**(np.rint(np.log2(img_size[1]))))]
    if crop_size[0] > 0 and crop_size[0] > 0:
    #if crop_size[0] > 0 and crop_size[1] > 0 and ((crop_origin[0] + crop_size[0]) < img_shape[0]) and ((crop_origin[1] + crop_size[1]) < img_shape[1]):
        img = tf.expand_dims(img, axis=0)
        x1 = crop_origin[0] / img_size[0]
        y1 = crop_origin[1] / img_size[1]
        x2 = (crop_origin[0] + crop_size[0]) / img_size[0]
        y2 = (crop_origin[1] + crop_size[1]) / img_size[1]
        img = tf.image.crop_and_resize(img, boxes=
        [[x1, y1, x2, y2]], crop_size=[size[0], size[1]], box_indices=[0])
        img = tf.squeeze(img, 0)
    elif size[0] > 0 and size[1] > 0 and  (size[0] != img_size[0] or size[1] != img_size[1]):
        img = tf.image.resize(img, [size[0], size[1]])

    # resize the image to the desired size.
    return img


#
# def decode_img(file_path, img_format= 0, img_size = [0, 0], crop_origin=[0, 0], crop_size= [150, 150], img_resize= [150,0], convert_power_of_2 = False):
#     """
#     :param file_path:
#     :param img_format: image format:  1 = 'jpg', 2 = 'bmp', 3 = 'png'
#     :param img_resize: resize in pixels (heigth, width). if heigth == 0, then size = crop. if heigth <> 0 and width == 0, then it keeps aspect
#     :param crop_origin: x1, y1 coordinates (x=vertical, y=horizontal)
#     :param crop_size: size in pixels. (heigth, width). if 0, then no crop is applied
#     :return: image
#     """
#
#     # convert the compressed string to a 3D uint8 tensor
#     img = tf.io.read_file(file_path)
#     if img_format == 1: img = tf.image.decode_jpeg(img, channels=3)
#     elif img_format== 2: img = tf.image.decode_bmp(img, channels=3)
#     elif img_format == 3: img = tf.image.decode_png(img, channels=3)
#     else: print('Image format not recognised, please use bmp, png or jpeg')
#
#     # Use `convert_image_dtype` to convert to floats in the [0,1] range.
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     # Re-size image as per img_resize input
#     # If Higth = 0 then image to crop size, or if no crop then keep original size
#     # if if Higth > 0 and Width = 0, then resize width as per Hight by keeping same ratio
#     size= [0,0]
#     if img_resize[0] > 0:
#         if convert_power_of_2: size[0] = int(2**(np.rint(np.log2(img_resize[0]))))
#         else: size[0] = img_resize[0]
#         if img_resize[1] > 0:
#             if convert_power_of_2: size[1] = int(2**(np.rint(np.log2(img_resize[1]))))
#             else: size[1] = img_resize[1]
#         elif crop_size[0] == 0 or crop_size[1] == 0: size[1] = int(2**(np.rint(np.log2(int(( size[0]/img_size[0]) * img_size[1])))))
#         else:  size[1] = int(2**(np.rint(np.log2(int((size[0]/crop_size[0]) * crop_size[1])))))
#     elif crop_size[0] > 0 and crop_size[1] > 0: [int(2 ** (np.rint(np.log2(crop_size[0])))), int(2 ** (np.rint(np.log2(crop_size[1]))))]
#     else: size = [int(2**(np.rint(np.log2(img_size[0])))), int(2**(np.rint(np.log2(img_size[1]))))]
#     if crop_size[0] > 0 and crop_size[0] > 0:
#         img = tf.expand_dims(img, axis=0)
#         x1 = crop_origin[0] / img_size[0]
#         y1 = crop_origin[1] / img_size[1]
#         x2 = (crop_origin[0] + crop_size[0]) / img_size[0]
#         y2 = (crop_origin[1] + crop_size[1]) / img_size[1]
#         img = tf.image.crop_and_resize(img, boxes=
#         [[x1, y1, x2, y2]], crop_size=[size[0], size[1]], box_indices=[0])
#         img = tf.squeeze(img, 0)
#     elif size[0] > 0 and size[1] > 0 and  (size[0] != img_size[0] or size[1] != img_size[1]):
#         img = tf.image.resize(img, [size[0], size[1]])
#
#     # resize the image to the desired size.
#     return img


def create_file_path_mask_dataset(path_to_images, path_to_notation, class_name_list, img_params = [0, [0, 0], [0, 0], [0, 0], [0, 0]]):
    # list of files in folder (with full path)
    list_of_image_paths = [os.path.join(path, name) for path, subdirs, files in os.walk(path_to_images) for name in files]
    np_image_names = np.array([os.path.split(fp)[1] for fp in list_of_image_paths])
    list_of_notation_paths = notation_list(path_to_images, path_to_notation, class_name_list)
    # list of masks delcaration
    list_of_masks = list()
    list_of_notation_files_indexer = list_of_notation_paths
    # create and crop a null mask for images with no deffects
    null_mask = np.zeros(img_params[1][0] * img_params[1][1] * len(class_name_list), dtype='uint8').reshape(
        img_params[1][0], img_params[1][1], len(class_name_list))
    null_mask = crop_nd_size(null_mask, img_size=img_params[1], crop_origin=img_params[2], crop_size=img_params[3],
                 img_resize=img_params[4])
    # initialise flags
    match = False
    # index through lists
    for image_path in list_of_image_paths:
        for notation_path in list_of_notation_files_indexer:
            # strip path to compare names.
            notation_name = os.path.splitext(os.path.basename(notation_path))[0]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            if image_name == notation_name:
                get_json_data_out = get_json_data(notation_path)
                # create mask from json file
                mask = regions_to_array(get_json_data_out['annotations_list'][0], class_name_list,
                                        tuple(get_json_data_out['image_shape']))
                #crop mask
                mask = crop_nd_size(mask, img_size=img_params[1], crop_origin=img_params[2],
                                         crop_size=img_params[3],
                                         img_resize=img_params[4])
                #append mask to list
                list_of_masks.append(mask)
                list_of_notation_files_indexer.remove(notation_path)
                match = True
                break
            else: continue
        # if no mask was found, add null mask
        if not match: list_of_masks.append(null_mask)
        match = False
    np_image_paths = np.asarray(list_of_image_paths)
    np_notation_masks = np.asarray(list_of_masks)

    return np_array_to_dataset((np_image_names, np_image_paths, np_notation_masks))



#########################################





def process_path_seg( ds_inputs, img_params, class_name_list):
    """
    Function to be applied to the dataset tensor, to create a dataset with images and masks
    :param ds_inputs: tensor with image name[0], image file paths [1] and masks file paths[2].
    :param img_params: img_format[1], img_size[x,y], crop_origin=[x, y], crop_size= [x, y], img_resize= [x, y]
    :param notation_path_list: list of strings with json files
    :param class_name_list: list of class to be included in dataset
    :return: image and mask
    """

    #########################
    # decode image and crop/resize
    #########################
    img = decode_img(ds_inputs[1],
                     img_format= img_params[0],
                     img_size=img_params[1],
                     crop_origin=img_params[2],
                     crop_size=img_params[3],
                     img_resize=img_params[4])

    return ds_inputs[0], img, ds_inputs[2]


def process_path(file_path, img_params, class_names):
    """
    TODO
    :param file_path:
    :param class_names:
    :param img_params:
    :return:
    """
    #img_format = 'bmp'
    #Should be an argument from call
    label = get_label_from_subfolders(file_path, class_names)
    # load the raw data from the file as a string
    file_name = tf.strings.split(file_path, '\\')[-1]
    img = decode_img(file_path,
                     img_format=img_params[0],
                     img_size= img_params[1],
                     crop_origin=img_params[2],
                     crop_size=img_params[3],
                     img_resize=img_params[4])
    return file_name, img, label


def convert_tensor_to_list_of_strings(tensor_str):
    list_str = []
    np_str = tensor_str.numpy()
    for item in np_str:
        list_str.append(str(item, 'utf-8'))
    return list_str

def convert_tensor_to_list_of_int(tensor_int):
    list_int = []
    np_int = tensor_int.numpy()
    for item in np_int:
        list_int.append(item)
    return list_int


def print_sample(ds, name, class_names, num_of_samples_per_class = 1):
    """
    TODO
    :param ds:
    :param name:
    :param class_names:
    :return:
    """
    dataset_size = get_dataset_size(ds)
    print("- %s dataset: %s images" %(name, dataset_size))
    for class_label in class_names:
        i = 0
        for file_name, image, label in ds.take(dataset_size):
            x = label.numpy()
            x = x[::-1]
            label_idex = int(''.join(map(lambda x: str(int(x)), x)), 2) - 1
            if class_names[label_idex] == class_label:
                plt.figure()
                plt.imshow(image)
                plt.title(name + " - " + class_label)
                plt.xlabel(file_name.numpy().decode('UTF-8'))
                plt.show(block=False)
                i += 1
                if i >= num_of_samples_per_class: break
            else: pass


def print_np_samples_with_masks(np_img_names, np_images, np_masks, name,  class_name_list = [], num_of_samples = 1):
    num_of_classes = len(class_name_list)
    dataset_size = np_images.shape[0]
    print("- %s dataset: %s images" % (name, dataset_size))
    for j in range(num_of_samples):
        plt.subplot(212)
        plt.title(np_img_names[j])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(np_images[j]))
        plt.axis('off')
        for i in range(num_of_classes):
            plt.subplot(2, num_of_classes, i + 1)
            plt.title('Dataset: ' + name + 'Class: ' + class_name_list[i])
            plt.imshow(np_masks[j,:, :, i], cmap='gray')
            plt.axis('off')
        plt.show(block=False)


def print_batch_shape(batch, name):
    """
    TODO
    :param batch:
    :param name:
    :return:
    """
    for image_batch, label_batch in batch.take(1):
        print("%s BATCH SHAPE:" %(name))
        print("  - Images: %s images, %s x %s pix, %s channels" % (
        image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], image_batch.shape[3]))
        print("  - Labels: %s labels, %s classes" % (label_batch.shape[0], label_batch.shape[1]))

def ds_to_single_np_array(ds):
    """
    Convert a dataset into 2 numpy arrays as follows:
    - images with shape (num_of_images, H, W, ch)
    - lables with shape (num_of_lables, labels). The label arrays are converted from boolean (True, False) to integer (1, 0)
    :param ds: dataset
    :return: numpy array: images, labels
    """
    dataset_size = get_dataset_size(ds)
    array = np.array([])
    ini = True
    for item in ds.take(dataset_size):
        if ini:
            array = np.expand_dims(item.numpy(), axis=0)
            ini = False
        else:
            array = np.append(array, np.expand_dims(item.numpy(), axis=0), axis=0)
    return array

def ds_to_np_array(ds, type = 'label' ):
    """
    Convert a dataset into 2 numpy arrays as follows:
    - images with shape (num_of_images, H, W, ch)
    - lables with shape (num_of_lables, labels). The label arrays are converted from boolean (True, False) to integer (1, 0)
    :param ds: dataset; type = 'label' or 'mask'
    :return: numpy array: images, labels
    """
    dataset_size = get_dataset_size(ds)
    images = np.array([])
    labels = np.array([])
    ini = True
    for item in ds.take(dataset_size):
        if ini:
            images = np.expand_dims(item[0].numpy(), axis=0)
            if type == 'label':
                labels = np.expand_dims(item[1].numpy().astype('int'), axis=0)
            elif type == 'mask':
                labels = np.expand_dims(item[1].numpy(), axis=0)
            ini = False
        else:
            images = np.append(images, np.expand_dims(item[0].numpy(), axis=0), axis=0)
            if type == 'label':
                labels = np.append(labels, np.expand_dims(item[1].numpy().astype('int'), axis=0), axis=0)
            elif type == 'mask':
                labels = np.append(labels, np.expand_dims(item[1].numpy(), axis=0), axis=0)
    return images, labels


def ds_with_names_to_np_array(ds, type = 'label' ):
    """
    Convert a dataset into 2 numpy arrays as follows:
    - images with shape (num_of_images, H, W, ch)
    - lables with shape (num_of_lables, labels). The label arrays are converted from boolean (True, False) to integer (1, 0)
    :param ds: dataset; type = 'label' or 'mask'
    :return: numpy array: images, labels
    """
    dataset_size = get_dataset_size(ds)
    names =  np.array([])
    images = np.array([])
    labels = np.array([])
    ini = True
    for item in ds.take(dataset_size):
        if ini:
            names = np.expand_dims(item[0].numpy().decode('UTF-8'), axis=0)
            images = np.expand_dims(item[1].numpy(), axis=0)
            if type == 'label':
                labels = np.expand_dims(item[2].numpy().astype('int'), axis=0)
            elif type == 'mask':
                labels = np.expand_dims(item[2].numpy(), axis=0)
            ini = False
        else:
            names = np.append(names, np.expand_dims(item[0].numpy().decode('UTF-8'), axis=0), axis=0)
            images = np.append(images, np.expand_dims(item[1].numpy(), axis=0), axis=0)
            if type == 'label':
                labels = np.append(labels, np.expand_dims(item[2].numpy().astype('int'), axis=0), axis=0)
            elif type == 'mask':
                labels = np.append(labels, np.expand_dims(item[2].numpy(), axis=0), axis=0)
    return names, images, labels

def ds_to_np_vector(ds):
    """
    Convert a dataset into 2 numpy vectors (1D arrays) as follows:
    - images
    - lables . The label arrays are converted from boolean (True, False) to integer (1, 0)
    :param ds: dataset
    :return: numpy array: images, labels
    """
    dataset_size = get_dataset_size(ds)
    images = np.array([])
    labels = np.array([])
    ini = True
    for item in ds.take(dataset_size):
        if ini:
            img_shape = item[0].numpy().shape
            lbl_shape = item[1].numpy().shape
            ini = False
        images = np.concatenate((images, item[0].numpy()), axis=None)
        labels = np.concatenate((labels, item[1].numpy().astype('int')), axis=None)
    return images, labels, dataset_size, img_shape, lbl_shape

def np_array_to_dataset(np_input):
    """
    TODO
    :param np_images:
    :param np_labels:
    :return:
    """
    ds = tf.data.Dataset.from_tensor_slices(np_input)
    return ds


def notation_list(path_to_images = '', path_to_json = '', list_of_deffects= []):
    defect_dict = dict_from_jsons(path_to_json, path_to_images, recursive=True)
    JSONList = defect_dict['JSONList']
    DefectDict = defect_dict['DefectDict'].items()

    inclusion_list = []
    for x, y in DefectDict:
        for deffect in list_of_deffects:
            if x == deffect:
                inclusion_list += y
    return [JSONList[index] for index in inclusion_list]



def create_filter_mask(searchlist = [], keywords = [], full_match = True):
    if len(searchlist) == 0 or len(keywords) == 0 : raise RuntimeError('Empty filter list provided')
    mask = np.zeros(len(searchlist), dtype=np.bool).tolist()
    if full_match:
        ids = [i for i, val in enumerate(searchlist) if val in keywords]
    else:
        ids = [i for i, val in enumerate(searchlist) if any(xs in val for xs in keywords)]
    for i in ids: mask[i] = True
    return mask


def create_dataset_with_split(images_array=np.array([]), labels_array=np.array([]), split_ratio=0.7,
                              seg_conv_type='Thresholding', seg_ThresholdNormZeroToOne=0.0):
    """
    Split dataset maintaining balance between classes and shuffling images.
    :param np_images:
    :param np_labels:
    :return:
    """

    # Detect whther labels are segmentation masks or binaray classes
    if labels_array.ndim == 2:  # Classifier labels
        lablels_converted_to_classes = labels_array
    else:  # segmentation masks
        lablels_converted_to_classes,_ = convert_seg_mask_to_classification_array(labels_array,
                                                                                conv_type=seg_conv_type,
                                                                                ThresholdNormZeroToOne=seg_ThresholdNormZeroToOne)
    # create datasets from numpy arrays
    mask = np.zeros(lablels_converted_to_classes.shape[0], dtype=np.int)
    for i in range(lablels_converted_to_classes.shape[-1]): mask = mask | lablels_converted_to_classes[:, i].astype(int)
    # Check if mask contains all labels, meaning the "no_defect_category" was included, and if so, leave first category out
    if np.sum(mask) >= lablels_converted_to_classes.shape[0]:
        mask = np.zeros(lablels_converted_to_classes.shape[0], dtype=np.int)
        for i in range(lablels_converted_to_classes.shape[-1]-1): mask = mask | lablels_converted_to_classes[:, i+1].astype(
            int)
    ds_TrainXVal_deffect = np_array_to_dataset((images_array[mask > 0, ...], labels_array[mask > 0, ...]))
    ds_training_deffect, ds_val_deffect = split_dataset(ds_TrainXVal_deffect, split_ratio=split_ratio, shuffle=True)

    ds_TrainXVal_no_deffect = np_array_to_dataset((images_array[mask == 0, ...], labels_array[mask == 0, ...]))
    ds_training_no_deffect, ds_val_no_deffect = split_dataset(ds_TrainXVal_no_deffect, split_ratio=split_ratio, shuffle=True)

    ds_training = ds_training_deffect.concatenate(ds_training_no_deffect)
    del ds_training_deffect, ds_training_no_deffect
    ds_val = ds_val_deffect.concatenate(ds_val_no_deffect)
    del ds_val_deffect, ds_val_no_deffect


    return ds_training.shuffle(buffer_size=get_dataset_size(ds_training), reshuffle_each_iteration= False), ds_val.shuffle(buffer_size=get_dataset_size(ds_val), reshuffle_each_iteration= False)


########################################################################
##### New

def get_dataset_info_from_folders(path_to_images, class_name_list=[], path_to_notation='', include_extension_in_name = True):
    """
    Get dataset information from a directory of folders.
    If the path_to_mask parameters is empty, it will assume the labels are based on sub-folders,
    one per class. If the path to maks is not empty, the function will look for individual json
    files within the mask folder. If either an image has not mask file or a mask file exists without
    an image file, then maks or image would be excluded from the final list.
    The function return:
    array_image_names: image names containing extension
    array_image_paths: Image full path
    array_notation: If segmentation mask,then it retures a path to the json file. if classification
    based on folder name, then it returns an array with 0, 1.

    :param path_to_images: superfolder with subfolders of images.
    :param class_name_list: list with class names
    :param path_to_notation: folder with notation jsons. leave empty for class notation
    :param include_extension_in_name: if false, the name array will exclude extension.
    """
    #

    # list of files in folder (with full path)


    if len(path_to_notation) > 0:
        Segmentation = True
        array_notation_paths = np.array(
            [os.path.join(path, name) for path, subdirs, files in os.walk(path_to_notation) for name in files])
        if os.path.splitext(array_notation_paths[0])[1].lower() == '.json': json_notation = True
        else: json_notation = False
        if len(class_name_list) == 0 and json_notation:
            raise RuntimeError('A path to notation was given but no Classes were listed. At least one class is needed to look for notation within the .json files')
    else:
        Segmentation = False
        json_notation = False

    # list of files in folder (with full path)
    array_image_paths = np.array(
        [os.path.join(path, name) for path, subdirs, files in os.walk(path_to_images) for name in files])

    if Segmentation and json_notation:
        # compile the list of notation paths that contain notation within the specified list of classes.
        # compile a class array based on notation within notation files. dimension [:,0] contain 'no defect' class
        list_of_notation_paths, array_classes = notation_list_nd_class_array(path_to_images, path_to_notation, class_name_list)

        # list of masks delcaration
        list_of_masks_paths = list()
        list_of_notation_files_indexer = list_of_notation_paths

        # initialise flags
        match = False
        # index through lists
        for image_path in array_image_paths:
            for notation_path in list_of_notation_files_indexer:
                # strip path to compare names.
                notation_name = os.path.splitext(os.path.basename(notation_path))[0]
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                if image_name == notation_name:

                    # append mask to list
                    list_of_masks_paths.append(notation_path)
                    list_of_notation_files_indexer.remove(notation_path)
                    match = True
                    break
                else:
                    continue
            # if no mask was found, add null mask
            if not match: list_of_masks_paths.append('')
            match = False
        array_notation_paths = np.asarray(list_of_masks_paths)
    else:
        if (Segmentation and json_notation) or not Segmentation: array_notation_paths = None

        # Extract class names from folder names
        class_name_list_from_folder = np.array(os.listdir(path_to_images))

        if len(class_name_list) > 0 and not any(i in class_name_list_from_folder for i in class_name_list):
            raise RuntimeError(
                'The provided class list does not match folder names. Folder names: ' + class_name_list_from_folder)
        else:
            if len(class_name_list) == 0:
                class_name_list = class_name_list_from_folder

            num_of_clasess = len(class_name_list)
            array_image_paths = np.empty(0)
            array_classes = np.empty((0, num_of_clasess), int)
            class_num = 0
            for class_name in class_name_list:
                files_in_class = np.array(
                    [os.path.join(path, name) for path, subdirs, files in
                     os.walk(os.path.join(path_to_images, class_name)) for name in files])
                array_image_paths = np.append(array_image_paths, files_in_class)
                class_binary_array = np.zeros((files_in_class.shape[0], num_of_clasess), int)
                class_binary_array[:, class_num] = 1

                array_classes = np.append(array_classes, class_binary_array, axis=0)
                class_num += 1
        # filter notation files if Segmentation and notation files are mask images.
        if Segmentation:
            list_notation_paths = list()
            for img_path in array_image_paths:
                match = False
                if array_notation_paths.shape[0] > 0:
                    for notation_path in array_notation_paths:
                        if os.path.splitext(os.path.basename(img_path))[0] == os.path.splitext(os.path.basename(notation_path))[0]:
                            list_notation_paths.append(notation_path)
                            array_notation_paths = np.array([s for s in array_notation_paths if s != notation_path])
                            match = True
                            break
                if not match: list_notation_paths.append('')
            array_notation_paths = np.array(list_notation_paths)

    if include_extension_in_name:
        array_image_names = np.array([os.path.split(fp)[1] for fp in array_image_paths])
    else:
        array_image_names = np.array([os.path.splitext(os.path.basename(fp))[0] for fp in array_image_paths])



    return array_image_names, array_image_paths, array_classes, array_notation_paths


def notation_list_nd_class_array(path_to_images = [], path_to_notation = [], class_name_list= []):
    """
    Returns a list of paths of json files corresponding to images withing the image paths and classes within
    the class list.
    it will also return a class array to perform classification where the first column correspond to the "no defect" class.
    :param path_to_images:
    :param path_to_notation:
    :param class_name_list:
    :return:
    """
    # Check if notation files are masks or json files
    array_image_paths = np.array(
        [os.path.join(path, name) for path, subdirs, files in os.walk(path_to_images) for name in files])


    # returns 'JSONList': list of paths with valid .json files and
    # 'DefectDict' a dictionary with all the defects within the json files and a list of indexes per defect
    # of json files containing the defect.
    defect_dict = dict_from_jsons(path_to_notation, path_to_images, recursive=True)
    JSONList = defect_dict['JSONList']
    DefectDict = defect_dict['DefectDict'].items()
    # Create classification notation array where class 0 is "defect free" or "good".
    class_array = np.zeros([len(array_image_paths), len(class_name_list)+1], dtype=bool)
    summary_list = []
    for x, y in DefectDict:
        Cls_idx = 1
        for deffect in class_name_list:
            if x == deffect:
                summary_list += y
                class_array[y, Cls_idx] = True
            Cls_idx +=1
    # Class 0 identifies not defect.
    class_array[:,0] = np.logical_not(np.any(class_array[:, 1:], axis=1))
    #check if there is at least one image labled as good and if not, then delet the summary dimmenssion
    if not np.any(class_array[:, 0]):
        class_array = class_array[:, 1:]
    return [JSONList[index] for index in summary_list], class_array



def get_dataset_from_numpy_file(ds_path = '', ds_split_name = '', key_filter = ['ids', 'images', 'masks', 'labels']):
    """
    reads the ds_info.json withthe dataset info. and load data based on keys.
    :param ds_path: Full pat to folder that contains both the ds_info.json file and the .npz file
    :param ds_split_name: name of the dataset split matching the info. in the json file and the .npz file name.
    :param key_filter: list with the keys to load.
    :return: returns the data within the .npz file class_name_list, np_image_names_array, np_image_array, np_classes_array, np_mask_array
    """
    # read .json file split info.
    ds_json_file = os.path.join(ds_path, 'ds_info.json')
    if os.path.isfile(ds_json_file):
        ds_info_dict = read_from_json(ds_json_file)
        splits = [s for s in ds_info_dict.keys() if 'split' in s]
    else:
        raise RuntimeError('The specified folder: "' + ds_path + '" must contain the dataset info file "ds_info.json"')

    split_exists = False
    for split in splits:
        if ds_info_dict[split]['split_name'] == ds_split_name:
            split_info = ds_info_dict[split]
            split_exists = True
    if not split_exists: raise RuntimeError(
        '"ds_info.json" does not contain information for dataset split "' + ds_split_name + '"')


    ds_name = os.path.join(ds_path, (ds_split_name + '.npz'))
    if not os.path.isfile(ds_name):
        raise RuntimeError('Dataset split "' + ds_split_name + '" does not exist in folder "' + ds_path + '"')

    # ini variables
    np_image_names_array = None
    np_image_array = None
    np_mask_array = None
    np_classes_array = None

    # outputs
    class_name_list = split_info['class_names']

    with np.load(ds_name, mmap_mode = 'r') as data:
        if 'ids' in split_info['dataset_keys'] and 'ids' in key_filter: np_image_names_array = data['ids']
        if 'images' in split_info['dataset_keys'] and 'images' in key_filter: np_image_array = data['images']
        if 'masks' in split_info['dataset_keys'] and 'masks' in key_filter: np_mask_array = data['masks']
        if 'labels' in split_info['dataset_keys'] and 'labels' in key_filter: np_classes_array = data['labels']

    return class_name_list, np_image_names_array, np_image_array, np_classes_array, np_mask_array



def get_info_from_dataset_json(DATASET_INFO_NAME_PATH, IMG_SPLIT_SUBFOLDER):
    info_dict = {}
    root = os.path.join(os.getcwd())
    if os.path.split(root)[-1] == 'custom_lib':
        root = os.path.split(root)[0]
    dataset_info_dict = read_from_json(os.path.join(root, 'datasets', DATASET_INFO_NAME_PATH, 'ds_info.json'))
    # get preProcessing function

    if 'img_params' in dataset_info_dict.keys():
        info_dict['img_params'] = dictionary_to_array(dataset_info_dict['img_params'])
    else:
        info_dict['img_params'] = None
    # get img augment params
    if 'img_augmentation' in dataset_info_dict.keys():
        info_dict['img_augmentation'] = dataset_info_dict['img_augmentation']
    else:
        info_dict['img_augmentation'] = None
    # get dataframe.ccv path
    info_dict['dataframe_path'] = os.path.join(root, 'datasets', DATASET_INFO_NAME_PATH, IMG_SPLIT_SUBFOLDER + '.csv')
    if not os.path.isfile(info_dict['dataframe_path']): raise RuntimeError(
        'Dataset split file: "' + info_dict['dataframe_path'] + '" does not exist in folder')
    # get img dataset folder
    info_dict['dataset_folder'] = os.path.join(dataset_info_dict['root_folder'], IMG_SPLIT_SUBFOLDER)
    # get mask folder
    if 'mask_folder' in dataset_info_dict.keys():
        info_dict['notation_folder'] = os.path.join(dataset_info_dict['root_folder'], dataset_info_dict['mask_folder'])
    else:
        info_dict['notation_folder'] = None
    # Get dataset split info.
    split_info = None
    for split_name in [key for key in dataset_info_dict.keys() if 'split' in key]:
        if dataset_info_dict[split_name]['split_name'] == IMG_SPLIT_SUBFOLDER:
            split_info = dataset_info_dict[split_name]
            break
    if split_info is None: raise RuntimeError(
        'Dataset split name: "' + IMG_SPLIT_SUBFOLDER + '" does not exist in ds_info.json')
    #get class names
    info_dict['class_names'] = split_info['class_names']
    # get dataframe column names.
    info_dict['img_path_column'] = split_info['dataframe_columns'][1]
    if len(split_info['dataframe_columns']) < 4:
        info_dict['notation_path_column'] = ''
        info_dict['segmentation_mask'] = False
        info_dict['class_column'] = split_info['dataframe_columns'][2]
    else:
        info_dict['notation_path_column'] = split_info['dataframe_columns'][2]
        info_dict['segmentation_mask'] = True
        info_dict['class_column'] = split_info['dataframe_columns'][3]
    if info_dict['notation_folder'] is None:
        info_dict['notation_path_column'] = ''
        info_dict['segmentation_mask'] = False
    info_dict['split_info'] = split_info

    return info_dict





