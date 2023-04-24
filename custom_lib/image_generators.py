import pandas
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import *
import numpy as np
import pandas as pd
from bscai.annotation.load import regions_to_array, get_json_data
import os
from pathlib import Path
from custom_lib.image_functions import read_img_from_folder, read_mask_from_folder, read_img_from_folder_w_processing, read_mask_from_folder_w_processing
from custom_lib.image_functions import GetImageProcessingFuncs
from custom_lib.data_log_functions import convert_cls_str_to_bool
import math

#Todo: Generate datasets in script 02b with generators (apply batch stats and record stats in json to be used in other sets
#Todo: Compensate for ubalanced datasets by spliting the dataset in classess first and applying augmentation to create a
#balanced dataset. This would require to generate batches as per below example instead of passing the generator to the fit function.
#Todo: add custome function place holder for augmentation. This may require a different approach to the keras generator.

"""ds = tf.data.Dataset.from_generator(
    lambda: gen,
    output_types=(covert_to_tf_type(sample_image.dtype), covert_to_tf_type(sample_label.dtype)),
    output_shapes=([batch_size,sample_image.shape[1],sample_image.shape[2],sample_image.shape[3]], [batch_size,sample_label.shape[1]]))

ds = ds.repeat(epochs)
ds = ds.batch(batch_size)
iterator = tf.compat.v1.data.make_one_shot_iterator(ds)

batch_img,batch_lbl = iterator.get_next()

or 
"""


def covert_to_tf_type(python_type=''):
    """
    Convert to tensorflow type
    :param python_type: string with python type
    :return: tf.datatype
    """
    if python_type == 'int8':
        type = tf.int8
    elif python_type == 'int16':
        type = tf.int16
    elif python_type == 'int32':
        type = tf.int32
    elif python_type == 'int64':
        type = tf.int64
    elif python_type == 'float16':
        type = tf.float16
    elif python_type == 'float32':
        type = tf.float32
    elif python_type == 'float64':
        type = tf.float64
    return type


def img_augmentation_gen(images_array, labels_array, mask_class = False, transfomation_dict: dict = None, batch_size = 1 ):

    #instanciate generator class
    datagen = ImageDataGenerator(**transfomation_dict)
    if mask_class:
        transfomation_dict_mask = transfomation_dict.copy()
        del transfomation_dict_mask['brightness_range']
        del transfomation_dict_mask['rescale']
        datagen_mask = ImageDataGenerator(**transfomation_dict_mask)

    # seed to ensure mask and images are sync
    seed = 1
    # fit generator to image array. This will apply batch stats if required and seed the generator.
    datagen.fit(images_array, augment=True, seed=seed)
    if mask_class:
        datagen_mask.fit(labels_array, augment=True, seed=seed)

    # construct the iterator
    if mask_class:
        gen_img = datagen.flow(images_array, batch_size=batch_size, seed=seed)
        gen_mask = datagen_mask.flow(labels_array, batch_size=batch_size, seed=seed)
        gen = zip(gen_img, gen_mask)
    else:
        gen = datagen.flow(images_array, labels_array, batch_size=batch_size, seed=seed)

    return gen


def img_gen(images_array, labels_array, mask_class = False, transfomation_dict: dict = None, batch_size = 1 ):

    #instanciate generator class
    datagen = ImageDataGenerator(**transfomation_dict)
    if mask_class:
        transfomation_dict_mask = transfomation_dict.copy()
        del transfomation_dict_mask['brightness_range']
        del transfomation_dict_mask['rescale']
        datagen_mask = ImageDataGenerator(**transfomation_dict_mask)

    # seed to ensure mask and images are sync
    seed = 1
    # fit generator to image array. This will apply batch stats if required and seed the generator.
    datagen.fit(images_array, augment=True, seed=seed)
    if mask_class:
        datagen_mask.fit(labels_array, augment=True, seed=seed)

    # construct the iterator
    if mask_class:
        gen_img = datagen.flow(images_array, batch_size=batch_size, seed=seed)
        gen_mask = datagen_mask.flow(labels_array, batch_size=batch_size, seed=seed)
        gen = zip(gen_img, gen_mask)
    else:
        gen = datagen.flow(images_array, labels_array, batch_size=batch_size, seed=seed)

    return gen


#img = tf.image.resize(img, [int(2 ** (np.rint(np.log2(img_resize[0])))), int(2 ** (np.rint(np.log2(img_resize[1]))))])
def resize_func(img, img_resize):
    size = [int(2 ** (np.rint(np.log2(img_resize[0])))), int(2 ** (np.rint(np.log2(img_resize[1]))))]
    return tf.image.resize(**img)

####################################################################################################################
############################## NEW #################################################################################


def read_files_from_folder_gen(file_paths, read_file_func, batch_size = 1, img_params =  [[0,0],[0,0],[0,0],[0,0]], class_name_list = [], return_partial_batch = False):
    """
    Iterator to read images from folders and return a batch of tensor image per iteration.
    It uses the function read_img_from_folder to read each image into a tensor.
    :param image_paths: array with full path to image files, including file name and extension
    :param read_file_func: function to read the files (e.g. images, masks)
    :param batch_size: number of tensors to be return at each iteration. Carful .. if to large, it would run out of GPU memory. Try 10 to 50.
    :param img_params: Image parameters as per dictionary.
    :param return_partial_batch: If last batch is incomplete and param is set False, then images will not be yield
    """
    if batch_size == 0: batch_size = len(file_paths)
    num_of_batch_iterations = len(file_paths)//batch_size
    iter = 0
    iter_batch = 0
    for file_path in file_paths:
        file = read_file_func(file_path,
                             img_size= img_params[0],
                             crop_origin=img_params[1],
                             crop_size=img_params[2],
                             img_resize=img_params[3],
                             class_name_list= class_name_list)

        iter += 1
        if iter == 1:
            file_tensor = tf.expand_dims(file, axis=0)
        else:
            file_tensor = tf.concat([file_tensor, tf.expand_dims(file, axis=0)], axis=0)

        if iter >= batch_size:
            iter = 0
            iter_batch +=1
            yield file_tensor

    if return_partial_batch and iter > 0: yield file_tensor

def read_files_from_folder_to_numpy(file_paths, read_file_func, batch_size = 1, img_params = [[0,0],[0,0],[0,0],[0,0]], class_name_list = []):
    """
    Call iterator function  read_files_from_folder_gen and returns a numpy array with all the file content in the paths array.
    based on the file reading function.
    :param image_paths: Array with image paths
    :param read_file_func: function to read the files (e.g. images, masks)
    :param batch_size: number of images to store in Tensors at each iteration of the generator. Careful, if too large,
            GPU could run out of memory. Try 10 to 50.
    :param img_params: array with image information as per dictionary
    :return: numpy array with all the file content in the paths array.
    """
    read_file_gen = read_files_from_folder_gen(file_paths, read_file_func, batch_size=batch_size, img_params=img_params,
                                               class_name_list = class_name_list, return_partial_batch=True)
    end_of_file_array = False
    first_loop=True
    while not end_of_file_array:
        try:
            tf_array = next(read_file_gen)
            if first_loop:
                np_file_array = tf_array.numpy()
            else:
                np_file_array = np.append(np_file_array, tf_array.numpy(), axis=0)
            first_loop = False
        except StopIteration:
            end_of_file_array = True
    return np_file_array



def img_augment_gen(np_image_array, np_classes_array, np_mask_array, transfomation_dict: dict = None, batch_size = 1, class_filter = []):

    #Check class filter
    if len(class_filter) > 0:
        for i in class_filter:
            image_array = np_image_array[np_classes_array[:,i], ...]
            if np_mask_array is not None: mask_array = np_mask_array[np_classes_array[:, i], ...]


    #check if mask or lable
    if np_mask_array is not None:
        Seg = True
    else:
        Seg = False

    #instanciate generator class
    datagen = ImageDataGenerator(**transfomation_dict)
    if Seg:
        transfomation_dict_mask = transfomation_dict.copy()
        del transfomation_dict_mask['brightness_range']
        del transfomation_dict_mask['rescale']
        datagen_mask = ImageDataGenerator(**transfomation_dict_mask)


    # set a seed to ensure mask and images are sync
    seed = 1
    # Fitting generator if 'samplewise_center' or 'samplewise_std_normalization'
    if transfomation_dict_mask['samplewise_center'] or transfomation_dict_mask['samplewise_std_normalization']:
        datagen.fit(image_array, augment=True, seed=seed)
        if Seg:
            datagen_mask.fit(mask_array, augment=True, seed=seed)

    # construct the iterator
    if Seg:
        gen_img = datagen.flow(image_array, batch_size=batch_size, seed=seed)
        gen_mask = datagen_mask.flow(mask_array, batch_size=batch_size, seed=seed)
        gen = zip(gen_img, gen_mask)
    else:
        gen = datagen.flow(image_array, batch_size=batch_size, seed=seed)

    return gen



##############################################################################################################################

### Img Data Generator

class ImgDatasetGenerator(Sequence):
    """

    """
    def __init__(self, PreProcessingImgParams = None, Augmentation = None,  dataframe_path = '', dataset_folder = '', notation_folder = '', img_path_column = 'image_paths',
                 notation_path_column = '', class_name_list = [], class_column = 'classes', class_filter = [], exclude_imgs_in_class = [], batch_size = 16, val_split = 0,
                 val_split_idx = None , dim=(224,224,3), shuffle_dataset = False, random_seed_initialisation = False):
        """
        :param PreProcessFunction  (refer to custom_lib.image_functions.GetImageProcessingFunc)
        :param Augmentation (refer to custom_lib.image_functions.GetImageProcessingFunc)
        :param dataframe_path: Pandas dataframe with columns for image path to folder, classification array and notation path to folder (if mask exists)
        :param dataset_folder: Root folder where all other paths are based on.
        :param img_path_column: column name in dataframe for image paths
        :param notation_path_column: columns name in dataframe for notation paths (if they exists)
        :param class_column: column name in dataframe for class arrays.
        :param class_filter: Specify index of classes to keep in the class array and notation masks. Any other classes will be removed from class array,
         however, images from other classes will be retained for training.
        :param exclude_imgs_in_class: specify index of classes to exclude images from. All images that are True to these classes will not be added to the batches.
        classes will be removed from class array.
        :param batch_size: number of images returned at each itter.
        :param val_split: percentage val split [0 to 1] where 0.7 equals 70% for training and 30% for Validation
        :param val_split_idx: If a val_split greater than 0 was passed to the Training generator, the __init__ of this instance will randomly select
        idx to be used within the instance and return the remaining indx to be used in the validation instance to self.val_idxs.
        Then the array in Training_Instance.val_idxs can be passed to Val_instance.val_split_idx so that the validation instance only uses the val idx.
        Validation split is random but respect class balances so that both splits, training and val has the same class balance.
        :param dim:
        :param shuffle_dataset: Shuffle dataset to create random batches between epochs. This would retain self.idx_batch_img_names for printing and reviewing.
        :param augment_batch: It false, augmentation is applied to images as the are read from directory. If True, augmentation is applied to the batch, so the full batch is stored in memmory.
        :param  random_seed_initialisation: If True, the random seed generator will be initialised with a random value.
        """
        'input error checking'
        #if :
        #    raise RuntimeError('')


        'initialisation'
        self.idx_batch_img_names = []
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle_dataset = shuffle_dataset
        self.class_name_list = class_name_list
        #'read dataframe, filter and save stats'
        self.df = pd.read_csv(dataframe_path)
        #'transform to a boolean array'
        np_class_array = self.df[class_column].to_numpy()
        class_array = convert_cls_str_to_bool(np_class_array)
        self.df_class_array = class_array
        self.PreProcessingImgParams = PreProcessingImgParams
        #initialise index array with all elements in order.
        idx_array = np.arange(class_array.shape[0]).astype('int')
        #'filter by classes - remove images'
        #initiaise exclusion list
        exclusion_list = np.empty(shape=0, dtype='int')

        #exclude images by class-
        if len(exclude_imgs_in_class) > 0 and val_split_idx is None:
            for i in exclude_imgs_in_class:
                exclusion_list = np.append(exclusion_list, np.argwhere(class_array[:,i] == True).astype('int'))
            exclusion_list = np.unique(exclusion_list)
            idx_array = np.setdiff1d(idx_array, exclusion_list)

        #filter classes array
        if len(class_filter) > 0: # and val_split_idx is None:
            exclude_imgs_in_class = np.unique(np.append(np.setdiff1d(np.arange(class_array.shape[1]).astype('int'), class_filter), exclude_imgs_in_class))

        self.class_array = class_array[:, np.setdiff1d(np.arange(class_array.shape[1]).astype('int'), exclude_imgs_in_class)]
       #validation instance takes inputs.
        if val_split_idx is not None:
            idx_array = val_split_idx
        idx_array = np.sort(np.unique(idx_array))

        #validation split in the testig Generator
        if val_split > 0 and val_split_idx is None:
            #num_of_img = int(np.floor(len(self.img_path_array) * (1-val_split)))
            # get num of samples per class after applying split % so that split would have the same proportion of classess.
            num_samples_per_class = [int(np.floor((val_split) * np.sum(self.class_array[idx_array, i]))) for i in range(self.class_array.shape[1])]
            idx_array_split = np.empty(0)
            idx_array_split_Neg = np.empty(0)
            # per class column, select a random num_samples_per_class and add it to the buffer.
            for i in range(self.class_array.shape[1]):
                cls_idx_array = np.where(self.class_array[:, i])[0].astype(int)
                cls_idx_array = np.intersect1d(idx_array, cls_idx_array).astype(int)
                # Random Selection of idx for the first split (Testing)
                random_selection1 = cls_idx_array[np.random.choice(len(cls_idx_array)-1, num_samples_per_class[i], replace=False)]
                idx_array_split = np.append(idx_array_split, random_selection1, axis=0).astype(int)
                # Non selected idx for the Val split
                random_selection2 = np.setdiff1d(cls_idx_array, random_selection1).astype(int)
                idx_array_split_Neg = np.append(idx_array_split_Neg, random_selection2, axis=0).astype(int)

            # check if there are idx that did not belong to any class.
            classless_idxs =  np.setdiff1d(idx_array, np.append(idx_array_split, idx_array_split_Neg, axis=0)).astype(int)
            if classless_idxs.shape[0] > 0:
                random_selection = classless_idxs[np.random.choice(classless_idxs.shape[0] - 1, int(np.floor((val_split) * classless_idxs.shape[0])), replace=False)]
                idx_array_split = np.append(idx_array_split, random_selection, axis=0)
                random_selection = np.setdiff1d(classless_idxs, random_selection)
                idx_array_split_Neg = np.append(idx_array_split_Neg, random_selection, axis=0)

            idx_array = np.sort(np.unique(idx_array_split)).astype(int)
            self.val_idxs = np.sort(np.unique(idx_array_split_Neg)).astype(int)
        else:
            self.val_idxs = np.empty(0)

        self.class_array = self.class_array[idx_array]
        self.idx_array = idx_array

        #'read path arrays'
        join_path_func = lambda x: (os.path.join(dataset_folder, str(x))) if len(str(x)) > 3 else ''

        self.df_img_path_array = np.array([join_path_func(x) for x in self.df[img_path_column].to_numpy()])

        if len(notation_path_column) == 0:
            self.ground_truth = 'label'
            self.notation_path_array = []
            self.df_notation_path_array = None
        else:
            self.ground_truth = 'mask'
            join_path_func = lambda x: (os.path.join(notation_folder, str(x))) if len(str(x)) > 3 else ''
            #self.notation_path_array = np.array([join_path_func(x) for x in self.df[notation_path_column].to_numpy()])[idx_array]
            self.df_notation_path_array = np.array([join_path_func(x) for x in self.df[notation_path_column].to_numpy()])


        # Get image and mask processing functions (PreProcessing will scale and crop and PostProcessing will augment and resize)
        self.ImgPreProcessingFunc, self.ImgPostProcessingFunc = GetImageProcessingFuncs(ImgCropnResize=PreProcessingImgParams, Augmentation=Augmentation, scale=1. / 255)
        # define seed generator

        if random_seed_initialisation:
            initialisation = np.random.randint(1, high=9999, size=None, dtype=int)
        else:
            initialisation = 123
        self.seed_gen = tf.random.Generator.from_seed(initialisation, alg='philox')


        #TODO: Consider adding code to fit batch stats for Img generator.

        self.dataset_qty = len(self.idx_array)
        if self.batch_size > self.dataset_qty: raise RuntimeError('batch_size [' + str(self.batch_size) + '] must be less than the qty of available images in the generator [' + str(self.dataset_qty) + ']')
        self.on_epoch_end()

    def __len__(self):
        ':returns the number of batches per epoch'
        return int(math.ceil(len(self.img_path_array) / self.batch_size))

    def read_first_img(self):
        img = read_img_from_folder(self.img_path_array[0], self.PreProcessFunction)
        return img


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # seed generators
        self.seed = index + 1
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.img_path_array))
        self.indexes = self.idx_array
        if self.shuffle_dataset:
            np.random.shuffle(self.indexes)
        self.class_array = self.df_class_array[self.indexes]
        if self.df_notation_path_array is not None:
            self.notation_path_array = self.df_notation_path_array[self.indexes]
        self.img_path_array = self.df_img_path_array[self.indexes]

    def __data_generation(self, list_IDs_temp):
        'Generates a batch of samples'
        # inititialisation
        iter = 0
        self.idx_batch_img_names = [os.path.splitext(os.path.split(path)[1])[0] for path in self.df_img_path_array[list_IDs_temp]]
        # Read batch IDx
        for idx in list_IDs_temp:
            # generate seed from generator
            seed = self.seed_gen.make_seeds(2)[0]
            # read image file
            #img = read_img_from_folder(self.df_img_path_array[idx], self.PreProcessFunction)

            # Read image, scale/crop and apply augmentation
            img = read_img_from_folder_w_processing(file_path=self.df_img_path_array[idx], ImgPreProcessingFunc=self.ImgPreProcessingFunc, ImgPostProcessingFunc=self.ImgPostProcessingFunc, seed=seed)
            # read maks file
            if self.ground_truth == 'mask':
                # mask = read_mask_from_folder(self.df_notation_path_array[idx], self.class_name_list, self.PreProcessFunction, img.shape[:-1].as_list())
                mask = read_mask_from_folder_w_processing(file_path=self.df_notation_path_array[idx], ImgPreProcessingFunc=self.ImgPreProcessingFunc,ImgPostProcessingFunc=self.ImgPostProcessingFunc, seed=seed,
                                                          class_name_list=self.class_name_list,  mask_size=self.PreProcessingImgParams[0])

            else:
                mask = None

            # Agregate arrays
            iter += 1
            if iter == 1:
                batch_of_imgs = tf.expand_dims(img, axis=0)
                if self.ground_truth == 'mask':
                    batch_of_labels = tf.expand_dims(mask, axis=0)
            else:
                batch_of_imgs = tf.concat([batch_of_imgs, tf.expand_dims(img, axis=0)], axis=0)
                if self.ground_truth == 'mask':
                    batch_of_labels = tf.concat([batch_of_labels, tf.expand_dims(mask, axis=0)], axis=0)
            # if ground truth are labels, then slice the class array.
        if self.ground_truth == 'label':
            batch_of_labels = self.df_class_array[list_IDs_temp, :]




       #TODO: Both Pre and Post processing are applied to single images when reading from folder.
        # Investigate applying both Pre and Post or only Post to a batch instead by using
        # either tf.map_fn or tf.vectorized_fn. At the moment this does not work as:
        # I cannot pass multiple inputs to the map functions and even if we could, we want to use
        # a different seed per image and no per batch.


        # Apply PostProcessFunction to augment batch of images

        # if self.ImgPostProcessingFunc is not None and self.augment_batch:
        #
        #     if self.ground_truth == 'mask':
        #         tf.vectorized_map()
        #         gen_img = self.ImgPostProcessingFunc.flow(batch_of_imgs,  batch_size=self.batch_size, seed=self.seed)
        #         batch_of_imgs = next(gen_img)
        #         gen_mask = self.ImgPostProcessingFunc.flow(batch_of_labels, batch_size=self.batch_size, seed=self.seed)
        #         batch_of_labels = next(gen_mask)
        #         batch_of_imgs = tf.convert_to_tensor(batch_of_imgs, dtype='float32')
        #         batch_of_labels = tf.convert_to_tensor(batch_of_labels, dtype='float32')
        #     else:
        #         gen_img = self.ImgPostProcessingFunc.flow(batch_of_imgs, batch_size=self.batch_size, seed=self.seed)
        #         batch_of_imgs = next(gen_img)
        #         batch_of_imgs = tf.convert_to_tensor(batch_of_imgs, dtype='float32')
        #
        # if self.func_resize is not None:
        #     batch_of_imgs = tf.map_fn(fn =self.func_resize, elems = batch_of_imgs)
        #     if self.ground_truth == 'mask': batch_of_labels = tf.map_fn(fn =self.func_resize, elems = batch_of_labels)


        return batch_of_imgs, batch_of_labels




def numpy_dataset_from_generator(DataGen:ImgDatasetGenerator):

    '''
    Returns a numpy dataset from a datagen that generate batches of
    tensors.
    :param DataGen:
    :return:
    '''
    #Initialise empty numpy arrays with the shape of x and y tensors
    batch_x, batch_y = DataGen.__getitem__(0)
    l = []
    for dim in batch_x.shape.dims:
        l.append(dim.value)
    l[0] = 0
    np_dataset_x = np.empty(l).astype('float32')
    l = []

    if hasattr(batch_y.shape, "dims"):
        lbl_is_mask = True # Lable is mask
        for dim in batch_y.shape.dims:
            l.append(dim.value)

    else:
        lbl_is_mask = False  # Lable is mask
        for dim in batch_y.shape:
            l.append(dim)
    l[0] = 0
    np_dataset_y = np.empty(l).astype('float32')
    img_names = []

    #iterate through the generator and concatenate
    for batch_x, batch_y in DataGen:
        np_dataset_x = np.concatenate([np_dataset_x, batch_x.numpy()], axis=0)
        if lbl_is_mask: np_dataset_y = np.concatenate([np_dataset_y, batch_y.numpy()], axis=0)
        else: np_dataset_y = np.concatenate([np_dataset_y, batch_y], axis=0)
        img_names += DataGen.idx_batch_img_names
    return np_dataset_x, np_dataset_y, np.array(img_names)