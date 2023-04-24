import pydoc

import keyboard
from custom_lib.image_functions import read_img_from_folder
from custom_lib.json_function import read_from_json
from time import time
import matplotlib.pyplot as plt
import os
import numpy as np
from custom_lib.image_functions import read_mask_from_folder
import cv2
import gc
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.colors as mcolors


def print_indexer_from_folder(image_path_array = [], PreProcessFunction = None, next_key = 'enter', abort_key = '', message = 'Press Enter to print a new image or any other key to abort..', delay = 0.5):
    start_time = time()
    timer = 0
    while timer < delay:
        timer = time() - start_time
    print(message)
    key = keyboard.read_key()
    for img_path in image_path_array:
        if abort_key != '' and key == abort_key or abort_key == '' and key != next_key: break
        img = read_img_from_folder(img_path, PreProcessFunction)
        plt.imshow(img)
        plt.title(os.path.splitext(os.path.split(img_path)[1])[0])
        plt.show()
        listening = True
        while listening:
            start_time = time()
            timer = 0
            while timer < delay:
                timer = time() - start_time
            key = keyboard.read_key()
            if key != next_key and (abort_key != '' and key != abort_key): listening = True
            else: listening = False
    print('************ All images have been printed **************')

def print_stacked_img_n_mask(img, label, title = 'Image with no title', hide_axis=True):
# print image and mask stacked
    if len(label.shape) < 3: # label is not a mask.
        plt.figure()
        plt.imshow(img)
        if hide_axis: plt.axis('off')
        plt.title(title)
        plt.show()
        plt.close()
    else: # label is a mask
        plt.figure()
        plt.subplot(212)
        plt.imshow(img)
        if hide_axis: plt.axis('off')
        for z in range(label.shape[-1]):
            plt.subplot(2, label.shape[-1], z + 1)
            plt.imshow(label[..., z], cmap='gray')
            if hide_axis: plt.axis('off')
        plt.title(title)
        plt.show()

def print_indexer_from_array(np_image_array = [], np_label_array = [], np_title_array = [], next_key = 'enter', abort_key = '', message = 'Press Enter to print a new image or any other key to abort..', delay = 0.5, hide_axis = True):
    start_time = time()
    timer = 0
    while timer < delay:
        timer = time() - start_time
    print(message)
    key = keyboard.read_key()
    for i in range(np_image_array.shape[0]):
        if abort_key != '' and key == abort_key or abort_key == '' and key != next_key: break
        print_stacked_img_n_mask(np_image_array[i, ...], np_label_array[i, ...], np_title_array[i], hide_axis = hide_axis)
        listening = True
        while listening:
            start_time = time()
            timer = 0
            while timer < delay:
                timer = time() - start_time
            key = keyboard.read_key()
            if key != next_key and (abort_key != '' and key != abort_key): listening = True
            else: listening = False
    print('************ All images have been printed **************')


def convert_json_to_png(json_path_folder = '', png_path_folder = '', class_name_list = []):
    # list files in folder
    json_path_list = [os.path.join(path, name) for path, subdirs, files in
         os.walk(json_path_folder) for name in files  if '.json' in name]
    if len(png_path_folder) == 0: png_path_folder = json_path_folder
    qty_total = len(json_path_list)
    # check if there are already png files in case the conversion was aborte.
    existing_png_path_list = [os.path.join(path, name) for path, subdirs, files in
                      os.walk(png_path_folder) for name in files if '.png' in name]
    existing_json_png_path_list = [p.replace('.png', '.json') for p in existing_png_path_list]
    existing_json_png_path_list = [p.replace(png_path_folder, json_path_folder) for p in existing_json_png_path_list]
    #filter json files that already have a png conversion
    json_path_list = [p for p in json_path_list if p not in existing_json_png_path_list]

    if png_path_folder == json_path_folder:
        png_path_list = json_path_list
    else:
        if os.path.exists(png_path_folder):
            png_path_list = [p.replace(json_path_folder, png_path_folder) for p in json_path_list]
        else:
            raise RuntimeError('Folder "' + png_path_folder  +  '" does not exist. Create folder or leave png_path_folder input blank')
    png_path_list = [p.replace('.json', '.png') for p in png_path_list]
    qty_convert = len(png_path_list)
    qty_converted = 0
    print('****************  Conversion Started *****************')
    for json, png in zip(json_path_list, png_path_list):
        mask = read_mask_from_folder(json, class_name_list)
        mask = mask.numpy()
        if np.max(mask) <= 1: #Normalise before converting data type
            mask = mask * 255.0
        mask = mask.astype(np.uint8)
        cv2.imwrite(png, mask)
        qty_converted += 1
        print('Dataset Total: [' + str(qty_total) + '] - Batch left quantity: [' + str(qty_convert) + '] - Batch progress: [' + str(qty_converted) + ']')
        del (mask)
        gc.collect()

def check_for_exclusions(x=[], exclusion_array = []):
    #result = x.copy()
    result = x.tolist()
    if len(exclusion_array) > 0:
        for string in result:
            for del_str in exclusion_array:
                if del_str in string:
                    result.remove(string)
    return np.array(result)

def check_for_inclusions(x=[], inclusion_array = []):
    if len(inclusion_array) > 0:
        result = []
        for string in x:
                for inc_str in inclusion_array:
                    if inc_str in string:
                        result.append(string)
    else:
        result = x
    return result
def check_search_files(GPU = 0):
    path_to_search_files = os.path.join(os.getcwd(), 'models', '00_search_files', '00_cue')
    if not os.path.isdir(path_to_search_files): raise RuntimeError('Folder ' + path_to_search_files + ' does not exist')
    array_search_paths = np.array([os.path.join(path, name) for path, subdirs, files in os.walk(path_to_search_files)
                                   for name in files
                                   if os.path.splitext(name)[1].lower() == '.json'
                                   ])
    if len(array_search_paths) < 1: raise RuntimeError(
        'Folder ' + path_to_search_files + ' does not contain serach files wtih .json extension')
    # filter files
    filtered_path_to_search_files = []
    for search_file in array_search_paths:

        data = read_from_json(search_file)
        if 'model_trial_info' in data.keys() and 'model_dataset_info' in data.keys() and 'model_tune_param' in data.keys():
            if data['model_trial_info']['status'] != 'complete' and data['model_trial_info']['status'] != 'skip':
                if GPU is not None:
                    if data['model_trial_info']['GPU_Select'] == GPU:
                        filtered_path_to_search_files.append(search_file)
                        print(search_file)
                else:
                    filtered_path_to_search_files.append(search_file)
                    print(search_file)
    num_of_searches = len(filtered_path_to_search_files)
    print('######################################################################################')
    print('######################################################################################')
    print('********  SEARCH SCHEDULE with ' + str(num_of_searches) + ' FILES IN QUE ******************')
    print('######################################################################################')
    print('######################################################################################')
    return filtered_path_to_search_files

def check_matching_extensions(file_path = '', matching_extension = ''):
    new_path = file_path.replace(os.path.splitext(os.path.split(file_path)[1])[1], matching_extension)
    if os.path.exists(new_path):
        return new_path
    else:
        return None


def check_trained_model_files(exclusions=[], inclusions=[]):
    path_to_search_files = os.path.join(os.getcwd(), 'models', '01_trained')
    if not os.path.isdir(path_to_search_files): raise RuntimeError('Folder ' + path_to_search_files + ' does not exist')
    array_search_paths_h5 = np.array([os.path.join(path, name) for path, subdirs, files in os.walk(path_to_search_files)
                                   for name in files
                                   if os.path.splitext(name)[1].lower() == '.h5'
                                   ])
    if len(array_search_paths_h5) < 1: raise RuntimeError(
        'Folder ' + path_to_search_files + ' does not contain trained model files wtih .h5 extension')
    # filter files
    array_search_paths_h5 = check_for_exclusions(array_search_paths_h5, exclusions)
    array_search_paths_h5 = check_for_inclusions(array_search_paths_h5, inclusions)
    array_search_paths_json = []
    for search_path_h5 in array_search_paths_h5:
        json_path = check_matching_extensions(search_path_h5, '.json')
        if json_path is not None:
            array_search_paths_json.append(json_path)
        else:
            array_search_paths_h5.remove(search_path_h5)

    num_of_model_files = len(array_search_paths_json)
    print('######################################################################################')
    print('######################################################################################')
    print('********  Models for testing: ' + str(num_of_model_files) + ' FILES IN QUE ***********')
    print('######################################################################################')
    print('######################################################################################')
    return array_search_paths_h5, array_search_paths_json

def check_log_files( exclusion_string = [], inclusion_string= []):
    path_to_search_files = os.path.join(os.getcwd(), 'logs')
    if not os.path.isdir(path_to_search_files): raise RuntimeError('Folder ' + path_to_search_files + ' does not exist')
    array_search_paths = np.array([os.path.join(path, name) for path, subdirs, files in os.walk(path_to_search_files)
                                   for name in files
                                   if os.path.splitext(name)[1].lower() == '.csv'
                                   ])
    if len(array_search_paths) < 1: raise RuntimeError(
        'Folder ' + path_to_search_files + ' does not contain serach files wtih .csv extension')
    # filter files
    array_search_paths = check_for_exclusions(array_search_paths, exclusion_string)
    array_search_paths = check_for_inclusions(array_search_paths, inclusion_string)

    return array_search_paths


def build_df_from_logs(log_files):
    df_combined = None
    for log in log_files:
        df = pd.read_csv(log)
        file = os.path.splitext(os.path.split(log)[1])[0] + '_'
        df = df.astype({'id': 'string'})
        df['id'] = file + '-' + df['id']
        if df_combined is not None:
            df_combined = pd.concat([df_combined, df])
        else:
            df_combined = df
    df = df_combined

    df['class_distance_filtered'] = np.where(df['ACC'] == 1, df['class_distance'], 0)
    df['h_median_filtered'] = np.where(df['ACC'] == 1, df['h_median'], 0)
    df['ACC_class_distance'] = df['ACC'] + df['class_distance_filtered']
    df['ACC_h_median'] = df['ACC'] + df['h_median_filtered']
    df = df.drop(columns=['h_median_filtered', 'class_distance_filtered'])
    df['seg'] = np.where(df['model_name'].str.contains('deeplabV3'), 1, 0)
    df['seg'] = np.where(df['model_name'].str.contains('Unet'), 1, df['seg'])
    #df['ds_name'] = df.ds_name.apply(lambda x: os.path.split(x)[1])
    df['ds_name'] = df['trial_id']
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev1'), 'Full-50', df['ds_name'])
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev2'), 'Partial-50', df['ds_name'])
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev3'), 'Partial-40', df['ds_name'])
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev4'), 'Partial-30', df['ds_name'])
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev5'), 'Partial-20', df['ds_name'])
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev6'), 'Partial-10', df['ds_name'])
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev7'), 'Partial-05', df['ds_name'])
    df['ds_name'] = np.where(df['ds_name'].str.contains('_rev8'), 'Partial-01', df['ds_name'])
    df.sort_values(by=['trial_id'], inplace=True,
                   ascending=[True])

    return df


def filter_df(df, par1=['',''], par2=None, par3=None):
    df = df[df[par1[0]].str.match(par1[1])]
    if par2 is not None:
        df = df[df[par2[0]].str.match(par2[1])]
    if par3 is not None:
        df = df[df[par3[0]].str.match(par3[1])]
    return df


def print_single_model_ACC_BarH(df, model_name, Augmented):
    df_filtered = filter_df(df, ['model_name', model_name], ['Augmented', Augmented])

    np_ds_names = df_filtered.ds_name.to_numpy()
    np_ACC = df_filtered.ACC.to_numpy()
    np_distance = df_filtered.distance.to_numpy()
    np_ACC_TPA = df_filtered.ACC_TPA.to_numpy()

    # serach by model type (8 datasets, A, and Not Augmented)

    color_mask = (np_ACC < 1)
    color_array = np.chararray(len(color_mask))
    color_array[:] = 'g'
    color_array = color_array.astype(dtype='str')

    color_array[color_mask] = 'b'

    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(np_ds_names))
    ax.barh(y_pos, np_ACC, align='center', color=color_array)
    ax.set_yticks(y_pos, labels=np_ds_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('ACC')
    ax.set_title(model_name + ' Augmentation: ' + Augmented)
    plt.grid(color='b', linestyle='--', linewidth=1, axis='x')
    plt.show()
    return df_filtered

def print_single_model_TPA_BarH(df, model_name, Augmented, print=True):
    df_filtered = filter_df(df, ['model_name', model_name], ['Augmented', Augmented])

    if print:
        np_ds_names = df_filtered.ds_name.to_numpy()
        np_ACC = df_filtered.ACC.to_numpy()
        np_distance = df_filtered.distance.to_numpy()
        np_ACC_TPA = df_filtered.ACC_TPA.to_numpy()

        # serach by model type (8 datasets, A, and Not Augmented)

        color_mask = (np_ACC < 1)
        color_array = np.chararray(len(color_mask))
        color_array[:] = 'g'
        color_array = color_array.astype(dtype='str')

        color_array[color_mask] = 'b'

        plt.rcdefaults()
        fig, ax = plt.subplots()
        y_pos = np.arange(len(np_ds_names))
        ax.barh(y_pos, np_ACC_TPA, align='center', color=color_array)
        ax.set_yticks(y_pos, labels=np_ds_names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('ACC_TPA')
        ax.set_title(model_name + ' Augmentation: ' + Augmented)
        plt.grid(color='b', linestyle='--', linewidth=1, axis='x')
        plt.show()

        plt.close(fig)
    return df_filtered

def print_multiple_h_median_barH(df_NA, df_A, model_name= '', save_file = '', print = True):

    colour_dict = mcolors.CSS4_COLORS
    c_no_aug = mcolors.to_rgb(colour_dict['steelblue'])
    c_aug_w = mcolors.to_rgb(colour_dict['firebrick'])
    c_aug_b = mcolors.to_rgb(colour_dict['mediumseagreen'])
    c_acc_100 = mcolors.to_rgb(colour_dict['lime'])


    np_ds_names = df_NA.ds_name.to_numpy()
    np_ACC_NA = df_NA.ACC.to_numpy()
    np_distance_NA = df_NA.h_median.to_numpy()
    np_ACC_TPA_NA = df_NA.ACC_h_median.to_numpy()
    np_ACC_A = df_A.ACC.to_numpy()
    np_distance_A = df_A.h_median.to_numpy()
    np_ACC_TPA_A = df_A.ACC_h_median.to_numpy()



    color_array_NA = np.full((len(np_ACC_NA),3), c_no_aug)
    color_array_A = np.full((len(np_ACC_NA),3), c_aug_w)
    color_mask_A = (np_ACC_TPA_A >  np_ACC_TPA_NA)
    color_array_A[color_mask_A] = c_aug_b


    plt.rcdefaults()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    y_pos = np.arange(len(np_ds_names))
    ax1.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5], [0.5, 0.75, 1.0, 1.25, 1.5])
    ax1.set_xlim(0.5, 1.5)


    ax1.barh(y_pos, np_ACC_TPA_NA, align='center', color=color_array_NA)
    ax1.set_yticks(y_pos, labels=np_ds_names)
    ax1.invert_yaxis()  # labels read top-to-bottom
    ax1.set_xlabel('         ACC        + median[hist dist]')
    ax1.set_title('Not Augmented' )
    ax1.grid(color='b', linestyle='--', linewidth=1, axis='x')
    ax1.plot([1.0, 1.0], [-0.5, 7.5], color=c_acc_100)

    ax2.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5], [0.5, 0.75, 1.0, 1.25, 1.5])
    ax2.set_xlim(0.5, 1.5)
    ax2.barh(y_pos, np_ACC_TPA_A, align='center', color=color_array_A, tick_label = ['50%', '50%','40%','30%','20%','10%','5%', '1%'])
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('         ACC        + median[hist dist]')
    ax2.set_title('Augmented')
    ax2.grid(color='b', linestyle='--', linewidth=1, axis='x')
    ax2.plot([1.0, 1.0], [-0.5, 7.5], color=c_acc_100)

    fig.suptitle(model_name)

    if len(save_file)> 0:
        plt.savefig(save_file)
    if print: plt.show()
    plt.close(fig)



def print_multiple_ClassDistance_barH(df_NA, df_A, model_name= '', save_file = '', print = True):

    colour_dict = mcolors.CSS4_COLORS
    c_no_aug = mcolors.to_rgb(colour_dict['steelblue'])
    c_aug_w = mcolors.to_rgb(colour_dict['firebrick'])
    c_aug_b = mcolors.to_rgb(colour_dict['mediumseagreen'])
    c_acc_100 = mcolors.to_rgb(colour_dict['lime'])


    np_ds_names = df_NA.ds_name.to_numpy()
    np_ACC_NA = df_NA.ACC.to_numpy()
    np_distance_NA = df_NA.class_distance.to_numpy()
    np_ACC_TPA_NA = df_NA.ACC_class_distance.to_numpy()
    np_ACC_A = df_A.ACC.to_numpy()
    np_distance_A = df_A.class_distance.to_numpy()
    np_ACC_TPA_A = df_A.ACC_class_distance.to_numpy()



    color_array_NA = np.full((len(np_ACC_NA),3), c_no_aug)
    color_array_A = np.full((len(np_ACC_NA),3), c_aug_w)
    color_mask_A = (np_ACC_TPA_A >  np_ACC_TPA_NA)
    color_array_A[color_mask_A] = c_aug_b


    plt.rcdefaults()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    y_pos = np.arange(len(np_ds_names))
    ax1.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75,  2.0], [0.5, 0.75, 1.0, 1.25, 1.5, 1.75,  2.0])
    ax1.set_xlim(0.5, 2.0)


    ax1.barh(y_pos, np_ACC_TPA_NA, align='center', color=color_array_NA)
    ax1.set_yticks(y_pos, labels=np_ds_names)
    ax1.invert_yaxis()  # labels read top-to-bottom
    ax1.set_xlabel('ACC         + Class Distance')
    ax1.set_title('Not Augmented' )
    ax1.grid(color='b', linestyle='--', linewidth=1, axis='x')
    ax1.plot([1.0, 1.0], [-0.5, 7.5], color=c_acc_100)

    ax2.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    ax2.set_xlim(0.5, 2.0)
    ax2.barh(y_pos, np_ACC_TPA_A, align='center', color=color_array_A, tick_label = ['50%', '50%','40%','30%','20%','10%','5%', '1%'])
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('ACC         + Class Distance')
    ax2.set_title('Augmented')
    ax2.grid(color='b', linestyle='--', linewidth=1, axis='x')
    ax2.plot([1.0, 1.0], [-0.5, 7.5], color=c_acc_100)

    fig.suptitle(model_name)
    if len(save_file)> 0:
        plt.savefig(save_file)
    if print: plt.show()
    plt.close(fig)



def generate_plots(log_folder, print=True, exclusion_string = [], inclusion_string= []):


    log_files = check_log_files( exclusion_string = exclusion_string, inclusion_string= inclusion_string)

    path_to_log_files = os.path.join(os.getcwd(), 'logs', log_folder)
    if not os.path.exists(path_to_log_files):
        os.makedirs(path_to_log_files)
    df = build_df_from_logs(log_files)
    list_of_models = df['model_name'].drop_duplicates()
    list_of_datasets = df['ds_name'].drop_duplicates()

    for model_name in list_of_models:
        df_filtered_NA = print_single_model_TPA_BarH(df, model_name, 'N', print=False)
        df_filtered_A = print_single_model_TPA_BarH(df, model_name, 'Y', print=False)
        files_name_ClassDistance = os.path.join(path_to_log_files, model_name + '_CD.jpg')
        files_name_h_median = os.path.join(path_to_log_files, model_name + '_HM.jpg')
        print_multiple_ClassDistance_barH(df_filtered_NA, df_filtered_A, model_name, files_name_ClassDistance, print)
        print_multiple_h_median_barH(df_filtered_NA, df_filtered_A, model_name, files_name_h_median, print)




def create_sumary_logs_by_dataset(log_folder, exclusion_string = [], inclusion_string= []):

    log_files = check_log_files(exclusion_string=exclusion_string, inclusion_string=inclusion_string)
    path_to_log_files = os.path.join(os.getcwd(), 'logs', log_folder)
    if not os.path.exists(path_to_log_files):
        os.makedirs(path_to_log_files)
    df = build_df_from_logs(log_files)
    list_of_models = df['model_name'].drop_duplicates()
    list_of_datasets = df['ds_name'].drop_duplicates()

    list_of_models = list_of_models.to_numpy()
    list_of_datasets = list_of_datasets.to_numpy()
    for i in range(len(list_of_datasets)):
        log_name = list_of_datasets[i]
        df_f = filter_df(df, par1=['ds_name',list_of_datasets[i]], par2=['Augmented', 'N'])
        df_f = df_f.drop(columns=['id', 'trial_id', 'ds_split',  'ds_name', 'Augmented'])
        df_f.sort_values(by=['ACC', 'h_median', 'avg_inferencing_time_ms'], inplace=True,
                       ascending=[False, False, True])
        df_f.set_index('model_name', inplace=True, drop=True)
        df_fa = filter_df(df, par1=['ds_name',list_of_datasets[i]], par2=['Augmented', 'Y'])
        df_fa = df_fa.drop(columns=['id', 'trial_id', 'ds_split',  'ds_name', 'Augmented'])
        df_fa.set_index('model_name', inplace=True, drop=True)
        df_fa = df_fa.reindex(df_f.index)

        df_f['aug_ACC'] = df_fa['ACC']
        df_f['aug_LOSS'] = df_fa['LOSS']

        df_f['aug_class_distance'] = df_fa['class_distance']
        df_f['aug_ACC_class_distance'] = df_fa['ACC_class_distance']
        df_f['aug_h_median'] = df_fa['h_median']
        df_f['aug_ACC_h_median'] = df_fa['ACC_h_median']

        df_f['aug_diff'] = df_f['aug_ACC_h_median'] - df_f['ACC_h_median']

        df_f['aug_improves'] = np.where(df_f['aug_diff'] > 0, 1, df_f['aug_diff'])
        df_f['aug_improves'] = np.where(df_f['aug_improves'] < 0, -1, df_f['aug_improves'])


        df_f = df_f[['ACC', 'LOSS', 'Precision', 'recall', 'f1', 'class_distance', 'ACC_class_distance',
                     'h_median', 'ACC_h_median',  'aug_ACC', 'aug_LOSS',
                     'aug_class_distance', 'aug_ACC_class_distance',
                     'aug_h_median', 'aug_ACC_h_median',
                     'avg_inferencing_time_ms', 'aug_improves', 'seg']]
        df_f.to_csv(os.path.join(path_to_log_files, log_name+'.csv'))
        del df_fa