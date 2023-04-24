import tensorflow as tf
import os
import math
import time
import numpy as np
from custom_lib.model_build_funtions import check_GPUs_availibility, model_library_info, convert_seg_mask_to_classification_array
from custom_lib.dataset_functions import get_info_from_dataset_json
from custom_lib.image_generators import ImgDatasetGenerator
from custom_lib.metrics_functions import metrics_multi_class
import pandas as pd




def test_model(model = '', model_info = {}, dataset_info_name = '', dataset_test_split='', log_name = '', show_plots = True, test_Batch_size = 32):

    # Diagnostics
    if type(model) == str or type(model) == np.str_:
        model = tf.keras.models.load_model(model, compile=False)
    # Test model
    info_dict = get_info_from_dataset_json(dataset_info_name, dataset_test_split)
    img_params = info_dict['img_params'].copy()
    img_params[3] = model_info['tune_param']['P_input_size'][:-1]

    # check label type
    if model_library_info[model_info['tune_param']['p_model_name']]['label_type'] == 'mask' and info_dict[
        'segmentation_mask'] and \
            'notation_path_column' in info_dict.keys() and 'notation_folder' in info_dict.keys():
        p_label_type = 'mask'
        notation_path_column = info_dict['notation_path_column']
    else:  # If the dataset contains segmentation masks but the model used only support classes, the notation path column
        # is set to '' to force the image generator to create classification lables as ground thruth
        p_label_type = 'class'
        notation_path_column = ''

    TestDataGen = ImgDatasetGenerator(PreProcessingImgParams=img_params,
                                      dataframe_path=info_dict['dataframe_path'],
                                      dataset_folder=info_dict['dataset_folder'],
                                      notation_folder=info_dict['notation_folder'],
                                      img_path_column=info_dict['img_path_column'],
                                      notation_path_column=notation_path_column,
                                      class_name_list=info_dict['class_names'],
                                      class_column=info_dict['class_column'],
                                      shuffle_dataset=False,
                                      batch_size=test_Batch_size)
    start_time = time.time()
    predictions = model.predict(TestDataGen, steps=int(math.ceil(TestDataGen.dataset_qty / test_Batch_size)), verbose=1)
    # predictions = model.predict_generator(generator=TestDataGen,
    #                                       steps=int(math.ceil(TestDataGen.dataset_qty / test_Batch_size)), verbose=1)
    batch_prediction_time = time.time() - start_time
    avg_time_per_img = round(((batch_prediction_time / TestDataGen.dataset_qty) * 1000), 3)
    batch_prediction_time = round(batch_prediction_time, 3)
    if p_label_type == 'mask':
        predictions, _ = convert_seg_mask_to_classification_array(predictions, conv_type='Sum&NormZeroToOne')

    if len(log_name) > 0:

        if not os.path.isdir(os.path.join(os.getcwd(), 'logs', log_name)): os.makedirs(
            os.path.join(os.getcwd(), 'logs', log_name))
        if not os.path.isdir(
            os.path.join(os.getcwd(), 'logs', log_name, model_info['trial_info']['trial_id'])): os.makedirs(
            os.path.join(os.getcwd(), 'logs', log_name, model_info['trial_info']['trial_id']))

    metrics = metrics_multi_class(TestDataGen.class_array, predictions, TestDataGen.class_name_list,
                                  plot_title=model_info['trial_info']['trial_id'],
                                  report_path_name=os.path.join(os.getcwd(), 'logs', log_name,
                                                                model_info['trial_info']['trial_id']),
                                  dataset_name=dataset_info_name,
                                  dataset_split=info_dict['split_info']['split_name'],
                                  model_name=model_info['trial_info']['trial_id'],
                                  show_plots=show_plots)
    if len(log_name) > 0:
        # Read df if CSV exist
        path_to_log_file = os.path.join(os.getcwd(), 'logs', log_name + '.csv')
        if os.path.isfile(path_to_log_file):
            df = pd.read_csv(path_to_log_file)
            df_count = len(df.index) + 1
        else:
            # Otherwise declare an empty dataframe
            df = pd.DataFrame(
                columns=['id', 'model_name', 'trial_id', 'ds_name', 'ds_split', 'Augmented', 'ACC', 'LOSS', 'Precision', 'recall', 'f1',
                         'threshold', 'h_std', 'h_median','class_distance', 'avg_inferencing_time_ms' ])
            df_count = 1
        # append new values
        if model_info['tune_param']['p_augmentation']: Augmented = 'Y'
        else: Augmented = 'N'


        #TODO: metrics dictionary has changed ... update below to reflect new keys and updated keys
        df = df.append(
            {'id': str(df_count), 'model_name': model_info['tune_param']['p_model_name'], 'trial_id': model_info['trial_info']['trial_id'],
             'ds_name': metrics['info']['dataset_name'], 'ds_split': metrics['info']['dataset_split'],
             'Augmented': Augmented,
             'ACC': metrics[metrics['info']['class_name_list'][-1]]['acc'],
             'LOSS': metrics[metrics['info']['class_name_list'][-1]]['loss'],
             'Precision': metrics[metrics['info']['class_name_list'][-1]]['precision'],
             'recall': metrics[metrics['info']['class_name_list'][-1]]['recall'],
             'f1': metrics[metrics['info']['class_name_list'][-1]]['f1'],
             'threshold': metrics[metrics['info']['class_name_list'][-1]]['threshold'],
             'h_std': metrics[metrics['info']['class_name_list'][-1]]['h_std'],
             'h_median': metrics[metrics['info']['class_name_list'][-1]]['h_median'],
             'class_distance': metrics[metrics['info']['class_name_list'][-1]]['class_distance'],
             'avg_inferencing_time_ms': avg_time_per_img
             },
            ignore_index=True)
        # save Dataframe to CSV
        df.to_csv(path_to_log_file, index=False, encoding='utf8')
        summary_log = df
    else: summary_log = None
    print('Test Results for Dataset: [' + dataset_info_name + '] with split: [' + dataset_test_split + ']')
    print('#####################################################################')
    print('Prediction Time: ' + str(avg_time_per_img) + ' - Loss: ' + str(metrics[metrics['info']['class_name_list'][-1]]['loss']) + ' - Acc: ' +
          str(metrics[metrics['info']['class_name_list'][-1]]['acc']) + ' - class distance: ' +  str(metrics[metrics['info']['class_name_list'][-1]]['class_distance']) )
    return metrics, summary_log