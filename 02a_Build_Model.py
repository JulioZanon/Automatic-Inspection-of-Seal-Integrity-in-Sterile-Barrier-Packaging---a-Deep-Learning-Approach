# 02a_Build_Model Rev:01.00
"""
This script allow us to build a model, initialize it and save it for tuning.


Grid:
Model
Optimiser function
Loss function

Bayessian:
Bath size
Epochs before and after
Learning rate
Leraning decay
Optimser function momentum


#TENSORBOARD

%load_ext tensorboard
tensorboard --logdir models/01_trained/test    "where test is the name of the experiment"

"""

#imports
import tensorflow as tf
import numpy as np
from custom_lib.json_function import write_to_json, read_from_json
from custom_lib.model_tune import tune_model
from custom_lib.model_train_functions import get_trial_planer
from custom_lib.model_build_funtions import check_GPUs_availibility
from custom_lib.model_test import test_model
from custom_lib.user_functions import check_search_files
import os
import gc

v_save_model = True
v_test_model = True # dataset_test_split must contain the name of the test set and v_save_model must be true
log_name= 'Results from 02a'
v_show_plots = False
GPU = 0  #Set to None to ignore GPU selection
log_name = log_name + '_' + str(GPU)
GPU_count = check_GPUs_availibility(GPU)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
########################################################################################################################

while True:
    skip_sarch = False
    path_to_search_files = check_search_files(GPU)
    if len(path_to_search_files)> 0:
        for search_file in path_to_search_files:
            data = read_from_json(search_file)
            data['model_trial_info']['status'] = 'started'
            write_to_json(data, search_file)
            model_trial_info = data['model_trial_info']
            model_dataset_info = data['model_dataset_info']
            model_tune_param = data['model_tune_param']
            if 'trial_plan' in data.keys():
                trial_plan = data['trial_plan']
            else:
                trial_plan = []
            if  model_trial_info['trial_id'] == '':
                model_trial_info['trial_id'] = os.path.splitext(os.path.split(search_file)[1])[0]

            # instanciate trial generator
            trial_idx = get_trial_planer(model_trial_info, model_dataset_info, model_tune_param, trial_plan)
            print('######################################################################################')
            print('********  START TUNING TRIAL PLAN')
            print('*************************************************************************************')
            best_val_loss = 100000
            val_loss_history = []
            Path_to_model = os.path.join(os.getcwd(), 'models\\01_trained')
            os.makedirs(Path_to_model, exist_ok=True)

            try:
                for i in range(trial_idx.num_iter):
                    p_model_trial_info, p_model_dataset_info, p_model_tune_param = next(trial_idx)
                    if not p_model_tune_param['p_No_augmentation'] and p_model_tune_param['p_augmentation']: p_augmentation = True
                    else: p_augmentation = False
                    print('*************************************************************************************')
                    print('-- START Trial [' + str(i + 1) + '] of [' + str(trial_idx.num_iter) + '] -- trial_id:' +
                          p_model_trial_info['trial_id'])
                    print('*************************************************************************************')
                    #p_augmentation = True
                    model, model_info = tune_model(p_model_trial_info, p_model_dataset_info, p_model_tune_param, augment=p_augmentation)
                    gc.collect()
                    print('*************************************************************************************')
                    print(
                        '-- END Trial [' + str(i + 1) + '] of [' + str(trial_idx.num_iter) + '] -- trial_id:' + p_model_trial_info[
                            'trial_id'])
                    print('--RESULT: trial_id:' + p_model_trial_info['trial_id'] + ' val_acc: ' + str(
                        model_info['performance']['val_acc']) + '  val_loss: ' + str(model_info['performance']['val_loss']))

                    # Filter Nan valies as 0.
                    if np.isnan(model_info['performance']['val_loss']): RuntimeError('Nan')

                    val_loss_history.append(model_info['performance']['val_loss'])
                    if model_info['performance']['val_loss'] < best_val_loss or np.isnan(model_info['performance']['val_loss']):
                        best_val_loss = model_info['performance']['val_loss']
                        best_model = tf.keras.models.clone_model(model)
                        best_model.set_weights(model.get_weights())
                        best_model_info = model_info.copy()
                    if i > 0:
                        print('--BEST:   trial_id:' + best_model_info['trial_info']['trial_id'] + ' val_acc: ' + str(
                            best_model_info['performance']['val_acc']) + '  val_loss: ' + str(
                            best_model_info['performance']['val_loss']))
                    print('*************************************************************************************')
                del model
                gc.collect()
                # Train with Augmentation - No tuning .. just train same model
                if p_model_tune_param['p_No_augmentation'] and p_model_tune_param['p_augmentation']:
                    p_model_trial_info_A = p_model_trial_info.copy()
                    p_model_tune_param_A = p_model_tune_param.copy()
                    p_augmentation = True
                    p_model_tune_param_A['p_Batch_size'] = best_model_info['tune_param']['p_Batch_size']
                    p_model_tune_param_A['p_bayesian_max_trials'] = 0
                    p_model_tune_param_A['p_num_epochs'] = best_model_info['tune_param']['p_num_epochs']
                    p_model_tune_param_A['p_learning_rate_bArray'] = best_model_info['tune_param']['p_learning_rate']
                    p_model_tune_param_A['p_Exp_decay_rate_momentum1_bArray'] = best_model_info['tune_param'][
                        'p_Exp_decay_rate_momentum1']
                    p_model_tune_param_A['p_Exp_decay_rate_momentum2_bArray'] = best_model_info['tune_param'][
                        'p_Exp_decay_rate_momentum2']
                    p_model_tune_param_A['p_optimiser_function_bArray'] = best_model_info['tune_param']['p_optimiser_function']
                    p_model_tune_param_A['p_loss_function_bArray'] = best_model_info['tune_param']['p_loss_function']

                    print('*************************************************************************************')
                    print('-- Train with Augmentation trial_id: ' +
                          p_model_trial_info_A['trial_id'])
                    print('*************************************************************************************')
                    model_A, model_info_A = tune_model(p_model_trial_info_A, p_model_dataset_info, p_model_tune_param_A, augment=p_augmentation)
                    print('*************************************************************************************')
                    print('--RESULT: trial_id:' + p_model_trial_info_A['trial_id'] + ' val_acc: ' + str(
                        model_info_A['performance']['val_acc']) + '  val_loss: ' + str(model_info_A['performance']['val_loss']))

                #Save model and Test
                if v_save_model:
                    #save best model
                    model_path = os.path.join(Path_to_model, best_model_info['trial_info']['trial_id'] + '.h5')
                    best_model.save(model_path)
                    write_to_json(best_model_info, os.path.join(Path_to_model, best_model_info['trial_info']['trial_id']  + '.json'))
                    data['model_trial_info']['status'] = 'complete'
                    os.remove(search_file)
                    write_to_json(data, search_file.replace('00_cue', '02_complete'))

                    print('-- BEST MODEL SAVED as: ' + model_path)
                    if not data['model_trial_info']['delet_logs']:
                        print('To see results in TENSORBOARD, type the following two lines and click in the link')
                        print('%load_ext tensorboard')
                        print('tensorboard --logdir models/01_trained/'+ best_model_info['trial_info']['trial_id'])

                    # save Augmented model
                    if p_model_tune_param['p_No_augmentation'] and p_model_tune_param['p_augmentation']:
                        model_info_A['trial_info']['trial_id'] = model_info_A['trial_info']['trial_id'] + '_A'
                        model_path_A = os.path.join(Path_to_model, model_info_A['trial_info']['trial_id'] + '.h5')
                        model_A.save(model_path_A)
                        write_to_json(model_info_A,
                                      os.path.join(Path_to_model, model_info_A['trial_info']['trial_id'] + '.json'))

                        print('-- AUGMENTED MODEL SAVED as: ' + model_path_A)

                    # Test model from saved folder
                    if v_test_model and len(model_dataset_info['dataset_test_split']) > 0:
                        metrics, summary_log = test_model(model=model_path,
                                                          model_info=best_model_info,
                                                          dataset_info_name=model_dataset_info['dataset_info_name'],
                                                          dataset_test_split=model_dataset_info['dataset_test_split'],
                                                          log_name=log_name,
                                                          show_plots=v_show_plots,
                                                          test_Batch_size=8)

                        if p_model_tune_param['p_No_augmentation'] and p_model_tune_param['p_augmentation']:
                            metrics_A, summary_log_A = test_model(model=model_path_A,
                                                                  model_info=model_info_A,
                                                                  dataset_info_name=model_dataset_info['dataset_info_name'],
                                                                  dataset_test_split=model_dataset_info['dataset_test_split'],
                                                                  log_name=log_name,
                                                                  show_plots=v_show_plots,
                                                                  test_Batch_size=8)


            except Exception as err:
                data['model_trial_info']['status'] = 'err'
                os.remove(search_file)
                write_to_json(data, search_file.replace('00_cue', '01_err'))
                print('Search aborted with error: ' + str(err))
    else: break

