# model_tune Rev:01.00


# imports
from custom_lib.model_train_functions import compile_model, get_hyp_param_tuner_func, kt_BayesianOptimization, extract_history_from_tuner
from custom_lib.model_build_funtions import get_num_of_layers, model_library_info
from custom_lib.json_function import calculate_model_inp_size, get_model_from_library
from custom_lib.dataset_functions import get_info_from_dataset_json
from custom_lib.model_fit_callbacks import DisplayCallback, callback_earlystop_val_loss
from custom_lib.image_generators import ImgDatasetGenerator
import os
import gc
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow as tf


#
model_trial_info = {
    "trial_id": "VGG16_rev7",
    "silent_search_error": True,
    "print_plots": False,
    "GPU_Select": 0,
    "delet_logs": True,
    "status": "err"
}
model_dataset_info = {
    'dataset_info_name': 'middle_seal_breach\\rev_7_Partial-05',
    # Path to the dataset info. files "ds_info.json" and ".CSV" split files. This should be in project folder ".\datasets\"
    'dataset_split': 'TrainingXVal',
    # Dataset split name. Folder above should conatin a ".csv" file with same name conating split inforatmion. tyical names: 'TrainingXVal', 'Testing'.
    'dataset_v_class_filter': [],
    # class indexes to be included. Empty list will include all classes. This filter does not exclude images, only labels.
    'dataset_exclude_imgs_in_class': [],  # Class index for classes to be excluded.
    'dataset_Val_split_ratio': 0.7  # Split ration between Trainin
}
model_tune_param = {
    "p_Batch_size": 16,
    "p_model_name": "VGG16",
    "P_input_size": [
        64,
        128,
        3
    ],
    "p_early_stop_patience": 3,
    "p_augmentation": False,
    "p_bayesian_max_trials": 4,
    "p_num_epochs": 0,
    "p_learning_rate_bArray": [
        5e-05,
        1e-05,
        0.0001,
        2e-05
    ],
    "p_optimiser_function_bArray": [
        "RMSprop"
    ],
    "p_Exp_decay_rate_momentum1_bArray": [
        0.9,
        0.85,
        0.95,
        0.02
    ],
    "p_Exp_decay_rate_momentum2_bArray": [
        0.985,
        0.96,
        0.999,
        0.005
    ],
    "p_loss_function_bArray": [
        "BinaryCrossentropy"
    ],
    "P_transfer_weights": "",
    "p_num_epochs_transfer_init": 2,
    "p_learning_rate_transfer_init": 0.0001,
    "p_freez_first_layers_init": 0,
    "p_transfer_model_to_FC": "GlobalMaxPooling2D",
    "p_CNN_Depth": 2,
    "p_conv_dropout": 0.05,
    "p_fc_dropout": 0.05,
    "p_pooling_cnv": "AveragePooling2D",
    "p_pooling_fc": "GlobalAveragePooling2D"

}


def tune_model(model_trial_info={}, model_dataset_info={}, model_tune_param={}, augment = False):
    """
    Initialise model (if transder weights are required) and then tune hyper parameters either with a
    set of parameters or performing a bayesian optimisation.
    :param model_trial_info:
    :param model_dataset_info:
    :param model_tune_param:
    :param augment:
    :return: Returns the trained model and the information dictionary.
    """

    # Get summary information required to run the script from the info.json file
    info_dict = get_info_from_dataset_json(model_dataset_info['dataset_info_name'], model_dataset_info['dataset_split'])
    # Get image pre-processing parameters such as size and cropping.
    if 'img_params' in info_dict:
        img_params = info_dict['img_params'].copy()
    else:
        img_params = None
    # Check if search is valid
    ##########################
    skip_search = False
    # Get model info
    model_info = model_library_info[model_tune_param['p_model_name']]
    # Set input size
    if model_tune_param['P_input_size'] is None:
        img_input_shape = model_info['min_input_size']
    elif len(model_tune_param['P_input_size']) == 3:
        if model_tune_param['P_input_size'][0] < model_info['min_input_size'][0] or model_tune_param['P_input_size'][1] < model_info['min_input_size'][1] or \
                model_tune_param['P_input_size'][2] < model_info['min_input_size'][2]:
            if model_trial_info['silent_search_error']:
                skip_search = True
            else:
                raise RuntimeError('P_input_size ' + str(
                    model_tune_param['P_input_size']) + 'not supported by model ' + model_tune_param['p_model_name'] + ' with design size of ' + str(
                    model_info['min_input_size']))
        img_input_shape = calculate_model_inp_size(model_info['min_input_size'], model_tune_param['P_input_size'])
    else:
        img_input_shape = calculate_model_inp_size(model_info['min_input_size'],
                                                   [img_params[3][0], img_params[3][1], 3])
    img_params[3] = img_input_shape[:-1]
    num_of_classes = len(info_dict['class_names'])

    if model_tune_param['P_transfer_weights'] == '':
        P_transfer_weights = model_info['weights'][0]
    else:
        if model_tune_param['P_transfer_weights'] not in model_info['weights']:
            if model_trial_info['silent_search_error']:
                skip_search = True
            else:
                raise RuntimeError(
                    'P_transfer_weights "' + model_tune_param['P_transfer_weights'] + '" not supported by model ' + model_tune_param['p_model_name'] + ' with design weights: ' +
                    model_info['weights'])

    if model_info['w_transfer_init_required'] and ( model_tune_param['p_num_epochs_transfer_init'] <= 0 ):
        if model_trial_info['silent_search_error']:
            skip_search = True
        else:
            raise RuntimeError(
                'Model requires weight transfer, therefore p_num_epochs_transfer_init should be > 0')

    if model_info['w_transfer_init_required'] and (model_tune_param['p_learning_rate_transfer_init'] <= 0 ):
        if model_trial_info['silent_search_error']:
            skip_search = True
        else:
            raise RuntimeError(
                'Model requires weight transfer, therefore p_learning_rate_transfer_init should be > 0')


    if model_info['w_transfer_init_required'] and (len(model_tune_param['p_transfer_model_to_FC']) == 0):
        if model_trial_info['silent_search_error']:
            skip_search = True
        else:
            raise RuntimeError(
                'Model requires weight transfer, therfore p_transfer_model_to_FC list should have at least one element')

    p_loss_function_filter = []
    if isinstance(model_tune_param['p_loss_function_bArray'], list):
        for lf in model_tune_param['p_loss_function_bArray']:
            if lf in model_info['loss_functions']: p_loss_function_filter.append(lf)
    elif model_tune_param['p_loss_function_bArray'] in model_info['loss_functions']:
        p_loss_function_filter.append(model_tune_param['p_loss_function_bArray'])
    if len(p_loss_function_filter) > 0:
        p_loss_function = p_loss_function_filter
    else:
        if model_trial_info['silent_search_error']:
            skip_search = True
        else:
            raise RuntimeError('Model dos not support any of the specified loss functions')
    ##########################
    # If search is valid ...
    ##########################
    if not skip_search:
        # check label type (is it a mask) to add column to the dataset df
        if model_library_info[model_tune_param['p_model_name']]['label_type'] == 'mask' and info_dict['segmentation_mask'] and \
                'notation_path_column' in info_dict.keys() and 'notation_folder' in info_dict.keys():
            p_label_type = 'mask'
            notation_path_column = info_dict['notation_path_column']
        else:  # If the dataset contains segmentation masks but the model used only support classes, the notation path column
            # is set to '' to force the image generator to create classification lables as ground thruth
            p_label_type = 'class'
            notation_path_column = ''
        # check if dataset must be Augmented
        if not augment:
            Augmentation = None
        else:
            Augmentation = info_dict['img_augmentation']

        # Instanciate the dataset generators
        TrainingDataGen = ImgDatasetGenerator(PreProcessingImgParams=img_params,
                                              Augmentation=Augmentation,
                                              dataframe_path=info_dict['dataframe_path'],
                                              dataset_folder=info_dict['dataset_folder'],
                                              notation_folder=info_dict['notation_folder'],
                                              img_path_column=info_dict['img_path_column'],
                                              notation_path_column=notation_path_column,
                                              class_name_list=info_dict['class_names'],
                                              class_column=info_dict['class_column'],
                                              class_filter=model_dataset_info['dataset_v_class_filter'],
                                              exclude_imgs_in_class=model_dataset_info['dataset_exclude_imgs_in_class'],
                                              batch_size=model_tune_param['p_Batch_size'],
                                              val_split=model_dataset_info['dataset_Val_split_ratio'],
                                              val_split_idx=None, shuffle_dataset=True)

        ValidationDataGen = ImgDatasetGenerator(PreProcessingImgParams=img_params,
                                                Augmentation=None,
                                                dataframe_path=info_dict['dataframe_path'],
                                                dataset_folder=info_dict['dataset_folder'],
                                                notation_folder=info_dict['notation_folder'],
                                                img_path_column=info_dict['img_path_column'],
                                                notation_path_column=notation_path_column,
                                                class_name_list=info_dict['class_names'],
                                                class_column=info_dict['class_column'],
                                                class_filter=model_dataset_info['dataset_v_class_filter'],
                                                exclude_imgs_in_class=model_dataset_info['dataset_exclude_imgs_in_class'],
                                                batch_size=model_tune_param['p_Batch_size'],
                                                val_split=model_dataset_info['dataset_Val_split_ratio'],
                                                val_split_idx=TrainingDataGen.val_idxs,
                                                shuffle_dataset=True)

        # Get model
        if P_transfer_weights == '':  P_transfer_weights = model_info['weights'][0]
        # Prepare model arguments
        arguments = {
            'freeze_layers': True,
            'transfer_model_to_FC': model_tune_param['p_transfer_model_to_FC'],
            'VGGbnV1_depth': model_tune_param['p_CNN_Depth'],
            'VGGbnV1_conv_dropout': model_tune_param['p_conv_dropout'],
            'VGGbnV1_fc_dropout': model_tune_param['p_fc_dropout'],
            'VGGbnV1_activation': 'relu',
            'VGGbnV1_pooling_cnv': model_tune_param['p_pooling_cnv'],
            'VGGbnV1_pooling_fc': model_tune_param['p_pooling_fc'],
            'VGGbnV1_initializer': 'glorot_uniform'
        }
        if model_info['required_arguments'] is not None:
            model_args = {}
            for key in model_info['required_arguments']:
                model_args.update({key: arguments[key]})
        else:
            model_args = None

        # get model form function
        if model_info['source_code'] == 'get_model_from_library':
           model = get_model_from_library(base_model_name=model_tune_param['p_model_name'], img_shape=img_input_shape,
                                           weights=P_transfer_weights, num_of_classes=num_of_classes, args=model_args)

        # compile model with optimiser and loss function
        class_freqs = info_dict['split_info']['split_num_of_samples_per_class']
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        # Model Compile and fit will execute twice for transfer learning to train before unfreezing layers
        # or once if the model does not require retraining with frozen layers.
        ###########################################################################################################
        # Initialise transfer model ###############################################################################
        print("************* Initialising model")
        if model_info['w_transfer_init_required'] and model_tune_param['p_num_epochs_transfer_init'] > 0:
            learning_rate = model_tune_param['p_learning_rate_transfer_init']
            num_epochs = model_tune_param['p_num_epochs_transfer_init']
            cb1 = DisplayCallback()

            model = compile_model(model, lr=learning_rate, momentum1=model_tune_param['p_Exp_decay_rate_momentum1_bArray'],
                                  momentum2=model_tune_param['p_Exp_decay_rate_momentum2_bArray'], optimiser_func=model_tune_param['p_optimiser_function_bArray'],
                                  loss_func=p_loss_function, class_freqs=class_freqs)
            history = model.fit(TrainingDataGen,
                                epochs=num_epochs,
                                verbose=2,  # 0 = silent, 1 = progress bar, 2 = one line per epoch
                                steps_per_epoch=int(math.ceil(TrainingDataGen.dataset_qty / model_tune_param['p_Batch_size'])),
                                validation_steps=int(math.ceil(ValidationDataGen.dataset_qty / model_tune_param['p_Batch_size'])),
                                validation_data=ValidationDataGen,
                                shuffle=False,
                                validation_freq=1,
                                callbacks=[cb1])

            acc += history.history['acc']
            val_acc += history.history['val_acc']
            loss += history.history['loss']
            val_loss += history.history['val_loss']
            # Make all layers trainable
            num_of_layers, not_trainable_layers = get_num_of_layers(model)
            # unfreeze layers from base transfer model
            if model_tune_param['p_freez_first_layers_init'] < not_trainable_layers:
                model.trainable = True
                if model_tune_param['p_freez_first_layers_init'] > 0:
                    for layer in model.layers[:model_tune_param['p_freez_first_layers_init']]:
                        layer.trainable = False

        if model_tune_param['p_early_stop_patience'] > 0:
            cb1 = callback_earlystop_val_loss(resolution=0.01, patience=model_tune_param['p_early_stop_patience'])
        else:
            cb1 = DisplayCallback()
        ###########################################################################################################
        # Train Model ###############################################################################
        if model_tune_param['p_num_epochs'] == 0:
            ###########################################################################################################
            # Hyperparameter Tuning with Bayesian optimiser #############

            compile_model_with_search = get_hyp_param_tuner_func(model, lr=model_tune_param['p_learning_rate_bArray'],
                                                                 momentum1=model_tune_param['p_Exp_decay_rate_momentum1_bArray'],
                                                                 momentum2=model_tune_param['p_Exp_decay_rate_momentum2_bArray'],
                                                                 optimiser_func=model_tune_param['p_optimiser_function_bArray'],
                                                                 loss_func=p_loss_function, class_freqs=class_freqs)
            root = os.path.join(os.getcwd())
            if os.path.split(root)[-1] == 'custom_lib':
                root = os.path.split(root)[0]
            directory = os.path.join(root, 'models\\01_trained')

            tuner = kt_BayesianOptimization(compile_model_with_search, objective=kt.Objective("val_loss", direction="min"),
                                            max_trials=model_tune_param['p_bayesian_max_trials'], overwrite=True,
                                            directory=directory, project_name=model_trial_info['trial_id'])
            cb2 = tf.keras.callbacks.TensorBoard(os.path.join(directory,model_trial_info['trial_id']))
            tuner.search(TrainingDataGen,
                         # epochs=num_epochs, # Epochs search set to 10-30 in custom_lib.model_train_functions class kt_BayesianOptimization
                         initial_epoch=0,  # last_epoch,
                         verbose=2,  # 0 = silent, 1 = progress bar, 2 = one line per epoch
                         steps_per_epoch=int(math.ceil(TrainingDataGen.dataset_qty / model_tune_param['p_Batch_size'])),
                         validation_steps=int(math.ceil(ValidationDataGen.dataset_qty / model_tune_param['p_Batch_size'])),
                         validation_data=ValidationDataGen,
                         shuffle=False,
                         validation_freq=1,
                         callbacks=[cb1, cb2])
            #gc.collect()

            # get data from best run
            best_trial = tuner.oracle.get_best_trials(1)[0].trial_id

            tuner_acc, tuner_val_acc, tuner_loss, tuner_val_loss = extract_history_from_tuner(best_trial, directory, model_trial_info['trial_id'])

            best_hyperparameters = tuner.get_best_hyperparameters(1)[0].values
            if isinstance(model_tune_param['p_learning_rate_bArray'], list):
                learning_rate = model_tune_param['p_learning_rate_bArray'][0]
            else:
                learning_rate =     model_tune_param['p_learning_rate_bArray']
            if isinstance(model_tune_param['p_optimiser_function_bArray'], list):
                optimiser_function = model_tune_param['p_optimiser_function_bArray'][0]
            else:
                optimiser_function = model_tune_param['p_optimiser_function_bArray']
            if isinstance(model_tune_param['p_Exp_decay_rate_momentum1_bArray'], list):
                momentum1 = model_tune_param['p_Exp_decay_rate_momentum1_bArray'][0]
            else:
                momentum1 = model_tune_param['p_Exp_decay_rate_momentum1_bArray']
            if isinstance(model_tune_param['p_Exp_decay_rate_momentum2_bArray'], list):
                momentum2 = model_tune_param['p_Exp_decay_rate_momentum2_bArray'][0]
            else:
                momentum2 = model_tune_param['p_Exp_decay_rate_momentum2_bArray']
            if isinstance(p_loss_function, list):
                loss_function = p_loss_function[0]
            else:
                loss_function = p_loss_function
            for key in best_hyperparameters.keys():
                if key == 'epochs': num_epochs = best_hyperparameters[key]
                if key == 'learning_rate': learning_rate = best_hyperparameters[key]
                if key == 'optimiser_func': optimiser_function = best_hyperparameters[key]
                if key == 'momentum_1': momentum1 = best_hyperparameters[key]
                if key == 'momentum_2': momentum2 = best_hyperparameters[key]
                if key == 'loss_func': loss_function = best_hyperparameters[key]

            # RESULTS
            acc += tuner_acc
            val_acc += tuner_val_acc
            loss += tuner_loss
            val_loss += tuner_val_loss
            model = tuner.get_best_models()[0]

            if model_trial_info['delet_logs'] and os.path.isdir(os.path.join(directory,model_trial_info['trial_id'])): shutil.rmtree(os.path.join(directory,model_trial_info['trial_id']) )

        else:
            ###############################################################################################################
            # Hyperparameter Tuning with a define set of parameters: [0] of each list.

            num_epochs = model_tune_param['p_num_epochs']
            if isinstance( model_tune_param['p_learning_rate_bArray'], list):
                learning_rate = model_tune_param['p_learning_rate_bArray'][0]
            else:
                learning_rate = model_tune_param['p_learning_rate_bArray']
            if isinstance(model_tune_param['p_optimiser_function_bArray'], list):
                optimiser_function = model_tune_param['p_optimiser_function_bArray'][0]
            else:
                optimiser_function = model_tune_param['p_optimiser_function_bArray']
            if isinstance(model_tune_param['p_Exp_decay_rate_momentum1_bArray'], list):
                momentum1 = model_tune_param['p_Exp_decay_rate_momentum1_bArray'][0]
            else:
                momentum1 = model_tune_param['p_Exp_decay_rate_momentum1_bArray']
            if isinstance(model_tune_param['p_Exp_decay_rate_momentum2_bArray'], list):
                momentum2 = model_tune_param['p_Exp_decay_rate_momentum2_bArray'][0]
            else:
                momentum2 = model_tune_param['p_Exp_decay_rate_momentum2_bArray']
            if isinstance(p_loss_function, list):
                loss_function = p_loss_function[0]
            else:
                loss_function = p_loss_function

            if model_info['w_transfer_init_required']:
                combined_num_epochs = num_epochs + model_tune_param['p_num_epochs_transfer_init']
                last_epoch = history.epoch[-1] + 1
                #model.set_weights(model_init_weights)
            else:
                last_epoch = 0
                combined_num_epochs = num_epochs

            model = compile_model(model, lr=learning_rate, momentum1=momentum1,
                                  momentum2=momentum2, optimiser_func=optimiser_function,
                                  loss_func=loss_function, class_freqs=class_freqs)

            history = model.fit(TrainingDataGen,
                                epochs=combined_num_epochs,
                                initial_epoch=last_epoch,
                                verbose=2,  # 0 = silent, 1 = progress bar, 2 = one line per epoch
                                steps_per_epoch=int(math.ceil(TrainingDataGen.dataset_qty / model_tune_param['p_Batch_size'])),
                                validation_steps=int(math.ceil(ValidationDataGen.dataset_qty / model_tune_param['p_Batch_size'])),
                                validation_data=ValidationDataGen,
                                shuffle=False,
                                validation_freq=1,
                                callbacks=[cb1])

            acc += history.history['acc']
            val_acc += history.history['val_acc']
            loss += history.history['loss']
            val_loss += history.history['val_loss']

        # print training & XVal results
        if model_trial_info['print_plots']:
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            if model_info['w_transfer_init_required']:
                plt.ylim([0.8, 1])
                plt.plot([model_tune_param['p_num_epochs_transfer_init'] - 1, model_tune_param['p_num_epochs_transfer_init'] - 1],
                         plt.ylim(), label='Unfreeze layers')
            plt.xticks(np.arange(len(acc)), np.arange(1, len(acc) + 1))
            plt.legend(loc='lower right')
            plt.title(model_trial_info['trial_id'])
            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            if model_info['w_transfer_init_required']:
                plt.ylim([0, 1.0])
                plt.plot([model_tune_param['p_num_epochs_transfer_init'] - 1, model_tune_param['p_num_epochs_transfer_init'] - 1],
                         plt.ylim(), label='Unfreeze layers')
            plt.xticks(np.arange(len(loss)), np.arange(1, len(loss) + 1))
            plt.legend(loc='upper right')
            plt.xlabel('epoch')
            plt.show()
            plt.close()

        model_run_info = {
            'trial_id': model_trial_info['trial_id']

        }
        model_dataset_info = {
            'dataset_info_name': model_dataset_info['dataset_info_name'],
            'dataset_split': model_dataset_info['dataset_split'],
            'dataset_v_class_filter': model_dataset_info['dataset_v_class_filter'],
            'dataset_exclude_imgs_in_class': model_dataset_info['dataset_exclude_imgs_in_class'],
            'dataset_Val_split_ratio': model_dataset_info['dataset_Val_split_ratio']
        }
        if 'dataset_test_split' in model_dataset_info:
            model_dataset_info.update({'dataset_test_split' : model_dataset_info['dataset_test_split']})

        model_tune_param = {
            'p_Batch_size': model_tune_param['p_Batch_size'],
            'p_model_name': model_tune_param['p_model_name'],
            'P_input_size': img_input_shape,
            'p_early_stop_patience': model_tune_param['p_early_stop_patience'],
            'p_augmentation': augment,
            'p_bayesian_max_trials': model_tune_param['p_bayesian_max_trials'],
            'p_num_epochs': num_epochs,
            'p_learning_rate': learning_rate,
            'p_optimiser_function': optimiser_function,
            'p_Exp_decay_rate_momentum1': momentum1,
            'p_Exp_decay_rate_momentum2': momentum2,
            'p_loss_function': loss_function,
            'P_transfer_weights': P_transfer_weights,
            'p_num_epochs_transfer_init': model_tune_param['p_num_epochs_transfer_init'],
            'p_learning_rate_transfer_init': model_tune_param['p_learning_rate_transfer_init'],
            'p_freez_first_layers_init': model_tune_param['p_freez_first_layers_init'],
            'p_transfer_model_to_FC': model_tune_param['p_transfer_model_to_FC'],
            'p_CNN_Depth': model_tune_param['p_CNN_Depth'],
            'p_conv_dropout': model_tune_param['p_conv_dropout'],
            'p_fc_dropout': model_tune_param['p_fc_dropout'],
            'p_pooling_cnv': model_tune_param['p_pooling_cnv'],
            'p_pooling_fc': model_tune_param['p_pooling_fc']
        }
        model_performance = {
            'acc': acc[-1],
            'val_acc': val_acc[-1],
            'loss': loss[-1],
            'val_loss': val_loss[-1],
        }

        model_info = {
            'trial_info': model_run_info,
            'dataset_info': model_dataset_info,
            'tune_param': model_tune_param,
            'performance': model_performance
        }

        return model, model_info
    else:
        return None, None


#model, model_info = tune_model(model_trial_info, model_dataset_info, model_tune_param)