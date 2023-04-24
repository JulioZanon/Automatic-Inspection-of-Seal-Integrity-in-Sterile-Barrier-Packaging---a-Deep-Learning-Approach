from __future__ import absolute_import, division, print_function, unicode_literals
from custom_lib.dataset_functions import *
from custom_lib.custom_CNNs.VGGbnV1 import VGGbnV1
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201,EfficientNetB0, EfficientNetB1, EfficientNetB2, \
    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, InceptionResNetV2,InceptionV3,MobileNetV2, \
    MobileNetV3Large, MobileNetV3Small, NASNetLarge, NASNetMobile, ResNet50V2, ResNet101V2,ResNet152V2, VGG16,VGG19,Xception
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.losses import * #BinaryCrossentropy, hinge, categorical_crossentropy, categorical_hinge, SparseCategoricalCrossentropy
layers = tf.keras.layers
from .parameter_serarch_functions import *
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils.data_utils import get_file
import tensorflow.keras as keras
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.utils import get_source_inputs
from tensorflow.compat.v1.image import resize_bilinear
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
import tensorflow.tools.compatibility.tf_upgrade_v2


"""
Activation functions: Always use first ReLU. If improvements are required, try Leaky, Maxout, ELu. (Don't use TanH or Sigmoid).
Loss: use Softmax (Crossentropy) or SVM (Hinge). Simillar performance.
Optimisers: Start with RMSprop (option of adding decay). A more complex function is ADAM, which combines RMSprop with momentum. SGD can work ok but can overshoot. Avoid all others.
"""

##############################################################################################################
############# COMMON to all models ###########################################################################
##############################################################################################################

# Dictionary for Keras transfer models and layer flattening design.
# Dictionaries are required to build tuning plans as these are build with numeric arrays and would not accept strings.


model_library_info = {
'DenseNet121': {
    'id': 1,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library', # Function used to acquire the model
    'label_type': 'class', # Classification vs Segmentation. Segmentation models can be used for classification
    'weights': ['imagenet'], #Transfer weights.
    'w_transfer_init_required': True, #Transfer learning requires inititaliseing learning with base model frozen.
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],#Dictionary keys for custom model arguments
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'DenseNet169': {
    'id': 2,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'DenseNet201': {
    'id': 3,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB0': {
    'id': 4,
    'min_input_size': [224, 224, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB1': {
    'id': 5,
    'min_input_size': [240, 240, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB2': {
    'id': 6,
    'min_input_size': [260, 260, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB3': {
    'id': 7,
    'min_input_size': [300, 300, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB4': {
    'id': 8,
    'min_input_size': [380, 380, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB5': {
    'id': 9,
    'min_input_size': [456, 456, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB6': {
    'id': 10,
    'min_input_size': [528, 528, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'EfficientNetB7': {
    'id': 11,
    'min_input_size': [600, 600, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet', 'noisy-student'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'InceptionResNetV2': {
    'id': 12,
    'min_input_size': [75, 75, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'InceptionV3': {
    'id': 13,
    'min_input_size': [75, 75, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'MobileNetV2': {
    'id': 14,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'MobileNetV3Large': {
    'id': 15,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'MobileNetV3Small': {
    'id': 16,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'ResNet50V2': {
    'id': 19,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'ResNet101V2': {
    'id': 20,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'ResNet152V2': {
    'id': 21,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'VGG16': {
    'id': 22,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'VGG19': {
    'id': 23,
    'min_input_size': [32, 32, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'Xception': {
    'id': 24,
    'min_input_size': [71, 71, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': ['freeze_layers', 'transfer_model_to_FC'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': ['GlobalMaxPooling2D', 'GlobalAveragePooling2D']},
'Unet_mobileNetV2': {
    'id': 25,
    'min_input_size': [224, 224, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'mask',
    'weights': ['imagenet'],
    'w_transfer_init_required': False,
    'required_arguments': None,
    'loss_functions': ['soft_dice_loss', 'SparseCategoricalCrossentropy'],
    'transfer_model_to_FC': None},
'Unet_VGG16': {
    'id': 26,
    'min_input_size': [64, 256, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'mask',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': None,
    'loss_functions': ['soft_dice_loss', 'SparseCategoricalCrossentropy'],
    'transfer_model_to_FC': None},
'Unet_VGG19': {
    'id': 27,
    'min_input_size': [224, 224, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'mask',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': None,
    'loss_functions': ['soft_dice_loss', 'SparseCategoricalCrossentropy', 'BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC': None},
'Unet_ResNet50V2': {
    'id': 28,
    'min_input_size': [224, 224, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'mask',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': None,
    'loss_functions': ['soft_dice_loss', 'SparseCategoricalCrossentropy'],
    'transfer_model_to_FC': None},
'Unet_DenseNet121': {
    'id': 29,
    'min_input_size': [224, 224, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'mask',
    'weights': ['imagenet'],
    'w_transfer_init_required': True,
    'required_arguments': None,
    'loss_functions': ['soft_dice_loss', 'SparseCategoricalCrossentropy'],
    'transfer_model_to_FC': None},
'deeplabV3': {
    'id': 30,
    'min_input_size': [64, 512, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'mask',
    'weights': ['pascal_voc', 'cityscapes'],
    'w_transfer_init_required': True,
    'required_arguments': None,
    'loss_functions': ['SparseCategoricalCrossentropy'],
    'transfer_model_to_FC': None},
'VGGbnV1': {
    'id': 31,
    'min_input_size': [64, 64, 3],
    'source_code': 'get_model_from_library',
    'label_type': 'class',
    'weights': [None],
    'w_transfer_init_required': False,
    'required_arguments': ['VGGbnV1_depth', 'VGGbnV1_conv_dropout', 'VGGbnV1_fc_dropout', 'VGGbnV1_activation', 'VGGbnV1_pooling_cnv', 'VGGbnV1_pooling_fc', 'VGGbnV1_initializer'],
    'loss_functions': ['BinaryCrossentropy', 'BinaryCrossentropy_weighted'],
    'transfer_model_to_FC':None}
}



dict_flatten_layers = {
        'Flatten': 1,
        'GlobalAveragePooling2D': 2,
        'GlobalMaxPooling2D': 3,
        'NA' : 4
    }

dict_optimiser_function = {
    'RMSprop': 1,
    'Adam':2,
    'SGD': 3
}

dict_loss_function = {
    'BinaryCrossentropy': 1,
    'hinge': 2,
    'SparseCategoricalCrossentropy': 3,
    'categorical_crossentropy': 4,
    'categorical_hinge': 5,
    'soft_dice_loss': 6,
    'BinaryCrossentropy_weighted':7
}

dict_transfer_weights = {
    'imagenet': 1,
    'pascal_voc' : 2,
    'cityscapes' : 3,
    'None' : 4
}

dict_pooling_layers = {
    'MaxPooling2D' : 1,
    'AveragePooling2D' : 2,
    'GlobalMaxPooling2D' : 3,
    'GlobalAveragePooling2D' : 4
}


def calculate_model_inp_size(model_min_inp_size = [0, 0, 0], target_inp_size = [0, 0, 0] ):
    if len(model_min_inp_size) != 3 or len(target_inp_size) != 3: raise RuntimeError('The input size array must contain 3 elements, [H, W, Channels]')
    if target_inp_size[0] < model_min_inp_size[0] or target_inp_size[1] < model_min_inp_size[1]: raise RuntimeError(
        'Target Input size is smaller that model minimum size')

    target_inp_size_copy = target_inp_size.copy()
    inp_size = [0, 0, 0]

    if (model_min_inp_size[1] / model_min_inp_size[0]) != (target_inp_size[1] / target_inp_size[0]):
        if model_min_inp_size[0] > model_min_inp_size[1]:
            target_inp_size_copy[1] = int(np.rint(target_inp_size[0] * (model_min_inp_size[1] / model_min_inp_size[0])))
        elif model_min_inp_size[1] > model_min_inp_size[0]:
            target_inp_size_copy[0] = int(np.rint(target_inp_size[1] * (model_min_inp_size[0] / model_min_inp_size[1])))

    inp_size[0] = int(np.rint(target_inp_size_copy[0] / model_min_inp_size[0]) * model_min_inp_size[0])
    inp_size[1] = int(np.rint(target_inp_size_copy[1] / model_min_inp_size[1]) * model_min_inp_size[1])
    inp_size[2] = int(model_min_inp_size[2])
    return inp_size



def compute_class_freqs(num_of_samples_per_calss_array):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """

    positive_frequencies = np.array(num_of_samples_per_calss_array)
    N = np.sum(positive_frequencies)
    negative_frequencies = N - positive_frequencies

    return positive_frequencies / N, negative_frequencies / N




def get_num_of_layers(model):
    num_of_layers = len(model.layers)
    not_trainable_layers = 0
    for layer in model.layers:
        if layer.trainable:
            break
        else: not_trainable_layers += 1

    return num_of_layers, not_trainable_layers

def get_custom_loss_function(CustomLoss = ''):
    if CustomLoss == 'BinaryCrossentropy_weighted':
        loss = get_BinaryCrossentropy_weighted
        metrics = ['accuracy']
    elif CustomLoss == 'soft_dice_loss':
        loss = soft_dice_loss
        metrics = ['accuracy', dice_coefficient]
    else: raise RuntimeError('Custom loss function: [' + CustomLoss + '] not supported.')
    return loss, metrics



def convert_seg_mask_to_classification_array(mask_array, conv_type = 'Thresholding', ThresholdNormZeroToOne = 0.0, norm_pix_img: dict = None):
    """
    This function convert segmentation arrays into list arrays with summary scores to be used
    to perform classification tasks.
    :param mask_array: Array to be converted with shape [idx, H, W, class]
    :param conv_type: Type of converstion ('Thresholding', 'Sum&Sum&NormZeroToOne', 'blobSize&NormZeroToOne')
    :return: Converted array with shape [idx, convertion value, class]
    """

    # Normalise predictions
    if norm_pix_img is None:
        norm_min_pix = float(np.min(mask_array))
        norm_ptp_pix = float(np.ptp(mask_array))
    else:
        norm_min_pix = norm_pix_img['norm_min_pix']
        norm_ptp_pix = norm_pix_img['norm_ptp_pix']

    mask_array = (mask_array - norm_min_pix)/norm_ptp_pix
    classification_array = mask_array.sum(axis=(1, 2))

    if norm_pix_img is None:
        norm_min_img = float(np.min(classification_array))
        norm_ptp_img = float(np.ptp(classification_array))
    else:
        norm_min_img = norm_pix_img['norm_min_img']
        norm_ptp_img = norm_pix_img['norm_ptp_img']

    if conv_type == 'Thresholding':
        classification_array[classification_array <= ThresholdNormZeroToOne] = 0.0
        classification_array[classification_array > ThresholdNormZeroToOne] = 1.0
    elif conv_type == 'Sum&NormZeroToOne':
        classification_array = (classification_array - norm_min_img)/norm_ptp_img
    elif conv_type == 'blobSize&NormZeroToOne':
        #TODO: develop function with open CV to score blob size IT MAY BE TO SLOW ...
        pass
    norm_pred_mask = {
        'norm_min_pix': norm_min_pix,
        'norm_ptp_pix': norm_ptp_pix,
        'norm_min_img': norm_min_img,
        'norm_ptp_img': norm_ptp_img
    }
    return classification_array, norm_pred_mask



########################################################################################################
##############  LOSS FUNCTIONS
########################################################################################################
def dice_coefficient(y_true, y_pred, epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.
    DC(f,x,y)=1/N * [c=1 to C]∑(DCc(f,x,y)) where f of x gives the predicted lable.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.
    """
    axis = tuple(range(1, len(y_pred.shape) - 1))
    dice_numerator = (2 * K.sum(y_pred * y_true, axis=axis))  +  epsilon
    dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) +  epsilon
    dice_coefficient = K.mean(dice_numerator / dice_denominator)

    return dice_coefficient

def soft_dice_loss(y_true, y_pred, epsilon=0.00001):
    """
    Compute mean soft dice loss.
    analogue interpretation of dice_coefficient by using probabilities.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.
    """
    axis = tuple(range(1, len(y_pred.shape) - 1))
    dice_numerator = (2 * K.sum(y_pred * y_true, axis=axis))  +  epsilon
    dice_denominator = K.sum(y_pred**2, axis=axis) + K.sum(y_true**2, axis=axis) +  epsilon
    dice_loss = 1 - (K.mean(dice_numerator / dice_denominator))


    return dice_loss

def get_BinaryCrossentropy_weighted(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """

    def BinaryCrossentropy_weighted(y_true, y_pred):
        """
        Return weighted loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        y_true = tf.dtypes.cast(y_true, tf.float32)
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss += (-1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(epsilon + y_pred[:, i]))) + \
                    (-1 * K.mean(
                        neg_weights[i] * (1 - y_true[:, i]) * K.log(epsilon + 1 - y_pred[:, i])))
        return loss

    return BinaryCrossentropy_weighted

##############################################################################################################
############# CLASSIFICATION models ##########################################################################
##############################################################################################################

def get_model_from_library(base_model_name='MobileNetV2', img_shape=(150, 150, 3), weights='imagenet', num_of_classes = 2, args = None):
    # Function return kera based models

    if base_model_name == 'DenseNet121':
        base_model = DenseNet121(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='DenseNet169':
        base_model = DenseNet169(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='DenseNet201':
        base_model = DenseNet201(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB1':
        base_model = EfficientNetB1(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB2':
        base_model = EfficientNetB2(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB4':
        base_model = EfficientNetB4(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB5':
        base_model = EfficientNetB5(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB6':
        base_model = EfficientNetB6(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='InceptionResNetV2':
        base_model = InceptionResNetV2(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='InceptionV3':
        base_model = InceptionV3(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='MobileNetV2':
        base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='MobileNetV3Large':
        base_model = MobileNetV3Large(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='MobileNetV3Small':
        base_model = MobileNetV3Small(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='NASNetLarge':
        base_model = NASNetLarge(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='NASNetMobile':
        base_model = NASNetMobile(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='ResNet50V2':
        base_model = ResNet50V2(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='ResNet101V2':
        base_model = ResNet101V2(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='ResNet152V2':
        base_model = ResNet152V2(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='VGG16':
        base_model = VGG16(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='VGG19':
        base_model = VGG19(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='Xception':
        base_model = Xception(input_shape=img_shape, include_top=False, weights=weights)
        if args['freeze_layers']: base_model.trainable = False
        base_model = add_custom_layers(base_model, num_of_classes=num_of_classes, to_FC=args['transfer_model_to_FC'])
    elif base_model_name=='Unet_mobileNetV2':
        base_model = get_Unet_model(model=base_model_name, input_shape=img_shape, classes=num_of_classes,
                       weights=weights, freeze_transfer_weights=True)
    elif base_model_name=='Unet_VGG16':
        base_model = get_Unet_model(model=base_model_name, input_shape=img_shape, classes=num_of_classes,
                       weights=weights, freeze_transfer_weights=True)
    elif base_model_name=='Unet_VGG19':
        base_model = get_Unet_model(model=base_model_name, input_shape=img_shape, classes=num_of_classes,
                       weights=weights, freeze_transfer_weights=True)
    elif base_model_name=='Unet_ResNet50V2':
        base_model = get_Unet_model(model=base_model_name, input_shape=img_shape, classes=num_of_classes,
                       weights=weights, freeze_transfer_weights=True)
    elif base_model_name=='Unet_DenseNet121':
        base_model = get_Unet_model(model=base_model_name, input_shape=img_shape, classes=num_of_classes,
                       weights=weights, freeze_transfer_weights=True)
    elif base_model_name == 'deeplabV3':
        base_model = deeplab_v3_plus(input_shape=img_shape, classes=num_of_classes)
    elif base_model_name == 'VGGbnV1':
        if args is None: raise RuntimeError('Custom mdel ' + base_model_name +  ' requires argumantes but none was passed')
        base_model = VGGbnV1(input_shape=img_shape, classes=num_of_classes, depth=args['VGGbnV1_depth'], conv_dropout=args['VGGbnV1_conv_dropout'],
                        fc_dropout=args['VGGbnV1_fc_dropout'], activation=args['VGGbnV1_activation'], pooling_cnv=args['VGGbnV1_pooling_cnv'], pooling_fc= args['VGGbnV1_pooling_fc'],
                        initializer=args['VGGbnV1_initializer'])
    else: raise RuntimeError('Selected model: ' + base_model_name + ' is not supported by get_model_from_library')


    return base_model


def add_custom_layers(base_model, num_of_classes=2, to_FC ='GlobalMaxPooling2D'):
    """
    :param base_model: Transfer model from modelHub without the top layers
    :param num_of_classes: number of classess in the label array
    :param to_FC: 'GlobalMaxPooling2D', 'GlobalAvgPooling2D', 'layers.Flatten'
    :return: model
    """
    x = base_model.output

    if to_FC =='GlobalMaxPooling2D':
        x = layers.GlobalMaxPooling2D()(x)
    elif to_FC =='GlobalAveragePooling2D':
        x = layers.GlobalAveragePooling2D()(x)
    elif to_FC == 'Flatten':
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
    if num_of_classes == 1:
        x = layers.Dense(1)(x)
        prediction = layers.Activation(activation='sigmoid')(x)
    else:
        prediction = layers.Dense(num_of_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=base_model.input, outputs=prediction)

##############################################################################################################
############# SEGMENTATION models ##########################################################################
##############################################################################################################


def get_Unet_model(model = 'Unet_mobileNetV2', input_shape = [], classes = 2, weights = 'imagenet', freeze_transfer_weights=True):
    """
    This function selects from all available models. If adding new models, ensure that all model names ara added to the
    dictionary 'transfer_models' so that they can be called during hyper parameter searches.
    :param model: Model name
    :param input_shape: [H x W x channels]
    :param classes: number of classes.
    :param weights: transfer weights.
    :param freeze_transfer_weights: True / False
    :return: Model
    """
    if model == 'Unet_mobileNetV2':
        model = Unet_mobileNetV2(input_shape = input_shape, classes = classes, weights=weights)
    if model == 'Unet_VGG16':
        model = Unet_VGG16(input_shape = input_shape, classes = classes, weights=weights)
    if model == 'Unet_VGG19':
        model = Unet_VGG19(input_shape = input_shape, classes = classes, weights=weights)
    if model == 'Unet_ResNet50V2':
        model = Unet_ResNet50V2(input_shape = input_shape, classes = classes, weights=weights)
    if model == 'Unet_DenseNet121':
        model = Unet_DenseNet121(input_shape = input_shape, classes = classes, weights=weights)
    if not freeze_transfer_weights:
        model.trainable = True

    return model


def unet_model(down_stack, up_stack, input_shape: tuple = (224, 224, 3), output_channels: int = 2):
    """
    This function assembles U-Net models. For the function to work, Downsatack layers must be correctlly selected to
    match the up_stack layeres so that skips can be implemented accordangly.
    :param down_stack: Encoder Layers from transfer model that would be connected to the decoder.
    :param up_stack:  Decoder layers
    :param input_shape: [H x W x channels]
    :param output_channels: number of classes.
    :return: final Unet model ready to be compiled.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1] # Bottommost layer from the defined skip layers
    skips = reversed(skips[:-1]) # all other layers defined as skip in revers order

    # Upsampling and establishing the skip connections between downstack and upstack
    for up, skip in zip(up_stack, skips):
        x = up(x) # Conv2DTranspose => Batchnorm => Dropout => Relu
        concat = layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model by upsampling with num_filters = classess
    last = layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)



def Unet_mobileNetV2(input_shape: tuple = (224, 224, 3), classes: int = 2, weights='imagenet'):
    """
    Unet implmentation of [mobileNetV2] as oer Tensorflow 2.0 tutorial approach.
    https://www.tensorflow.org/tutorials/images/segmentation
    :param input_shape: (HxWxCh)
    :param classes: Number of classes
    :param weights: Transfer Weights
    :return: Model ready to be UNet-assembled with function unet_model()
    """
    #Define base transfer model (size in comments ara based on 224,224)
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights = weights)

    # ENCODER (Selection of layers within the origianl model)
    # Define layers that will be connected to the Decoder in the Upstack
    # Use names from the TF model. By convention, select half steps
    skip_layer_names = [
        'block_1_expand_relu',   # inp/2 x inp/2 x 96 filters (112x112x96)
        'block_3_expand_relu',   # inp/4 x inp/4 x 144 filters (56x56x144)
        'block_6_expand_relu',   # inp/8 x inp/8 x 192 filters (28x28x192)
        'block_13_expand_relu',  # inp/16 x inp/16 x 576 filters (14x14x576)
        'block_16_project',      # inp/32 x inp/32 x 320 filters (7x7x320)
    ]

    # Get defined layers from base model
    layers = [base_model.get_layer(name).output for name in skip_layer_names]
    # Create the Encoder model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    ##DECODER/UPSTACK
    # Build the model by upsampling. This is done with the pix2pix.upsample(num_of_filters, Kernel_size) function which
    # increase feature map size x2 by applying the following steps: Conv2DTranspose => Batchnorm => Dropout => Relu, where
    # Note that the final dimension in the upstack should be inp/2 as the unet_model funcion will add the last upsample step
    # to generate the final classes
    up_stack = [
        pix2pix.upsample(512, 3),  # Bottommost x 2 with 512 filters (7x7 -> 14x14)
        pix2pix.upsample(256, 3),  # Bottommost x 4 with 256 filters (14x14 -> 28x28)
        pix2pix.upsample(128, 3),  # Bottommost x 8 with 128 filters (28x28 -> 56x56)
        pix2pix.upsample(64, 3),   # Bottommost x 16 with 64 filters (56x56 -> 112x112)
    ]
    Unet_model = unet_model(down_stack, up_stack, input_shape, classes)
    # add classifier
    model = tf.keras.Sequential([
        Unet_model,
        tf.keras.layers.Activation('softmax')
    ])
    return model


def Unet_VGG16(input_shape: tuple = (224, 224, 3), classes: int = 2, weights='imagenet'):
    """
    Unet implmentation of [VGG16] as oer Tensorflow 2.0 tutorial approach.
    https://www.tensorflow.org/tutorials/images/segmentation
    :param input_shape: (HxWxCh)
    :param classes: Number of classes
    :param weights: Transfer Weights
    :return: Model ready to be UNet-assembled with function unet_model()
    """
    #Define base transfer model (size in comments ara based on 224,224)
    base_model = VGG16(input_shape=input_shape, include_top=False, weights = weights)

    # ENCODER (Selection of layers within the origianl model)
    skip_layer_names = [
        'block2_conv2',   # inp/2 x inp/2  (112x112)
        'block3_conv3',   # inp/4 x inp/4  (56x56)
        'block4_conv3',   # inp/8 x inp/8  (28x28)
        'block5_conv3',  # inp/16 x inp/16 (14x14)
    ]

    # Get defined layers from base model
    layers = [base_model.get_layer(name).output for name in skip_layer_names]
    # Create the Encoder model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    ##DECODER/UPSTACK
    up_stack = [
        pix2pix.upsample(256, 3),  # Bottommost x 4 with 256 filters (14x14 -> 28x28)
        pix2pix.upsample(128, 3),  # Bottommost x 8 with 128 filters (28x28 -> 56x56)
        pix2pix.upsample(64, 3),   # Bottommost x 16 with 64 filters (56x56 -> 112x112)
    ]
    Unet_model = unet_model(down_stack, up_stack, input_shape, classes)
    # add classifier
    model = tf.keras.Sequential([
        Unet_model,
        tf.keras.layers.Activation('softmax')
    ])
    return model

def Unet_VGG19(input_shape: tuple = (224, 224, 3), classes: int = 2, weights='imagenet'):
    """
    Unet implmentation of [VGG19] as oer Tensorflow 2.0 tutorial approach.
    https://www.tensorflow.org/tutorials/images/segmentation
    :param input_shape: (HxWxCh)
    :param classes: Number of classes
    :param weights: Transfer Weights
    :return: Model ready to be UNet-assembled with function unet_model()
    """
    #Define base transfer model (size in comments ara based on 224,224)
    base_model = VGG19(input_shape=input_shape, include_top=False, weights = weights)

    # ENCODER (Selection of layers within the origianl model)
    skip_layer_names = [
        'block2_conv2',   # inp/2 x inp/2  (112x112)
        'block3_conv4',   # inp/4 x inp/4  (56x56)
        'block4_conv4',   # inp/8 x inp/8  (28x28)
        'block5_conv4',  # inp/16 x inp/16 (14x14)
    ]

    # Get defined layers from base model
    layers = [base_model.get_layer(name).output for name in skip_layer_names]
    # Create the Encoder model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    ##DECODER/UPSTACK
    up_stack = [
        pix2pix.upsample(256, 3),  # Bottommost x 4 with 256 filters (14x14 -> 28x28)
        pix2pix.upsample(128, 3),  # Bottommost x 8 with 128 filters (28x28 -> 56x56)
        pix2pix.upsample(64, 3),   # Bottommost x 16 with 64 filters (56x56 -> 112x112)
    ]
    Unet_model = unet_model(down_stack, up_stack, input_shape, classes)
    # add classifier
    model = tf.keras.Sequential([
        Unet_model,
        tf.keras.layers.Activation('softmax')
    ])
    return model


def Unet_ResNet50V2(input_shape: tuple = (224, 224, 3), classes: int = 2, weights='imagenet'):
    """
    Unet implmentation of [ResNet50V2] as oer Tensorflow 2.0 tutorial approach.
    https://www.tensorflow.org/tutorials/images/segmentation
    :param input_shape: (HxWxCh)
    :param classes: Number of classes
    :param weights: Transfer Weights
    :return: Model ready to be UNet-assembled with function unet_model()
    """
    #Define base transfer model (size in comments ara based on 224,224)
    base_model = ResNet50V2(input_shape=input_shape, include_top=False, weights = weights)

    # ENCODER (Selection of layers within the origianl model)
    skip_layer_names = [
        'conv1_conv',   # inp/2 x inp/2 x 96 filters (112x112)
        'conv2_block3_1_relu',   # inp/4 x inp/4  (56x56)
        'conv3_block4_1_relu',   # inp/8 x inp/8  (28x28)
        'conv4_block6_1_relu',  # inp/16 x inp/16 (14x14)
        'conv5_block3_3_conv',  # inp/32 x inp/32 (7x7)
    ]

    # Get defined layers from base model
    layers = [base_model.get_layer(name).output for name in skip_layer_names]
    # Create the Encoder model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    ##DECODER/UPSTACK
    up_stack = [
        pix2pix.upsample(512, 3),  # Bottommost x 2 with 512 filters (7x7 -> 14x14)
        pix2pix.upsample(256, 3),  # Bottommost x 4 with 256 filters (14x14 -> 28x28)
        pix2pix.upsample(128, 3),  # Bottommost x 8 with 128 filters (28x28 -> 56x56)
        pix2pix.upsample(64, 3),   # Bottommost x 16 with 64 filters (56x56 -> 112x112)
    ]
    Unet_model = unet_model(down_stack, up_stack, input_shape, classes)
    # add classifier
    model = tf.keras.Sequential([
        Unet_model,
        tf.keras.layers.Activation('softmax')
    ])
    return model


def Unet_DenseNet121(input_shape: tuple = (224, 224, 3), classes: int = 2, weights='imagenet'):
    """
    Unet implmentation of [DenseNet121] as oer Tensorflow 2.0 tutorial approach.
    https://www.tensorflow.org/tutorials/images/segmentation
    :param input_shape: (HxWxCh)
    :param classes: Number of classes
    :param weights: Transfer Weights
    :return: Model ready to be UNet-assembled with function unet_model()
    """
    # Define base transfer model (size in comments ara based on 224,224)
    base_model = DenseNet121(input_shape=input_shape, include_top=False, weights=weights)

    # ENCODER (Selection of layers within the origianl model)
    skip_layer_names = [
        'conv1/relu',  # inp/2 x inp/2  (112x112)
        'pool2_relu',  # inp/4 x inp/4 (56x56)
        'pool3_relu',  # inp/8 x inp/8 (28x28)
        'pool4_relu',  # inp/16 x inp/16 (14x14)
        'conv5_block16_1_relu',  # inp/32 x inp/32  (7x7)
    ]

    # Get defined layers from base model
    layers = [base_model.get_layer(name).output for name in skip_layer_names]
    # Create the Encoder model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    ##DECODER/UPSTACK
    up_stack = [
        pix2pix.upsample(512, 3),  # Bottommost x 2 with 512 filters (7x7 -> 14x14)
        pix2pix.upsample(256, 3),  # Bottommost x 4 with 256 filters (14x14 -> 28x28)
        pix2pix.upsample(128, 3),  # Bottommost x 8 with 128 filters (28x28 -> 56x56)
        pix2pix.upsample(64, 3),  # Bottommost x 16 with 64 filters (56x56 -> 112x112)
    ]
    Unet_model = unet_model(down_stack, up_stack, input_shape, classes)
    # add classifier
    model = tf.keras.Sequential([
        Unet_model,
        tf.keras.layers.Activation('softmax')
    ])
    return model


def deeplab_v3_plus(input_shape: tuple = (512, 512, 3), classes: int = 2, weights = 'pascal_voc',
                    input_tensor=None, OS: int = 16, backbone = 'xception'):
#TODO: this function does not work for OS = 8
#TODO: This function only works with xception backbone and ''pascal_voc
    """
    This function creates a Deeplabv3+ segmentation model based on the architecture documented here:
    https://github.com/MLearing/Keras-Deeplab-v3-plus/blob/master/model.py

    Args:
        :param input_shape: Shape of image data passed to model
        :param classes: Number of classes to use in the model
        :param weights: one of 'pascal_voc' (pre-trained on pascal voc) or None (random initialization)
            if using pascal_voc weights, use keras.applications.imagenet_utils.preprocess_input(x, mode='tf')
        :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the
            model.
        :param OS: determines input_shape/feature_extractor_output ratio. One of {8,16}
        :param final_layer_activation: Final layer activation function type. Options are 'sigmoid' and 'softmax'

    Returns:
        :return: A keras model instance
    """
    if not (weights in {'pascal_voc', None, 'cityscapes'}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `pascal_voc` '
                         '(pre-trained on PASCAL VOC) or cityscapes (pre-trained on cityscapes)')
    if not (backbone in {'xception'}):
        raise ValueError('Only  xception bacbone is currently available')

    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    #                                                                                       input shape (512, 512, 3) [OS=16]
    x = layers.Conv2D(32, (3, 3), strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    #                                                                                       input shape (256, 256, 32) [OS=16]
    x = layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = layers.Activation('relu')(x)

    x = conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = layers.Activation('relu')(x)
    #                                                                                       input shape (256, 256, 32) [OS=16]
    x = xception_block(x, [128, 128, 128], 'entry_flow_block1',
                       skip_connection_type='conv', stride=2,
                       depth_activation=False)
    #                                                                                       input shape (128, 128, 128) [OS=16]
    x, skip1 = xception_block(x, [256, 256, 256], 'entry_flow_block2',
                              skip_connection_type='conv', stride=2,
                              depth_activation=False, return_skip=True)
    #                                                                                       input shape (64, 64, 256) [OS=16]
    x = xception_block(x, [728, 728, 728], 'entry_flow_block3',
                       skip_connection_type='conv', stride=entry_block3_stride,
                       depth_activation=False)
    #                                                                                       input shape (32, 32, 728) [OS=16]
    for i in range(16):
        x = xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                           skip_connection_type='sum', stride=1, rate=middle_block_rate,
                           depth_activation=False)
    #                                                                                       input shape (32, 32, 728) [OS=16]
    x = xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                       skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                       depth_activation=False)
#                                                                                           input shape (32, 32, 1024) [OS=16]
    x = xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                       skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                       depth_activation=True)
    # end of feature extractor
    #                                                                                       input shape (32, 32, 2048) [OS=16]
    # branching for Atrous Spatial Pyramid Pooling
    # simple 1x1
    b0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = layers.Activation('relu', name='aspp0_activation')(b0)

    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # Image Feature branch
    out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = layers.AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = layers.Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = layers.Activation('relu')(b4)
    #                                                                                       input shape (8, 8) [OS=16]
    b4 = layers.UpSampling2D((out_shape, out_shape), interpolation='bilinear')(b4)
    #b4 = BilinearUpsampling((out_shape, out_shape))(b4) JZ: custom function changed to UpSampling2D to allow for model to be saved.
    #                                                                                       input shape (32, 32, 256) [OS=16]
    # concatenate ASPP branches & project
    x = layers.Concatenate()([b4, b0, b1, b2, b3])
    x = layers.Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)

    # DeepLab v.3+ decoder

    # Feature projection
    # x4 (x2) block


    x = layers.UpSampling2D((out_shape, out_shape), interpolation='bilinear')(x)  # 1024, 1024, 256
    #TODO: This change may not work with OS = 8 as the image scaling rate may be incorrect in above function
    #x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),int(np.ceil(input_shape[1] / 4))))(x) #JZ: custom function changed to UpSampling2D to allow for model to be saved.


    dec_skip1 = layers.Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = layers.BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = layers.Activation('relu')(dec_skip1) # 128, 128, 48
    x = layers.Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if classes == 21:
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = layers.Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    x = layers.UpSampling2D((out_shape, out_shape),interpolation='bilinear')(x)
    # TODO: This change may not work with OS = 8 as the image scaling rate may be incorrect in above function
    #x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x) JZ: custom function changed to UpSampling2D to allow for model to be saved.

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='deeplabv3_plus')

    # load weights
    weights_path = ''
    if backbone == 'xception':
        weights_file_name = 'deeplabv3_xception.h5'
    elif backbone == 'mobilenetv2':
        weights_file_name = 'deeplabv3_mobilenetv2.h5'

    if weights == 'pascal_voc':
        weights_path = os.path.join(os.getcwd(), 'weights\\pascal_voc', weights_file_name)
    elif weights == 'cityscapes':
        weights_path = os.path.join(os.getcwd(), 'weights\\cityscapes', weights_file_name)

    if weights_path is not None and weights_path != '': model.load_weights(weights_path, by_name=True)
    return model

def conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        return layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)

def xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                   rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                               kernel_size=1,
                               stride=stride)
        shortcut = layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

#layers.SeparableConv2D
def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)
    return x


class BilinearUpsampling(layers.Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = layers.InputSpec(ndim=4)
        if output_size:
            self.upsample_size = conv_utils.normalize_tuple(
                output_size, 2, 'size')
            self.upsampling = None
        else:
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.upsample_size[0]
            width = self.upsample_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return resize_bilinear(inputs, (self.upsample_size[0],
                                                       self.upsample_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Create a class to allow for multi-gpu model model-checkpointing where checkpoint results can still be loaded on
# CPU or single GPU machines (fixing bad design decision by keras to save multi-gpu models in a paralellized format
# that can only be loaded by machines with the same number of gpus as the training machine)
class AltModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, alternate_model, **kwargs):
        """
        Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
        :param filepath:
        :param alternate_model: Keras model to save instead of the default. This is used especially when training multi-
                                gpu models built with Keras multi_gpu_model(). In that case, you would pass the original
                                "template model" to be saved each checkpoint.
        :param kwargs:          Passed to ModelCheckpoint.
        """

        self.alternate_model = alternate_model
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before


def check_GPUs_availibility(gpu_id = -1):
    gpus = tf.config.list_physical_devices('GPU')
    GPU_count = len(tf.config.list_physical_devices('GPU'))
    keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False

    if gpu_id != -1 :
        if gpu_id < GPU_count:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        else:
            warnings.warn("Specified GPU is not reachable. GPU 0 will be used instead")
    else:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs available,", len(logical_gpus), "Logical GPU in use")
    return GPU_count