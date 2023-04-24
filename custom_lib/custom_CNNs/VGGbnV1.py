from __future__ import print_function
from __future__ import absolute_import

#import warnings
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs
layers = tf.keras.layers
#from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
#from keras.layers import Flatten, Dense, Input, Dropout
#from keras.layers import Convolution2D, MaxPooling2D
#from tensorflow.keras.utils import convert_all_kernels_in_model
from keras_applications.imagenet_utils import _obtain_input_shape


def conv_block(units, dropout=0.2, activation='relu', block=1, layer=1, initializer = 'zeros'):

    def layer_wrapper(inp):
        #TODO: try Deepwish Separable Conv form tf or Keras or custom (deepLab). keras.layers.SeparableConv2D or tf.layers.separable_conv2d
        # TODO: try dilated convolutions (atrous) insntead of Cnv2D of 3,3. It may be interesting for the first block
        # or as an input option if images are larger than x,y size.
        x = layers.Conv2D(units, (3, 3), padding='same', name='block{}_conv{}'.format(block, layer), kernel_initializer=initializer)(inp)
        x = layers.BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
        x = layers.Activation(activation, name='block{}_act{}'.format(block, layer))(x)
        x = layers.Dropout(dropout, name='block{}_dropout{}'.format(block, layer))(x)
        return x

    return layer_wrapper

def pool_block(pooling = 'MaxPooling2D', block=1):
    def layer_wrapper(inp):
        if pooling == 'MaxPooling2D':
            x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block{}_pool'.format(block))(inp)
        elif pooling == 'AveragePooling2D':
            x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block{}_pool'.format(block))(inp)
        return x
    return layer_wrapper

def dense_block(units, dropout=0.2, activation='relu', name='fc1', initializer = 'zeros'):

    def layer_wrapper(inp):
        x = layers.Dense(units, name=name, kernel_initializer=initializer)(inp)
        x = layers.BatchNormalization(name='{}_bn'.format(name))(x)
        x = layers.Activation(activation, name='{}_act'.format(name))(x)
        x = layers.Dropout(dropout, name='{}_dropout'.format(name))(x)
        return x
    return layer_wrapper

"""def get_initialiser(keras_fn_name = 'Glorot_Normal'):
    if keras_fn_name == 'glorot_normal':
        initializer = tf.keras.initializers.GlorotNormal() # Xavier
    elif keras_fn_name == 'glorot_uniform':
        initializer = tf.keras.initializers.GlorotUniform()
    else:
        initializer = None
    return initializer"""


def VGGbnV1(include_top=True, weights=None, input_tensor=None, input_shape=None,
            classes = 2, depth = 0,  conv_dropout=0.1, fc_dropout=0.3, activation='relu',
            pooling_cnv ='MaxPooling2D', pooling_fc = 'GlobalAveragePooling2D',
            initializer = 'zeros'):
    """
    VGG style model with bach normalisation and dropouts.
    :param include_top: True/False: Whether to include Fully connected
    :param weights: TODO
    :param input_tensor:
    :param input_shape:
    :param classes: Number of binary classes
    :param depth: int with values: (0): VGG11, (1): VGG16, (2): VGG19, (3): VGG26
    :param conv_dropout:  int. Dropout probability for Conv blocks (0-1)
    :param fc_dropout: int. Dropout probability for FC blocks (0-1)
    :param activation: Activation (Keras)
    :param pooling_cnv: Pooling in conv blocks - Keras: 'MaxPooling2D' or 'AveragePooling2D'
    :param pooling_fc = Pooling in Fully Connected layers - Keras: 'GlobalMaxPooling2D', 'GlobalAveragePooling2D'
    :param initializer = initialiser funtion. By default variables are initialised with zeros. Check Keras initialiser
     functions (e.g. 'glorot_uniform', 'glorot_normal')
    :return: model
    """
    if weights not in {None}:
        raise ValueError('No Weights are currently available for this version of the model')
    # Determine proper input shape

    if depth == 0: # VGG11
        min_size = 2**4
    elif depth == 1:
        min_size = 2**5
    elif depth == 2:
        min_size = 2**6
    elif depth == 3:
        min_size = 2**7
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=256,
                                      min_size=min_size,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    # This allows to use the model within another sequential model
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    #initializer = get_initialiser(keras_fn_name=initializer)

    block = 1
    x = conv_block(32, dropout=conv_dropout, activation=activation, block=block, layer=1, initializer = initializer)(img_input)
    x = conv_block(32, dropout=conv_dropout, activation=activation, block=block, layer=2, initializer = initializer)(x)
    x = pool_block(pooling = pooling_cnv, block=1)(x) #/2

    block = 2
    x = conv_block(64, dropout=conv_dropout, activation=activation, block=block, layer=1, initializer = initializer)(x)
    x = conv_block(64, dropout=conv_dropout, activation=activation, block=block, layer=2, initializer = initializer)(x)
    x = pool_block(pooling=pooling_cnv, block=block)(x) # /2

    block = 3
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=block, layer=1, initializer = initializer)(x)
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=block, layer=2, initializer = initializer)(x)
    if depth >= 1:
        x = conv_block(128, dropout=conv_dropout, activation=activation, block=block, layer=3, initializer = initializer)(x)
    if depth >=2:
        x = conv_block(128, dropout=conv_dropout, activation=activation, block=block, layer=4, initializer = initializer)(x)
    x = pool_block(pooling=pooling_cnv, block=block)(x)

    block = 4
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=1, initializer = initializer)(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=2, initializer = initializer)(x)
    if depth >= 1:
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=3, initializer = initializer)(x)
    if depth >=2:
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=4, initializer = initializer)(x)
    if depth >=3:
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=5, initializer = initializer)(x)
    x = pool_block(pooling=pooling_cnv, block=block)(x)

    block = 5
    if depth >= 1:
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=1, initializer = initializer)(x)
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=2, initializer = initializer)(x)
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=3, initializer = initializer)(x)
    if depth >=2:
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=4, initializer = initializer)(x)
    if depth >=3:
        x = conv_block(256, dropout=conv_dropout, activation=activation, block=block, layer=5, initializer = initializer)(x)
    if depth >= 1:
        x = pool_block(pooling=pooling_cnv, block=block)(x)

    block = 6
    if depth >= 2:
        x = conv_block(512, dropout=conv_dropout, activation=activation, block=block, layer=1, initializer = initializer)(x)
        x = conv_block(512, dropout=conv_dropout, activation=activation, block=block, layer=2, initializer = initializer)(x)
        x = conv_block(512, dropout=conv_dropout, activation=activation, block=block, layer=3, initializer = initializer)(x)
        x = conv_block(512, dropout=conv_dropout, activation=activation, block=block, layer=4, initializer = initializer)(x)
    if depth >= 3:
        x = conv_block(512, dropout=conv_dropout, activation=activation, block=block, layer=5, initializer = initializer)(x)
    if depth >= 2:
        x = pool_block(pooling=pooling_cnv, block=block)(x)

    block = 7
    if depth >= 3:
        x = conv_block(1024, dropout=conv_dropout, activation=activation, block=block, layer=1, initializer = initializer)(x)
        x = conv_block(1024, dropout=conv_dropout, activation=activation, block=block, layer=2, initializer = initializer)(x)
        x = conv_block(1024, dropout=conv_dropout, activation=activation, block=block, layer=3, initializer = initializer)(x)
        x = conv_block(1024, dropout=conv_dropout, activation=activation, block=block, layer=4, initializer = initializer)(x)
        x = conv_block(1024, dropout=conv_dropout, activation=activation, block=block, layer=5, initializer = initializer)(x)
        x = pool_block(pooling=pooling_cnv, block=block)(x)

    if include_top:
        # Flatten 2
        if pooling_fc == 'GlobalAveragePooling2D':
            x = layers.GlobalAveragePooling2D()(x) # x = layers.Flatten(name='flatten')(x)
        elif pooling_fc == 'GlobalMaxPooling2D':
            x = layers.GlobalMaxPooling2D()(x)

        # FC Layers
        x = dense_block(512, dropout=fc_dropout, activation=activation, name='fc1', initializer = initializer)(x) # 4096
        x = dense_block(512, dropout=fc_dropout, activation=activation, name='fc2', initializer = initializer)(x) # 4096

        # Classification block
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    """
    Original VGG16 for reference 
    # Block 1
    x = layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
        if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1000, activation='softmax', name='predictions')(x)
    
    """


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='VGGbnV1')

    # load weights
    #TODO
    return model