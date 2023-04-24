"""
This script is used to define various augmenter functions that can be used to apply image augmentations on batches of
images. The main goal of this module is to create a function that returns a augmenter function that can be used by
Keras Image Generators for applying augmentations on the images which can then be used to train various deep
learning models.

It is important to note that the module is specifically designed to support Keras Image Generators and thus focuses
on a more functional paradigm of returning functions and inputting functions. Also the main function that will be
used by the generator would be the get_image_augmenter_function which will return a function that will be used by
the Keras Generators to apply augmentations. This function will return augmenters based on the parameters specified.

TODO: Update this script and determine what should be deprecated and what should be kept and maintained
"""

from imgaug import augmenters as iaa


def get_imgaug_augmentation_function_name_map():
    return {
        'add': iaa.Add,
        'addelementwise': iaa.AddElementwise,
        'additivegaussiannoise': iaa.AdditiveGaussianNoise,
        'additivelaplacenoise': iaa.AdditiveLaplaceNoise,
        'additivepoissonnoise': iaa.AdditivePoissonNoise,
        'multiply': iaa.Multiply,
        'multiplyelementwise': iaa.MultiplyElementwise,
        'dropout': iaa.Dropout,
        'coarsedropout': iaa.CoarseDropout,
        'replaceelementwise': iaa.ReplaceElementwise,
        'impulsenoise': iaa.ImpulseNoise,
        'saltandpepper': iaa.SaltAndPepper,
        'coarsesaltandpepper': iaa.CoarseSaltAndPepper,
        'salt': iaa.Salt,
        'coarsesalt': iaa.CoarseSalt,
        'pepper': iaa.Pepper,
        'coarsepepper': iaa.CoarsePepper,
        'invert': iaa.Invert,
        'contrastnormalization': iaa.ContrastNormalization,
        'jpegcompression': iaa.JpegCompression,
        'alpha': iaa.Alpha,
        'alphaelementwise': iaa.AlphaElementwise,
        'simplexnoisealpha': iaa.SimplexNoiseAlpha,
        'frequencynoisealpha': iaa.FrequencyNoiseAlpha,
        'gaussianblur': iaa.GaussianBlur,
        'averageblur': iaa.AverageBlur,
        'medianblur': iaa.MedianBlur,
        'bilateralblur': iaa.BilateralBlur,
        'motionblur': iaa.MotionBlur,
        'withcolorspace': iaa.WithColorspace,
        'addtohueandsaturation': iaa.AddToHueAndSaturation,
        'changecolorspace': iaa.ChangeColorspace,
        'grayscale': iaa.Grayscale,
        'gammacontrast': iaa.GammaContrast,
        'sigmoidcontrast': iaa.SigmoidContrast,
        'logcontrast': iaa.LogContrast,
        'linearcontrast': iaa.LinearContrast,
        'allchannelshistogramequalization': iaa.AllChannelsHistogramEqualization,
        'histogramequalization': iaa.HistogramEqualization,
        'allchannelsclahe': iaa.AllChannelsCLAHE,
        'clahe': iaa.CLAHE,
        'convolve': iaa.Convolve,
        'sharpen': iaa.Sharpen,
        'emboss': iaa.Emboss,
        'edgedetect': iaa.EdgeDetect,
        'directededgedetect': iaa.DirectedEdgeDetect,
        'fliplr': iaa.Fliplr,
        'flipud': iaa.Flipud,
        'affine': iaa.Affine,
        'affinecv2': iaa.AffineCv2,
        'piecewiseaffine': iaa.PiecewiseAffine,
        'perspectivetransform': iaa.PerspectiveTransform,
        'elastictransformation': iaa.ElasticTransformation,
        'rot90': iaa.Rot90,
        'sequential': iaa.Sequential,
        'someof': iaa.SomeOf,
        'oneof': iaa.OneOf,
        'sometimes': iaa.Sometimes,
        'withcolorspace': iaa.WithColorspace,
        'withchannels': iaa.WithChannels,
        'noop': iaa.Noop,
        'lambda': iaa.Lambda,
        'assertlambda': iaa.AssertLambda,
        'assertshape': iaa.AssertShape,
        'channelshuffle': iaa.ChannelShuffle,
        'superpixels': iaa.Superpixels,
        'resize': iaa.Resize,
        'cropandpad': iaa.CropAndPad,
        'pad': iaa.Pad,
        'crop': iaa.Crop,
        'padtofixedsize': iaa.PadToFixedSize,
        'croptofixedsize': iaa.CropToFixedSize,
        'keepsizebyresize': iaa.KeepSizeByResize,
        'fastsnowylandscape': iaa.FastSnowyLandscape,
        'clouds': iaa.Clouds,
        'fog': iaa.Fog,
        'cloudlayer': iaa.CloudLayer,
        'snowflakes': iaa.Snowflakes,
        'snowflakeslayer': iaa.SnowflakesLayer
    }


def get_imgaug_augmentation_function(augmentation_function_name: str):
    """
    This function returns the actual imgaug augmentation function based on the name of the imgaug function name
    :param augmentation_function_name: The name of the imgaug augmentation function
    :return: The reference to the imgaug function which is responsible to apply the corresponding augmentation
    """
    try:
        return get_imgaug_augmentation_function_name_map()[augmentation_function_name.lower()]
    except KeyError:
        raise ValueError(f'Invalid augmentation function name: {augmentation_function_name}')


def get_all_augmentation_data_frame(file_path=f'./augmentation_readme.csv'):
    """
    :return: This function returns a dataframe that contains the information about all the imgaug augmentation
    functions. The information present includes the function signature of the augmentation function along with the
    corresponding docstring for the function
    """
    import pandas as pd
    return pd.read_csv(file_path)


def get_augmentation_function(augmentation_details: tuple):
    """
    Returns the actual function representing the augmentations with the appropriate call based on the
    parameters that need to be provided.

    :param augmentation_details: The tuple representing the parameters for the augmentation function and the type of
    augmentation function that needs to be returned.
    The tuple can be of 2 different size as follows:
        * Size 2 Tuple: First element of the tuple is the string name of the type of augmentation that should be
        applied. Then the second element should either be a tuple or a dictionary representing the arguments of the
        augmentation function. To use the default arguments of the augmentation functions, can just leave an empty
        tuple or empty dictionary as the second element
        * Size 3 Tuple: The first two things are similar to the Size 2 Tuple, however the last element is another
        tuple for which the first element is the list which is defined as the similar structure to the whole
        augmentation list on which the function will be called again to generate a nested augmentation list. While
        the other is the number of augmentation of it to be applied.

    :return: The augmentation function called with the appropriate parameters
    """
    augmentation_function_name, augmentation_function_primary_arguments = augmentation_details[0], \
                                                                          augmentation_details[1]
    augmentation_function = get_imgaug_augmentation_function(augmentation_function_name)
    if type(augmentation_function_primary_arguments) == tuple:
        if len(augmentation_details) == 3:
            return augmentation_function(*augmentation_function_primary_arguments,
                                         get_augmentation_sequence(*augmentation_details[-1]))
        else:
            return augmentation_function(*augmentation_function_primary_arguments)
    if type(augmentation_function_primary_arguments) == dict:
        if len(augmentation_details) == 3:
            return augmentation_function(**augmentation_function_primary_arguments,
                                         **{'children': get_augmentation_sequence(*augmentation_details[-1])})
        else:
            return augmentation_function(**augmentation_function_primary_arguments)


def get_augmentation_sequence(augmentations: list, augmentations_per_image_range: tuple):
    """
    This function is responsible to return a sequence defining the augmentations required

    :param augmentations: A list representing the types of augmentations that can be applied to the images. It
    is very important to keep in mind the structure of the augmentations list. Each element of the list is a tuple.
    The tuple can be of 2 different size as follows:
        * Size 2 Tuple: First element of the tuple is the string name of the type of augmentation that should be
        applied. Then the second element should either be a tuple or a dictionary representing the arguments of the
        augmentation function. To use the default arguments of the augmentation functions, can just leave an empty
        tuple or empty dictionary as the second element
        * Size 3 Tuple: The first two things are similar to the Size 2 Tuple, however the last element is another
        tuple for which the first element is the list which is defined as the similar structure to the whole
        augmentation list on which the function will be called again to generate a nested augmentation list. While
        the other is the number of augmentation of it to be applied.
    :param augmentations_per_image_range: The range of number of augmentations that should be applied to the images.
    This argument can either be an empty tuple representing a sequential model, or can be a tuple of length 1
    representing the number of augmentations to be applied randomly, or a tuple or size 2 representing the range of
    number of augmentations that can be applied

    :return: A sequence for the augmentations
    """
    augmentation_function_list = \
        [get_augmentation_function(augmentation_details) for augmentation_details in augmentations]
    if len(augmentations_per_image_range) == 1:
        return iaa.SomeOf((augmentations_per_image_range[0], augmentations_per_image_range[0]),
                          augmentation_function_list)
    elif len(augmentations_per_image_range) == 2:
        return iaa.SomeOf(augmentations_per_image_range,
                          augmentation_function_list)
    elif len(augmentations_per_image_range) == 0:
        return iaa.Sequential(augmentation_function_list, random_order=True)
    else:
        return []


def get_image_augmenter(augmentations: list, augmentations_per_image_range: tuple):
    """
    This function is responsible to return a function that expects a 4-D numpy array or a List of 3-D numpy array on
    which the image augmentations will be applied.

    :param augmentations: A list representing the types of augmentations that can be applied to the images. It
    is very important to keep in mind the structure of the augmentations list. Each element of the list is a tuple.
    The tuple can be of 2 different size as follows:
        * Size 2 Tuple: First element of the tuple is the string name of the type of augmentation that should be
        applied. Then the second element should either be a tuple or a dictionary representing the arguments of the
        augmentation function. To use the default arguments of the augmentation functions, can just leave an empty
        tuple or empty dictionary as the second element
        * Size 3 Tuple: The first two things are similar to the Size 2 Tuple, however the last element is another
        tuple for which the first element is the list which is defined as the similar structure to the whole
        augmentation list on which the function will be called again to generate a nested augmentation list. While
        the other is the number of augmentation of it to be applied.
    :param augmentations_per_image_range: The range of number of augmentations that should be applied to the

    :return: A function that takes in 1 argument which is either a 4-D numpy array of images or a list of 3-D array
    of images and apply the augmentations
    """
    sequence = get_augmentation_sequence(augmentations, augmentations_per_image_range)
    return sequence.augment_images
