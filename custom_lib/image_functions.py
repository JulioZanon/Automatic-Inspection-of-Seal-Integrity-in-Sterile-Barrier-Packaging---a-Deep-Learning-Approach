import numpy as np
import cv2
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Model
#from keras_cv.layers import Rescaling, RandomRotation, RandomCrop, RandomBrightness, Resizing

import os
from bscai.annotation.load import regions_to_array, get_json_data
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gc


def generate_list_of_files(path_to_derectory = '', list_of_file_names = [], keywords = [], file_extension = ''):
    #get list from provided names
    if len(list_of_file_names) > 0:
        files = list_of_file_names
        for i in range(len(list_of_file_names)):
            files[i] = os.path.join(path_to_derectory, list_of_file_names[i])
    #get list from folders and subfolders
    else:
        files = [os.path.join(path, name) for path, subdirs, files in os.walk(path_to_derectory) for name in files]
    #filter keywords
    filtered_files = []
    for i in range(len(files)):
        body, extension = os.path.splitext(os.path.basename(files[i]))
        if extension[1:] == file_extension or len(file_extension) == 0:
            keyword_match = False
            if len(keywords) > 0:
                for keyword in keywords:
                    if keyword in body: keyword_match = True
            else: keyword_match = True
            if keyword_match: filtered_files.append(files[i])
    return filtered_files


class img_preProcess:
	def __init__(self, img: None or np.array, file_name: None or str()):

		# check for inputs integrity
		if img is not None and file_name is not None:
			raise RuntimeError('Only one argument is accepted, either img or file_name')
		elif img is not None:
			if type(img) != np.ndarray:
				raise RuntimeError('img input of type: ' +  str(type(img)) + ' must by a nupmy array')
		elif file_name is not None:
			if os.path.splitext(file_name)[1] not in ['.jpeg', '.bmp', '.png']:
				if type(img) != np.ndarray:
					raise RuntimeError('file name: [' + file_name + '] is not an accepted image format. Only following formats are accepted: jpeg, bmp, png ')
		# initialise object
		self.img = img
		self.file_name = file_name
		# read, decode and normalise image
		if self.img is None:
			self.read_img()
		else:
			if np.max(self.img) > 1: self.normalise()

	def read_img(self):
		self.img = tf.io.read_file(self.file_name)
		format = os.path.splitext(self.file_name)[1]
		# convert the compressed string to a 3D uint8 tensor
		if format == '.jpeg':
			self.img = tf.image.decode_jpeg(self.img, channels=3)
		elif format == '.bmp':
			self.img = tf.image.decode_bmp(self.img, channels=3)
		elif format == '.png':
			self.img = tf.image.decode_png(self.img, channels=3)
		# normalise img [0 to 1]
		self.normalise()

	def normalise(self):
		self.img = tf.image.convert_image_dtype(self.img, tf.float32)

	def crop_resize(self, crop_origin = [0,0], crop_size= [150, 150], img_resize= [150,0], convert_power_of_2 = False):
		img_size = [self.img.shape[0], self.img.shape[1]]
		size = [0, 0]
		if img_resize[0] > 0:
			if convert_power_of_2:
				size[0] = int(2 ** (np.rint(np.log2(img_resize[0]))))
			else:
				size[0] = img_resize[0]
			if img_resize[1] > 0:
				if convert_power_of_2:
					size[1] = int(2 ** (np.rint(np.log2(img_resize[1]))))
				else:
					size[1] = img_resize[1]
			elif crop_size[0] == 0 or crop_size[1] == 0:
				size[1] = int(2 ** (np.rint(np.log2(int((size[0] / img_size[0]) * img_size[1])))))
			else:
				size[1] = int(2 ** (np.rint(np.log2(int((size[0] / crop_size[0]) * crop_size[1])))))
		elif crop_size[0] > 0 and crop_size[1] > 0:
			[int(2 ** (np.rint(np.log2(crop_size[0])))), int(2 ** (np.rint(np.log2(crop_size[1]))))]
		else:
			size = [int(2 ** (np.rint(np.log2(img_size[0])))), int(2 ** (np.rint(np.log2(img_size[1]))))]
		if crop_size[0] > 0 and crop_size[0] > 0:
			self.img = tf.expand_dims( self.img, axis=0)
			x1 = crop_origin[0] / img_size[0]
			y1 = crop_origin[1] / img_size[1]
			x2 = (crop_origin[0] + crop_size[0]) / img_size[0]
			y2 = (crop_origin[1] + crop_size[1]) / img_size[1]
			self.img = tf.image.crop_and_resize(self.img, boxes=
			[[x1, y1, x2, y2]], crop_size=[size[0], size[1]], box_indices=[0])
			self.img = tf.squeeze(self.img, 0)
		elif size[0] > 0 and size[1] > 0 and (size[0] != img_size[0] or size[1] != img_size[1]):
			self.img = tf.image.resize( self.img, [size[0], size[1]])
	def rotate(self):
		(h, w) = self.img.shape[:2]
		center = (w / 2, h / 2)
		M = cv2.getRotationMatrix2D(center, 180, 1.0)
		self.img = cv2.warpAffine(self.img, M, (w, h))
	def resize(self,  img_resize= [150,0]):
		self.img = tf.image.resize(self.img, img_resize)

	def get_img(self):
		return self.img


def create_heatmap_array(predictions = [], threshold = 0.2, colormap = cv2.COLORMAP_JET):
    #normalise prediciton
    predictions /= predictions.max() / 255
    #apply threshold to clear up the background
    predictions[predictions <  threshold]  = 0
    predictions = np.uint8(predictions)
    heatmaps = np.empty([predictions.shape[0], predictions.shape[-1], predictions.shape[1], predictions.shape[2], 3], dtype=np.uint8)
    for img_idx in range(predictions.shape[0]):
        for class_idx in range(predictions.shape[-1]):
            heatmaps[img_idx, class_idx, ...] = cv2.applyColorMap( predictions[img_idx, ..., class_idx], colormap)
    return heatmaps




class ClassifierActvMap:
	"""
	Grad-CAM is used to analyse the gradient flowing into the last conv layer and visualise the regions of the imate that most
	affect the classification prediction.
	Code adapted form:
	https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
	"""
	def __init__(self, model, classIdx, layerName=None, number_of_layers_before_last = 0):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		self.number_of_layers_before_last = number_of_layers_before_last

		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		i = 0
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				if i >= self.number_of_layers_before_last: return layer.name
				else: i += 1


		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

	def compute_heatmap(self, image, eps=1e-8, score = 0.0, colormap = cv2.COLORMAP_HOT):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output,
				self.model.output])

		# record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]

		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)

		# compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads

		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		# compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		# grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))

		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer

		max_scale = int(round(score * 255, 0))
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * max_scale).astype("uint8")
		heatmap = cv2.applyColorMap(heatmap, colormap)
		# return the resulting heatmap to the calling function
		return heatmap

##################################################################################################

def crop_nd_resize(img, img_size = [0, 0], crop_origin=[0, 0], crop_size= [150, 150], img_resize= [150,0], convert_power_of_2 = False):
    """
    Takes a multidimensional tensor (e.g. image with multiple channels or masks) and crop-resize the first 2 dimensions.
    If one of the dimensions in the resize argument is left as 0 while the other is an integer, it calculates the dimension left to 0
    to keep same ratio as the original size.
    :param img: multidimensional tensor, with at least 2 dimensions.
    :param img_size: resize in pixels (heigth, width). if heigth == 0, then size = crop. if heigth <> 0 and width == 0, then it keeps aspect
    :param crop_origin: x1, y1 coordinates (x=vertical, y=horizontal)
    :param crop_size: size in pixels. (heigth, width). if 0 in both dimensions, then no crop is applied
    :param img_resize: new image size.
    :param convert_power_of_2: if True, the new size will be calculated to be a multiple of a power of 2.
    :return: cropped Tesnor with new size.
    """
    #check dimensions of input and raise error is less than 2.
    is_not_2_dim = tf.less(tf.size(tf.shape(img)), tf.constant(2))
    if is_not_2_dim.numpy(): raise RuntimeError('crop_nd_resize requires images or maks of at least 2 dimensions.')

    # Re-size image as per img_resize input
    # If Higth = 0 then image to crop size, or if no crop then keep original size
    # if if Higth > 0 and Width = 0, then resize width as per Hight by keeping same ratio


    # If Higth = 0 then image to crop size, or if no crop then keep original size
    # if if Higth > 0 and Width = 0, then resize width as per Hight by keeping same ratio
    size= [0,0]
    if img_resize[0] > 0:
        if convert_power_of_2: size[0] = int(2**(np.rint(np.log2(img_resize[0]))))
        else: size[0] = img_resize[0]
        if img_resize[1] > 0:
            if convert_power_of_2: size[1] = int(2**(np.rint(np.log2(img_resize[1]))))
            else: size[1] = img_resize[1]
        elif crop_size[0] == 0 or crop_size[1] == 0: size[1] = int(2**(np.rint(np.log2(int(( size[0]/img_size[0]) * img_size[1])))))
        else:  size[1] = int(2**(np.rint(np.log2(int((size[0]/crop_size[0]) * crop_size[1])))))
    elif crop_size[0] > 0 and crop_size[1] > 0: [int(2 ** (np.rint(np.log2(crop_size[0])))), int(2 ** (np.rint(np.log2(crop_size[1]))))]
    else: size = [int(2**(np.rint(np.log2(img_size[0])))), int(2**(np.rint(np.log2(img_size[1]))))]
    if crop_size[0] > 0 and crop_size[0] > 0:
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

    return img




def read_mask_from_folder(file_path, img_size = [0, 0], crop_origin=[0, 0], crop_size= [150, 150], img_resize= [150,0], class_name_list = [] ):
	"""
	Takes a single path to a json file and decode the notation to create a tensor mask. The mask is then cropped and resized.
	It uses the functions regions_to_array and get_json_data from the bscai libraries.
	If the path is empty, then it creates an empty mask with the img_resize parameters.
	:param file_path: file path to json file
	:param img_size: image size
	:param img_resize: new image size
	:param crop_origin: origin of crop
	:param crop_size: crop size
	:param class_name_list: list with classes.
	:return: cropped and resized tensor image
	"""

	if len(file_path) > 0:
		# Create mask from json file.
		json_data = get_json_data(file_path)
		np_mask = regions_to_array(json_data['annotations_list'][0], class_name_list, tuple(json_data['image_shape']))
		# resize the image to the desired size.
		tf_mask = crop_nd_resize(tf.convert_to_tensor(np_mask), img_size=img_size, crop_origin=crop_origin, crop_size=crop_size, img_resize=img_resize)
	else:
		# creat null mask for empty paths.
		null_mask = np.zeros(img_resize[0] * img_resize[1] * len(class_name_list)).reshape(img_resize[0], img_resize[1], len(class_name_list))
		tf_mask = tf.convert_to_tensor(null_mask)
	return tf.cast(tf_mask, dtype='uint8')


def print_np_samples(class_name_list, np_image_names_array, np_image_array, np_classes_array, np_mask_array = None, num_of_samples_per_class = 1, mask_for_class_0_exists = False):

	if np_mask_array is not None and len(class_name_list) > 1 and not mask_for_class_0_exists:
		class_name_list = class_name_list[1:]
		np_classes_array = np_classes_array[:,1:]

	num_of_classes = len(class_name_list)
	dataset_size = np_image_array.shape[0]
	print("-Printing dataset: %s images" % (dataset_size))

	# get idx
	for i in range(num_of_classes):
		#randomly select num_of_samples_per_class for that class.
		idx = np.random.choice(np.where(np_classes_array[:,i])[0],num_of_samples_per_class)
		for j in idx:
			if np_mask_array is not None: plt.subplot(212)
			plt.title(np_image_names_array[j])
			plt.imshow(np_image_array[j])
			plt.axis('off')
			if np_mask_array is not None:
				plt.subplot(2, 1,1)
				plt.title('Class: ' + class_name_list[i])
				plt.imshow(np_mask_array[j,:, :, i], cmap='gray')
				plt.axis('off')
			plt.show(block=False)


@tf.function
def read_img_from_folder(file_path, PreProcessFunction):
	"""
	Reads image from folder. Assumes image has 3 channels.
	If PreProcessFunction function is not None, it applies the function.
	:param file_path: path to image file
	:param PreProcessFunction: preprocessing function. Typically cropping and resize.
	:return: returns image tensor
	"""
	img = tf.io.read_file(file_path)
	_, img_format = os.path.splitext(file_path)
	img_format = str(img_format).lower()
	if img_format == '.jpeg':
		img = tf.image.decode_jpeg(img, channels=3)
	elif img_format == '.bmp':
		img = tf.image.decode_bmp(img, channels=3)
	elif img_format == '.png':
		img = tf.image.decode_png(img, channels=3)
	else:
		raise RuntimeError('Image format not recognised, please use bmp, png or jpeg')

	# Use `convert_image_dtype` to convert to floats in the [0,1] range and preprocess img.
	img = tf.image.convert_image_dtype(img, tf.float32)
	# preprocess img with function
	if PreProcessFunction is not None:
		img = PreProcessFunction(img)
	return img


@tf.function
def read_img_from_folder_w_processing(file_path, ImgPreProcessingFunc=None, ImgPostProcessingFunc=None, seed=(0,0)):
	"""
	Reads image from folder. Assumes image has 3 channels.
	If PreProcessFunction function is not None, it applies the function.
	:param file_path: path to image file
	:param PreProcessFunction: preprocessing function. Typically cropping and resize.
	:return: returns image tensor
	"""
	img = tf.io.read_file(file_path)
	_, img_format = os.path.splitext(file_path)
	img_format = str(img_format).lower()
	if img_format == '.jpeg':
		img = tf.image.decode_jpeg(img, channels=3)
	elif img_format == '.bmp':
		img = tf.image.decode_bmp(img, channels=3)
	elif img_format == '.png':
		img = tf.image.decode_png(img, channels=3)
	else:
		raise RuntimeError('Image format not recognised, please use bmp, png or jpeg')

	# preprocess img with function
	if ImgPreProcessingFunc is not None:
		img = ImgPreProcessingFunc(img, seed=seed)
	if ImgPostProcessingFunc is not None:
		img = ImgPostProcessingFunc(img, seed=seed, mask=False)
	return img


@tf.function
def read_mask_from_folder_w_processing(file_path = '', ImgPreProcessingFunc=None, ImgPostProcessingFunc=None, seed=(0,0), class_name_list = [], mask_size = [0, 0]):
	"""
	Read mask from either an image file or decode the mask from a json file with schema "" and encoding "".
	If file path is empty, it create an empty mask with size = size.
	If preprocessing function is not None, it applies the function.
	:param file_path: path to either the mask sile of the json file with the encoded regions/
	:param size: if file_path is empty, then it creats an empty mask of size = size and #of Channels = # of classes.
	:param class_name_list: used to extract number of channels.
	:param PreProcessFunction: preprocessing function. Typically cropping and resizing.
	:param mask_size: original image size to create empty masks when no path is passed.
	:return: mask tensor
	"""
	#TODO , channels=len(class_name_list)
	# Tensorflow updates force a defined number of channels.
	# Filter no labled class from the class list as they will not have masks
	class_name_list = [cls for cls in class_name_list if cls != 'NoLabel']

	if len(file_path) > 0:
		_, mask_format = os.path.splitext(file_path)
		mask_format = str(mask_format).lower()
		if mask_format == '.json':
			get_json_data_out = get_json_data(file_path)
			mask = regions_to_array(get_json_data_out['annotations_list'][0], class_name_list,
									tuple(get_json_data_out['image_shape']))
			mask = tf.convert_to_tensor(mask, dtype=tf.float32)
		else:
			mask = tf.io.read_file(file_path)
			if mask_format == '.jpeg':
				mask = tf.image.decode_jpeg(mask, channels=len(class_name_list))
			elif mask_format == '.bmp':
				mask = tf.image.decode_bmp(mask, channels=len(class_name_list))
			elif mask_format == '.png':
				mask = tf.image.decode_png(mask, channels=len(class_name_list))
			else:
				raise RuntimeError('Mask format not recognised, please use bmp, png or jpeg')
	else:
		# Empty mask as no file path was provided
		mask = tf.expand_dims(tf.zeros(mask_size, dtype=tf.float32), axis=2)
		# add channels for each class
		if len(class_name_list) >1:
			for i in range(len(class_name_list) - 1):
				mask = tf.concat([mask, tf.expand_dims(tf.zeros(mask_size, dtype=tf.float32), axis=2)], axis=2)

	# preprocess img with function
	if ImgPreProcessingFunc is not None:
		mask = ImgPreProcessingFunc(mask, seed=seed)
	if ImgPostProcessingFunc is not None:
		mask = ImgPostProcessingFunc(mask, seed=seed, mask=True)

	# Threshold and round
	max_value = tf.reduce_max(mask)
	mask = tf.cond(tf.greater(max_value, 0), lambda: tf.math.round(tf.divide(mask, max_value)), lambda: mask)
	return mask

#
# @tf.function
# def read_mask_from_folder(file_path = '', class_name_list = [], PreProcessFunction = None, mask_size = [0, 0]):
# 	"""
# 	Read mask from either an image file or decode the mask from a json file with schema "" and encoding "".
# 	If file path is empty, it create an empty mask with size = size.
# 	If preprocessing function is not None, it applies the function.
# 	:param file_path: path to either the mask sile of the json file with the encoded regions/
# 	:param size: if file_path is empty, then it creats an empty mask of size = size and #of Channels = # of classes.
# 	:param class_name_list: used to extract number of channels.
# 	:param PreProcessFunction: preprocessing function. Typically cropping and resizing.
# 	:param mask_size: original image size to create empty masks when no path is passed.
# 	:return: mask tensor
# 	"""
#
# 	# Filter no labled class from the class list as they will not have masks
# 	class_name_list = [cls for cls in class_name_list if cls != 'NoLabel']
#
# 	if len(file_path) > 0:
# 		_, mask_format = os.path.splitext(file_path)
# 		mask_format = str(mask_format).lower()
# 		if mask_format == '.json':
# 			get_json_data_out = get_json_data(file_path)
# 			mask = regions_to_array(get_json_data_out['annotations_list'][0], class_name_list,
# 									tuple(get_json_data_out['image_shape']))
# 			mask = tf.convert_to_tensor(mask, dtype=tf.float32)
# 		else:
# 			mask = tf.io.read_file(file_path)
# 			if mask_format == '.jpeg':
# 				mask = tf.image.decode_jpeg(mask, channels=len(class_name_list))
# 			elif mask_format == '.bmp':
# 				mask = tf.image.decode_bmp(mask, channels=len(class_name_list))
# 			elif mask_format == '.png':
# 				mask = tf.image.decode_png(mask, channels=len(class_name_list))
# 			else:
# 				raise RuntimeError('Mask format not recognised, please use bmp, png or jpeg')
# 	else:
# 		# Empty mask as no file path was provided
# 		mask = tf.expand_dims(tf.zeros(mask_size, dtype=tf.float32), axis=2)
# 		# add channels for each class
# 		if len(class_name_list) >1:
# 			for i in range(len(class_name_list) - 1):
# 				mask = tf.concat([mask, tf.expand_dims(tf.zeros(mask_size, dtype=tf.float32), axis=2)], axis=2)
#
# 	# preprocess img with function
# 	if PreProcessFunction is not None:
# 		mask = PreProcessFunction(mask)
#
# 	# Threshold and round
# 	max_value = tf.reduce_max(mask)
# 	mask = tf.cond(tf.greater(max_value, 0), lambda: tf.math.round(tf.divide(mask, max_value)), lambda: mask)
# 	return mask
#
#
#
# ##### IMG PREPROCESSIGN FUNCTION (before Augmentation is applied (e.g. Cropping, resize, etc)
# def get_PreProcessFunction(PreProcessingImgParams):
# 	img_size = PreProcessingImgParams[0]
# 	crop_origin = PreProcessingImgParams[1]
# 	crop_size = PreProcessingImgParams[2]
# 	img_resize = PreProcessingImgParams[3]
# 	@tf.function
# 	def ImgGenfun(img):
# 		"""
# 		Preprocessing Function Rev:1.00
# 		Current preprocessing function crops and resize image with fixed inputs.
# 		Note that function uses tensors.
# 		:param img: interator img
# 		:return: preprocessed img
# 		"""
# 		if crop_size[0] == 0 or crop_size[1] == 0 or img_resize[0] == 0 or img_resize[1] == 0:
# 			return img
# 		else:
# 			y1 = crop_origin[0] / img_size[0]
# 			x1 = crop_origin[1] / img_size[1]
# 			y2 = (crop_origin[0] + crop_size[0]) / img_size[0]
# 			x2 = (crop_origin[1] + crop_size[1]) / img_size[1]
# 			boxes = tf.convert_to_tensor(tf.constant(np.expand_dims(np.array([y1, x1, y2, x2], dtype = 'f'), axis=0)))
# 			size = tf.convert_to_tensor(tf.constant(np.array([img_resize[0], img_resize[1]])),
# 										 dtype=tf.int32)
# 			box_idices = tf.convert_to_tensor(tf.constant(np.array([0])),
# 										dtype=tf.int32)
# 			img = tf.expand_dims(img, axis=0)
# 			img = tf.image.crop_and_resize(img, boxes=boxes, crop_size=size, box_indices=box_idices)
# 			img = tf.squeeze(img, 0)
# 			return img
# 	return ImgGenfun
#
#
# ##### IMG POSTPROCESSIGN FUNCTION (Augmentation and network size
# def get_PostProcessFunction(inp_params = None):
# 	"""
# 	Postprocessing Function Rev:1.00
# 	This function should return the function generators for augmentation.
# 	Current function uses the tensorflow ImageDataGenearator..
# 	In future versions, consider introducing custom augmentation functions.
# 	:param inp_params: input parameters of https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow
# 	:return: augmentator Gen.
# 	"""
#
# 	if inp_params is not None:
# 		ImgGenfun = ImageDataGenerator(**inp_params)
# 	else:
# 		def ImgGenfun(img):
# 			return(img)
# 	return ImgGenfun


def GetImageProcessingFuncs(ImgCropnResize = [[0,0], [0,0], [0,0], [0,0]], Augmentation = None, scale = 1./255):
	"""
	Function to process images for cropping, resizing and augmentation.
	:param ImgCropnResize: [[img_size], [crop_origin], [crop_size], [img_resize]]
	:param Augmentation: {
		'horizontal_shift_range': pixels in the vertical axis,
		'vertical_shift_range': pixels in the Horizontal axis,
		'zoom': [min, max] percentage (0.1 = 10%) where (-) zoom in and (+) zomm out
		'vertical_flip': flip along the vertical axis,
		'horizontal_flip': flip along the horizontal axis,
		'rotation': range where 1 = 90 deg. Function will randomise clockW and anticlockW
		'brightness_range': 0 to 1. Note: when >TF2.9 fix RandomBrightness but, then:[min, max] -1 to 1.
		}
	:param scale: Default 1/.255 (0 to 1)
	:return: Dunctrion with two inputs (img, seed)
	"""
	# ImgCropnResize
	img_size = ImgCropnResize[0]
	crop_origin = ImgCropnResize[1]
	crop_size = ImgCropnResize[2]
	img_resize = ImgCropnResize[3]
	if Augmentation is None: Augmentation = {}

	A_RotationLayer_b = False
	if 'rotation' in Augmentation:
		if Augmentation['rotation'] > 0 and Augmentation['rotation'] <= 1:
			A_RotationLayer_b = True
	# Crop
	A_Crop_layer_b = False
	crop_size_rnd = crop_size.copy()

	if crop_size[0] > 0 and crop_size[1] > 0 and img_resize[0] > 0 and img_resize[1] > 0 \
			and (img_resize[0] != img_size[0] or img_resize[1] != img_size[1]):

		# Expand crop if shift is needed
		if 'horizontal_shift_range' in Augmentation:
			if crop_origin[0] > Augmentation['horizontal_shift_range']:
				crop_origin[0] -= Augmentation['horizontal_shift_range']
			else:
				crop_origin[0] = 0

			if crop_origin[0] + crop_size[0] + 2 * Augmentation['horizontal_shift_range'] < img_size[0]:
				crop_size[0] += 2 * Augmentation['horizontal_shift_range']
			else:
				crop_size[0] = img_size[0]

		if 'vertical_shift_range' in Augmentation:
			if crop_origin[1] > Augmentation['vertical_shift_range']:
				crop_origin[1] -= Augmentation['vertical_shift_range']
			else:
				crop_origin[1] = 0

			if crop_origin[1] + crop_size[1] + 2 * Augmentation['vertical_shift_range'] < img_size[1]:
				crop_size[1] += 2 * Augmentation['vertical_shift_range']
			else:
				crop_size[1] = img_size[1]

		top_crop = crop_origin[0]
		bottom_crop = img_size[0] - (crop_origin[0] + crop_size[0])
		left_crop = crop_origin[1]
		right_crop = img_size[1] - (crop_origin[1] + crop_size[1])
		A_Crop_layer_b = True

	# Vertical and horzontal shift
	A_rnd_crop_layer_b = False
	apply_crop = False
	if 'horizontal_shift_range' in Augmentation:
		if Augmentation['horizontal_shift_range'] != 0: apply_crop = True
	if 'vertical_shift_range' in Augmentation:
		if Augmentation['vertical_shift_range'] != 0: apply_crop = True
	if apply_crop:
		A_rnd_crop_layer_b = True

	# Zoom rage [min %, max%] where 0 is no zoom, (-) is zoom in, (+) zoom out
	A_ZoomLayer_b = False
	if 'zoom' in Augmentation:
		if Augmentation['zoom'][0] != 0 or Augmentation['zoom'][1] != 0:
			A_ZoomLayer_b = True

	# Random Brightnes .
	A_brightness_layer_b = False
	if 'brightness_range' in Augmentation:
		if Augmentation['brightness_range'] != 0:
			A_brightness_layer_b = True

	# Random Flip
	A_flip_layer_b = False
	flip = None
	if 'vertical_flip' in Augmentation:
		if Augmentation['vertical_flip']:
			flip = "vertical"
	if 'horizontal_flip' in Augmentation:
		if Augmentation['horizontal_flip']:
			if flip is None:
				flip = "horizontal"
			else:
				flip = "horizontal_and_vertical"
	if flip is not None:
		A_flip_layer_b = True

	# Resize
	A_resize_layer_b = False
	if (crop_size != [0, 0] and img_resize != crop_size) or (crop_size == [0, 0]  and img_resize != img_size):
		A_resize_layer_b = True

	@tf.function
	def ImgPreProcessingFunc(img, seed= (0,0)):
		# Scale 0 to 255 : 0 to 1.
		img = tf.expand_dims(img, axis=0)
		Scale_Layer = tf.keras.layers.Rescaling(scale=scale, offset=0.0)
		img = Scale_Layer(img)
		if A_RotationLayer_b:
			A_RotationLayer = tf.keras.layers.RandomRotation( factor=Augmentation['rotation'], fill_mode='constant', interpolation='bilinear', seed=seed, fill_value=0.0,)
			img = A_RotationLayer(img)
		if A_ZoomLayer_b:
			A_ZoomLayer = tf.keras.layers.RandomZoom(height_factor=Augmentation['zoom'], width_factor=None,
													 fill_mode='constant', interpolation='bilinear', seed=seed,
													 fill_value=0.0)
			img = A_ZoomLayer(img)
		if A_Crop_layer_b:
			A_Crop_layer = tf.keras.layers.Cropping2D(cropping=((top_crop, bottom_crop), (left_crop, right_crop)),
													  data_format=None)
			img = A_Crop_layer(img)
		return tf.squeeze(img, 0)

	@tf.function
	def ImgPostProcessingFunc(img, seed= (0,0), mask=False):

		# Scale 0 to 255 : 0 to 1.
		img = tf.expand_dims(img, axis=0)
		if A_brightness_layer_b and not mask:
			# This version of TF does not have tf.keras.layers.RandomBrightness
			# A_brightness_layer = tf.keras.layers.RandomBrightness(
			# 	factor=(Augmentation['brightness_range'][0], Augmentation['brightness_range'][1]),
			# 	value_range=[0.0, 1.0], seed=seed)
			img = tf.image.stateless_random_brightness(image=img, max_delta=Augmentation['brightness_range'], seed=seed)
		if A_rnd_crop_layer_b:
			A_rnd_crop_layer = tf.keras.layers.RandomCrop(crop_size_rnd[0], crop_size_rnd[1], seed=seed)
			img = A_rnd_crop_layer(img)
		if A_resize_layer_b:
			A_resize_layer = tf.keras.layers.Resizing(img_resize[0], img_resize[1], interpolation='bilinear',
													  crop_to_aspect_ratio=False)
			img = A_resize_layer(img)
		if A_flip_layer_b:
			A_flip_layer = tf.keras.layers.RandomFlip(mode=flip, seed=seed)
			img = A_flip_layer(img)
		return tf.squeeze(img, 0)
	return ImgPreProcessingFunc, ImgPostProcessingFunc
