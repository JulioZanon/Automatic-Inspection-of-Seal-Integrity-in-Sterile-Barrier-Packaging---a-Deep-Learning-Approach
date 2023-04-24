import tensorflow as tf
import os
import glob

#region Read Dataset from TFRECORD

def _parse_tensor(example_proto):
    return tf.io.parse_tensor(example_proto['img'], tf.float32), tf.io.parse_tensor(example_proto['label'], tf.float32)

def _parse_raw(example_proto):
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
            }
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def _fixup_shape(images, labels):
        images.set_shape([64, 256, 3])
        labels.set_shape([64,256, 1])
        return images, labels


def dataset_from_tfrecord(tfrecord_folder_path):
    tfrecord_paths=get_tfrecord_paths(tfrecord_folder_path)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_paths)
    parsed_dataset = raw_dataset.map(_parse_raw)
    final_dataset = parsed_dataset.map(_parse_tensor)

    final_dataset=final_dataset.map(_fixup_shape)




    return final_dataset

def get_tfrecord_paths(tfrecord_folder_path):
    tfrecord_file_names = glob.glob(tfrecord_folder_path+'/dataset.tfrecords-*')
    tfrecord_paths=[]
    for tfrecord_name in tfrecord_file_names:
        tfrecord_paths.append(os.path.join(tfrecord_folder_path,tfrecord_name))
    return tfrecord_paths

#endregion

#region Write dataset to TFRECORD

def save_to_tfrecord(dataset,save_path,images_per_file=100):


    # the index of images flowing into each tfrecord file
    num = 0
    # the index of the tfrecord file
    recordFileNum = 0

    # name format of the tfrecord files
    recordFileName = ("dataset.tfrecords-%.3d" % recordFileNum)
    # tfrecord file writer
    writer = tf.io.TFRecordWriter(os.path.join(save_path , recordFileName))

    for image, label in dataset:
        num += 1
        print("Writing to tfrecord file %.2d " % recordFileNum , ", image %.3d /100" % num )
        if num > images_per_file:
            num = 1
            recordFileNum += 1
            recordFileName = ("dataset.tfrecords-%.3d" % recordFileNum)
            writer = tf.io.TFRecordWriter(os.path.join(save_path , recordFileName))
            print("Creating the %.3d tfrecord file" % recordFileNum)

        example = tf.train.Example(
                features=tf.train.Features(
                feature={
                "img": _bytes(image),
                "label": _bytes(label)}
            ))
        writer.write(example.SerializeToString())
    writer.close()

def _bytes(value):
  value = tf.io.serialize_tensor(value)
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#endregion

#
# #Test
# ds=dataset_from_tfrecord(r"C:\Users\aviceb\PycharmProjects\_Git\tensorflowtemplate01\TFT01\datasets\seal_breach_narrow_crop_64_256_Seg_Seal_Breach_0_small\rev_1_00\ds_training")
# for image,label in ds:
#     print(label.shape)