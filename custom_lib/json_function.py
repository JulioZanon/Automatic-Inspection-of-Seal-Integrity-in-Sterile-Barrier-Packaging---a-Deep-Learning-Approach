import json
from custom_lib.model_build_funtions import *
from custom_lib.parameter_serarch_functions import *
from .dataset_functions import *

def write_to_json(data,json_file_path):
    # Serializing json
    json_object = json.dumps(data,indent = 5)
    # Writing to sample.json
    with open(json_file_path, "w") as outfile:
        outfile.write(json_object)
    return 0

def read_from_json(file_path):
    # Opening JSON file
    with open(file_path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    return json_object

def array_to_dictionary(keys=[], values=[]):

    dictionary = dict()
    if len(keys) == len(values):
        for i in range(len(keys)):
            if keys[i] == 'optimiser_function': dictionary[keys[i]] = translate_from_dictionary(dictionary=dict_optimiser_function,id = int(values[i]))[0]
            elif keys[i] == 'loss_function': dictionary[keys[i]] = translate_from_dictionary(dictionary=dict_loss_function, id=int(values[i]))[0]
            elif keys[i] == 'transfer_models': dictionary[keys[i]] = translate_from_dictionary(dictionary=dict_transfer_models, id=int(values[i]))[0]
            elif keys[i] == 'flatten_layers': dictionary[keys[i]] = translate_from_dictionary(dictionary=dict_flatten_layers, id=int(values[i]))[0]
            elif keys[i] == 'img_extensions': dictionary[keys[i]] = translate_from_dictionary(dictionary=dict_image_format, id=int(values[i]))[0]
            else: dictionary[keys[i]] = values[i]
    else: print('Fail to create dictionary as list lengths are not the same')
    return dictionary

def dictionary_to_array(dictionary = dict()):

    array = []
    for key in dictionary:
        if key == 'optimiser_function': value = translate_from_dictionary(dictionary=dict_optimiser_function, string=dictionary[key])[1]
        elif key == 'loss_function':  value = translate_from_dictionary(dictionary=dict_loss_function,string=dictionary[key])[1]
        elif key == 'transfer_models': value = translate_from_dictionary(dictionary=dict_transfer_models, string=dictionary[key])[1]
        elif key == 'flatten_layers': value = translate_from_dictionary(dictionary=dict_flatten_layers, string=dictionary[key])[1]
        elif key == 'img_extensions': value = translate_from_dictionary(dictionary=dict_image_format, string=dictionary[key])[1]
        else: value = dictionary[key]
        array.append(value)
    return array


def array_to_jason(keys=[], values=[], json_file_path = ''):
    dictionary = array_to_dictionary(keys, values)
    write_to_json(dictionary, json_file_path)
    return dictionary