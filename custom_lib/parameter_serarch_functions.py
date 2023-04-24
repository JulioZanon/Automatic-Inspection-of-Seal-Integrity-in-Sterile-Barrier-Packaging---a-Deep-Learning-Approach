
# imports
import numpy as np
from custom_lib.model_build_funtions import *


def translate_from_dictionary(dictionary ={} ,string='', id = 0):
    """
    Fucntion traslate from a dictionary {string, id). If a strign is passed, then the function returns the id.
    if an id is passed then the funtion will return the string.
    :param dictionary: with the form {string, id}
    :param string: Optional. Function needs either string or id.
    :param id: Optional. Function needs either string or id.
    :return: string and id.
    """
    if string != '':
        id = dictionary[string]
    elif id > 0:
        string = list(dictionary.items())[id - 1][0]
    return string, id


def build_serach_array(new_parameter, parameter_name, search_array=np.array([]), name_array = [], dict={}):
    """
    This function add new serach parameter array to the existing serach array
    :param new_parameter:
    :param search_array:
    :param dict:
    :return:
    """
    new_parameter = np.array(new_parameter)
    # Convert strings into integers based on dictionary
    if '<U' in str(new_parameter.dtype) and len(dict) > 0:
        new_parameter_id = np.zeros(len(new_parameter))
        i = 0
        for parameter_str in new_parameter:
            new_parameter_id[i] = translate_from_dictionary(dict, string=parameter_str)[1]
            i += 1
        new_parameter = new_parameter_id
    # the first time the function is call, the new parameter becomes the search array
    if search_array.shape[0] == 0:
        search_array = np.expand_dims(new_parameter, axis=0).T
        name_array = [parameter_name]
    else:
        name_array.append(parameter_name)
        # convert all data types to be the same
        if new_parameter.dtype != search_array.dtype:
            if new_parameter.dtype == 'float64' or search_array.dtype == 'float64':
                np.float64(new_parameter)
                np.float64(search_array)
            elif new_parameter.dtype == 'float32' or search_array.dtype == 'float632':
                np.float32(new_parameter)
                np.float32(search_array)
            elif new_parameter.dtype == 'float16' or search_array.dtype == 'float616':
                np.float16(new_parameter)
                np.float16(search_array)
        # combine arrays with for loops
        new_search_array = np.array([])
        first_loop = True
        for search_a in search_array:
            for new_p in new_parameter:
                if first_loop:
                    new_search_array = np.expand_dims(np.append(search_a, new_p), axis=0)
                    first_loop = False
                else:
                    new_search_array = np.append(new_search_array, np.expand_dims(np.append(search_a, new_p), axis=0), axis=0)
        search_array = new_search_array
    return search_array, name_array


def del_from_serach_array(del_parameters, search_array):
    #TODO: delete combination of parameters from search array
    return search_array


