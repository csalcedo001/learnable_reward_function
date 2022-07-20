import os

import json
import pickle
from xlab.utils import merge_dicts



def get_config_from_file(path):
    if not os.path.exists(path):
        raise Exception('Error: path {} does not exist.'.format(path))
    
    extension = os.path.splitext(path)[-1][1:]

    if extension == 'json':
        loader = json
        open_type = 'r'
    elif extension == 'pkl':
        loader = pickle
        open_type = 'rb'
    else:
        raise Exception('Error: unsupported extension {}.'.format(extension))

    try:
        with open(path, open_type) as in_file:
            config = loader.load(in_file)
    except:
        raise Exception('Error: unable to open {}.'.format(path))
    
    return config


def get_config_from_string(config_str):
    try:
        config_arg = json.loads(config_str)
    except:
        config_arg = config_str
    
    if type(config_arg) != list:
        config_list = [config_arg]
    else:
        config_list = config_arg

    config = {}
    for config_list_elem in config_list:
        type_config = type(config_list_elem)

        if type_config == dict:
            new_config = config_list_elem
        elif type_config == str:
            if os.path.isdir(config_list_elem):
                config_path = os.path.join(config_list_elem, 'config.json')
                # Maybe validate if config.json is found inside folder
            elif os.path.isfile(config_list_elem):
                config_path = config_list_elem
            else:
                error_msg = \
                    "Error: string '{}' doesn't describe a folder " \
                    "or a file."
                raise Exception(error_msg.format(config_list_elem))

            new_config = get_config_from_file(config_path)
        
        config = merge_dicts(config, new_config)
    
    return config