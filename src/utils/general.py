import pickle
from os.path import join
import os
import numpy as np


def pickle_save(data_var, pkl_path, mode='wb'):
    with open(pkl_path, mode) as pkl_file:
        pickle.dump(data_var, pkl_file)


def pickle_load(pkl_path, mode='rb'):
    with open(pkl_path, mode) as pkl_file:
        data = pickle.load(pkl_file)
    return data


def join_path(root_path, path_list):
    if isinstance(path_list, str):
        path_list = [path_list]

    joined_path = root_path
    for path in path_list:
        joined_path = join(joined_path, path)
    return joined_path


def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def count_word(count_dict, word):
    try:
        count_dict[word] = count_dict[word] + 1
    except KeyError:
        count_dict[word] = 1
    return count_dict


def setup_folder(data_folder):
    search_chunks = join_path(data_folder, 'search_chunks')
    models = join_path(data_folder, 'models')
    extracted_txt = join_path(data_folder, 'extracted_txt')
    tensor = join_path(data_folder, 'tensor')
    vocab = join_path(data_folder, 'vocab')
    definition = join_path(data_folder, 'definition')
    check_mkdir(search_chunks)
    check_mkdir(models)
    check_mkdir(extracted_txt)
    check_mkdir(tensor)
    check_mkdir(vocab)
    check_mkdir(definition)


def softmax(x):
    """ Normalize a numeric array with softmax function

    :param x: Numeric data array
    :type x: np.array
    :return: Softmax normalized data
    :rtype: np.array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
