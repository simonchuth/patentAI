import pickle
from os.path import join
import os


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