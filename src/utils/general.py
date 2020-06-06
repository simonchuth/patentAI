import pickle


def pickle_save(data_var, pkl_path, mode='wb'):
    with open(pkl_path, mode) as pkl_file:
        pickle.dump(data_var, pkl_file)


def pickle_load(pkl_path, mode='rb'):
    with open(pkl_path, mode) as pkl_file:
        data = pickle.load(pkl_file)
    return data
