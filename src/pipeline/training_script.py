import gc
import argparse

import tensorflow as tf

from src.utils.general import pickle_load
from src.utils.general import pickle_save
from src.utils.general import join_path
from src.utils.general import check_mkdir

from src.model.model import DNN

import datetime

from os import listdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain_path", default=None)
    parser.add_argument("--model_output", default=None)
    parser.add_argument("--tensor_folder", default=None)
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--remark", default=None)

    args = parser.parse_args()

    if args.data_folder is None:
        model_output = args.model_output
        tensor_folder = args.tensor_folder
    else:
        date = datetime.datetime.today()
        date = date.strftime('%Y_%m_%d') + '_' + args.remark
        model_output = join_path(args.data_folder, ['models', date])
        check_mkdir(model_output)
        tensor_folder = join_path(args.data_folder, 'tensor')

    tensor_list = listdir(tensor_folder)
    train_tensors = [pickle_load(join_path(tensor_folder, name))
                     for name in tensor_list
                     if 'train' in name]
    X_train_list = [X for X, y in train_tensors]
    y_train_list = [y for X, y in train_tensors]
    X_train = tf.concat(X_train_list, axis=0)
    y_train = tf.concat(y_train_list, axis=0)
    X_test, y_test = pickle_load(join_path(tensor_folder, 'test.pkl'))

    del X_train_list, y_train_list
    gc.collect()

    model = DNN(model_output)

    if args.pretrain_path is not None:
        model.load_model(args.pretrain_path)

    if args.data_folder is not None:
        params_path = join_path(args.data_folder, 'params.pkl')
        params = pickle_load(params_path)
        model_params = model.get_params()
        for key in model_params.keys():
            key_name = 'model_params_' + key
            params[key_name] = model_params[key]
        pickle_save(params, params_path)

    model.fit(X_train, y_train, X_test, y_test)

    model.save_model()