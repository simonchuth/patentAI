import gc
import argparse

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
    parser.add_argument("--layers_num", default=[2048,
                                                 2048,
                                                 1024,
                                                 1024,
                                                 512,
                                                 512])
    parser.add_argument("--layers_act", default=['tanh',
                                                 'linear',
                                                 'tanh',
                                                 'linear',
                                                 'tanh',
                                                 'linear'])
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--es_patience", default=2)

    args = parser.parse_args()

    if args.data_folder is None:
        model_output = args.model_output
        tensor_folder = args.tensor_folder
    else:
        date = datetime.datetime.today().strftime('%Y_%m_%d')
        if args.remark is not None:
            date = date + '_' + args.remark
        model_output = join_path(args.data_folder, ['models', date])
        check_mkdir(model_output)
        tensor_folder = join_path(args.data_folder, 'tensor')

    tensor_list = listdir(tensor_folder)

    model = DNN(model_output, es_patience=args.es_patience, callbacks=['es'])

    if args.data_folder is not None:
        params_path = join_path(args.data_folder, 'params.pkl')
        try:
            params = pickle_load(params_path)
        except Exception:
            params = {}
        model_params = model.get_params()
        model_params['epochs'] = args.epochs
        model_params['es_patience'] = args.es_patience
        model_params['layers_num'] = args.layers_num
        model_params['layers_act'] = args.layers_act
        model_params['pretrain_path'] = args.pretrain_path
        model.set_params(model_params)

        model_params = model.get_params()
        for key in model_params.keys():
            key_name = 'model_params_' + key
            params[key_name] = model_params[key]
        pickle_save(params, params_path)

    print(params)
    if args.pretrain_path is not None:
        model.load_model(args.pretrain_path)
    else:
        model.compile_model()

    X_test, y_test = pickle_load(join_path(tensor_folder, 'test.pkl'))
    for i in range(50):
        for j, name in enumerate(tensor_list):
            if 'train' in name:
                X_train, y_train = pickle_load(join_path(tensor_folder, name))
                print(f'Cycle {i}: chunks {j}')
                model.fit(X_train, y_train, X_test, y_test)
                del X_train, y_train
                gc.collect()
        output_path = model_params['output_path']
        model_params['final_model'] = join_path(output_path,
                                                ['models', f'cycle_{i}.h5'])
        model.set_params(model_params)
        model.save_model()
        
    model_params['final_model'] = join_path(output_path,
                                            ['models', 'final_model.h5'])
    model.set_params(model_params)
    model.save_model()
