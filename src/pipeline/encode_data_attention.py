import argparse

import gc

from src.utils.general import pickle_load
from src.utils.general import pickle_save
from src.utils.general import join_path
from src.utils.general import check_mkdir

from src.utils.encode import encode_attention_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--mode", default='attention')

    args = parser.parse_args()

    extracted_pkl = join_path(args.data_folder, ['extracted_txt',
                                                 'extracted.pkl'])
    tensor_folder = join_path(args.data_folder, ['tensor',
                                                 args.mode])
    params_path = join_path(args.data_folder, 'params.pkl')

    check_mkdir(tensor_folder)

    dataset = pickle_load(extracted_pkl)

    for i, app in enumerate(dataset):
        print(f'{i}/{len(dataset)}')
        output = encode_attention_app(dataset)
        for j, definition in enumerate(output[2]):
            save_list = [output[0],
                         output[1],
                         definition[0],
                         definition[1],
                         definition[2]]
            train_name = f'{i}_{j}_train.pkl'
            train_path = join_path(tensor_folder, train_name)
            pickle_save(save_list, train_path)
            del output
            gc.collect()

    print('Completed')
