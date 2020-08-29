import argparse
import random
import gc

from src.utils.general import pickle_load
from src.utils.general import pickle_save
from src.utils.general import join_path
from src.utils.general import check_mkdir

from src.utils.mp_preprocess import chunk_doc
from src.utils.encode import encode_dnn_dataset
from src.utils.encode import encode_attention_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--extracted_pkl", default=None)
    parser.add_argument("--tensor_folder", default=None)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--random_seed", type=int, default=1)

    args = parser.parse_args()

    if args.data_folder is None:
        extracted_pkl = args.extracted_pkl
        tensor_folder = args.tensor_folder
    else:
        extracted_pkl = join_path(args.data_folder, ['extracted_txt',
                                                     'extracted.pkl'])
        tensor_folder = join_path(args.data_folder, ['tensor',
                                                     args.mode])
        params_path = join_path(args.data_folder, 'params.pkl')

        check_mkdir(tensor_folder)
        try:
            params = pickle_load(params_path)
        except Exception:
            params = {}
        params['encode_data_test_ratio'] = args.test_ratio
        params['encode_data_random_seed'] = args.random_seed
        pickle_save(params, params_path)

    dataset = pickle_load(extracted_pkl)

    random.Random(1).shuffle(dataset)
    test_size = int(len(dataset) * args.test_ratio)
    test_set = dataset[:test_size]
    train_set = dataset[test_size:]

    train_path = join_path(args.data_folder, 'train_data')
    check_mkdir(train_path)
    train_dict_word2idx = encode_knn_dataset(train_set, train_path)

    test_path = join_path(args.data_folder, 'test_data')
    check_mkdir(test_path)
    train_dict_word2idx = encode_knn_dataset(test_set, test_path)

    print('Completed')