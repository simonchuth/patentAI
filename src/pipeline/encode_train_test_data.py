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
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--mode", default='attention')

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
        params['encode_data_chunk_size'] = args.chunk_size
        pickle_save(params, params_path)

    dataset = pickle_load(extracted_pkl)

    random.Random(1).shuffle(dataset)
    test_size = int(len(dataset) * args.test_ratio)
    test_set = dataset[:test_size]
    train_set = dataset[test_size:]

    if args.chunk_size == 0:
        num_chunks = len(train_set)
        train_chunks = chunk_doc(train_set, num_chunks)
    elif len(train_set) > args.chunk_size:
        num_chunks = int(len(train_set) / args.chunk_size)
        train_chunks = chunk_doc(train_set, num_chunks)
    else:
        train_chunks = [train_set]

    print(f'Number of chunks: {len(train_chunks)}')

    for i, dataset in enumerate(train_chunks):
        print(f'Encoding train chunk {i}')
        if args.mode == 'dnn':
            context_tensor, target_tensor = encode_dnn_dataset(dataset)
            output = [context_tensor, target_tensor]
            del context_tensor, target_tensor
            gc.collect()
        if args.mode == 'attention':
            output = encode_attention_dataset(dataset)

        train_name = str(i) + '_train.pkl'
        train_path = join_path(tensor_folder, train_name)
        pickle_save(output, train_path)
        del output
        gc.collect()

    print('Encoding test chunk')
    if args.mode == 'dnn':
        context_tensor, target_tensor = encode_dnn_dataset(dataset)
        output = [context_tensor, target_tensor]
    if args.mode == 'attention':
        output = encode_attention_dataset(dataset)
    test_name = 'test.pkl'
    test_path = join_path(tensor_folder, test_name)
    pickle_save(output, test_path)

    print('Completed')