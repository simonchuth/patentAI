import argparse

from src.utils.general import pickle_load
from src.utils.general import pickle_save
from src.utils.general import join_path

from src.utils.mp_preprocess import chunk_doc
from src.utils.encode import encode_dataset

import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--extracted_pkl", default=None)
    parser.add_argument("--tensor_folder", default=None)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=400)

    args = parser.parse_args()

    if args.data_folder is None:
        extracted_pkl = args.extracted_pkl
        tensor_folder = args.tensor_folder
    else:
        extracted_pkl = join_path(args.data_folder, ['extracted_txt',
                                                     'extracted.pkl'])
        tensor_folder = join_path(args.data_folder, 'tensor')

    dataset = pickle_load(extracted_pkl)

    random.Random(1).shuffle(dataset)
    test_size = min(int(len(dataset) * args.test_ratio), args.chunk_size)
    test_set = dataset[:test_size]
    train_set = dataset[test_size:]

    if len(train_set) > args.chunk_size:
        num_chunks = int(len(train_set) / args.chunk_size)
        train_chunks = chunk_doc(train_set, num_chunks)
    else:
        train_chunks = [train_set]

    print(f'Number of chunks: {len(train_chunks)}')

    for i, dataset in enumerate(train_chunks):
        print(f'Encoding train chunk {i}')
        context_tensor, target_tensor = encode_dataset(dataset)
        output = [context_tensor, target_tensor]
        train_name = str(i) + '_train.pkl'
        train_path = join_path(tensor_folder, train_name)
        pickle_save(output, train_path)

    print('Encoding test chunk')
    context_tensor, target_tensor = encode_dataset(test_set)
    output = [context_tensor, target_tensor]
    test_name = 'test.pkl'
    test_path = join_path(tensor_folder, test_name)
    pickle_save(output, test_path)

    print('Completed')