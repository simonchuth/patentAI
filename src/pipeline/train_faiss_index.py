import argparse
import random
import time
import gc

import faiss
import numpy as np

from os import listdir

from src.utils.general import join_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--ncentroids", type=int, default=256)
    parser.add_argument("--code_size", type=int, default=64)
    parser.add_argument("--probe", type=int, default=4)

    args = parser.parse_args()

    train_folder_path = join_path(args.data_folder, 'train_data')

    train_files = [npfile for npfile in listdir(train_folder_path)
                   if npfile.endswith('keys.npy')]

    random.Random(args.random_seed).shuffle(train_files)
    sample_size = min(1000, int(len(train_files)*0.1))
    sample_files = train_files[sample_size:]

    sample_file_path = join_path(train_folder_path, sample_files[0])
    sample_np = np.load(sample_file_path)
    dimension = sample_np.shape[1]

    sample_np_list = []
    print(f'Loading sample data, dimension {dimension}')
    for npfile in sample_files:
        fname = join_path(train_folder_path, npfile)
        sample_np = np.load(fname)
        if sample_np.shape[1] == dimension:
            sample_np_list.append(sample_np)
        del sample_np

    sample_np = np.concatenate(sample_np_list, axis=0)
    del sample_np_list

    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer,
                             dimension,
                             args.ncentroids,
                             args.code_size,
                             8)
    index.nprobe = args.probe

    print('Training FAISS Index')
    start_time = time.time()
    index.train(sample_np.astype(np.float32))
    print(f'Training took {time.time() - start_time} s')
    del sample_np

    print('Adding keys')
    start_time = time.time()
    start_idx = 0
    vals_np_list = []
    for i, npfile in enumerate(train_files):
        try:
            fname = join_path(train_folder_path, npfile)
            to_add = np.load(fname)
            vals_fname = npfile[:-8] + 'vals.npy'
            vals_fname = join_path(train_folder_path, vals_fname)
            to_add_vals = np.load(vals_fname)
        except Exception:
            continue

        if to_add.shape[1] != dimension:
            continue
        end_idx = start_idx + to_add.shape[0]
        index.add_with_ids(to_add.astype(np.float32), np.arange(start_idx,
                                                                end_idx))
        vals_np_list.append(to_add_vals)
        start_idx = end_idx

        if (start_idx % 10000) == 0:
            print(f'Added {start_idx} tokens so far')
            print(f'Writing files {i}/{len(train_files)}')
        del to_add
        gc.collect()

    print(f'Adding keys took {time.time() - start_time} s')

    print('Saving faiss index')
    index_fpath = join_path(train_folder_path, 'index.trained')
    faiss.write_index(index, index_fpath)

    vals_np = np.concatenate(vals_np_list, axis=0)
    vals_fpath = join_path(train_folder_path, 'vals.trained')
    np.save(vals_fpath, vals_np)

    print(f'Completed: Trained {end_idx} entries')
