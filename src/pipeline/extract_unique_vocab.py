import argparse

from multiprocessing import Process, Manager, cpu_count

from src.utils.preprocess_ipos import combine_checkpoint_file
from src.utils.preprocess_ipos import extract_app

from src.utils.extract_pdf_data import extract_unique_vocab

from src.utils.general import pickle_save
from src.utils.general import pickle_load
from src.utils.general import join_path

from src.utils.mp_preprocess import chunk_doc

from src.utils.encode import encode_dict


def mp_unique_word(L, chunk_app_list):
    unique_word = extract_unique_vocab(chunk_app_list)
    L.append(unique_word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_folder", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--mp", type=bool, default=False)
    parser.add_argument("--vocab_size", type=int, default=500000)
    parser.add_argument("--max_length", type=int, default=50)

    args = parser.parse_args()

    if args.data_folder is None:
        if args.checkpoint_folder is not None:
            chunk_list = combine_checkpoint_file(args.checkpoint_folder)
        else:
            raise FileNotFoundError('No file path was provided')
        savepath = args.savepath
    else:
        savepath = join_path(args.data_folder, ['vocab', 'vocab_tensor.pkl'])
        chunk_folder = join_path(args.data_folder, 'search_chunks')
        chunk_list = combine_checkpoint_file(chunk_folder)
        params_path = join_path(args.data_folder, 'params.pkl')
        try:
            params = pickle_load(params_path)
        except Exception:
            params = {}
        params['extract_vocab_vocab_size'] = args.vocab_size
        params['extract_vocab_max_length'] = args.max_length
        pickle_save(params, params_path)

    app_list = extract_app(chunk_list)
    print(f'Total number of applications: {len(app_list)}')

    if args.mp:
        num_worker = cpu_count()
        chunk_list = chunk_doc(app_list, num_worker)
        print(f'Chunked into {len(chunk_list)} chunks')

        with Manager() as manager:
            L = manager.list()
            processes = []
            for chunk_items in chunk_list:
                p = Process(target=mp_unique_word,
                            args=(L, chunk_items))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            list_unique_word = list(L)

        unique_word = set()
        for unique_set in list_unique_word:
            unique_word = unique_word.union(unique_set)

    else:
        unique_word = extract_unique_vocab(app_list)

    num_word = len(unique_word)
    print(f'Number of unique words: {num_word}')

    print('Encoding')
    vocab_size = min(num_word, args.vocab_size)
    vocab_dict = encode_dict(unique_word,
                             vocab_size=vocab_size,
                             max_length=args.max_length)
    print('Completed encoding')

    print(f'Saving to {savepath}')
    pickle_save(unique_word, savepath)
