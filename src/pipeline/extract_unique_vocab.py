import argparse

from multiprocessing import Process, Manager, cpu_count

from src.utils.preprocess_ipos import combine_checkpoint_file
from src.utils.preprocess_ipos import extract_app

from src.utils.extract_pdf_data import extract_unique_vocab

from src.utils.general import pickle_save
from src.utils.general import pickle_load

from src.utils.mp_preprocess import chunk_doc


def mp_unique_word(L, chunk_app_list):
    unique_word = extract_unique_vocab(chunk_app_list)
    L.append(unique_word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_folder", default=None)
    parser.add_argument("--combined_file", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--mp", type=bool, default=False)

    args = parser.parse_args()

    if args.checkpoint_folder is not None:
        output_list = combine_checkpoint_file(args.checkpoint_folder)
    elif args.combined_file is not None:
        output_list = pickle_load(args.combined_file)
    else:
        raise FileNotFoundError('No file path was provided')

    app_list = extract_app(output_list)
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

    if args.savepath is not None:
        print(f'Saving to {args.savepath}')
        pickle_save(unique_word, args.savepath)
