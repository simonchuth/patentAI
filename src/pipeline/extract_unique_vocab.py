import argparse

from tqdm import tqdm

from src.utils.preprocess_ipos import combine_checkpoint_file
from src.utils.preprocess_ipos import extract_app

from src.utils.extract_pdf_data import extract_unique_vocab

from src.utils.general import pickle_save
from src.utils.general import pickle_load
from src.utils.general import join_path

from src.utils.encode import encode_data


def mp_unique_word(L, chunk_app_list):
    unique_word = extract_unique_vocab(chunk_app_list)
    L.append(unique_word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_folder", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--data_folder", default=None)
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

    unique_word = {}
    for app in tqdm(app_list):
        definition = app[2].lower().split(' ')
        definition = [w for w in definition if len(w) < args.max_length]
        for word in definition:
            try:
                unique_word[word]
            except KeyError:
                unique_word[word] = encode_data(word)

    print(f'Number of unique words: {len(unique_word)}')

    print(f'Saving to {savepath}')
    pickle_save(unique_word, savepath)
