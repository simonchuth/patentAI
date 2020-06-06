import argparse

from src.utils.encode import encode_dict

from src.utils.general import pickle_save
from src.utils.general import pickle_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unique_word_pkl", default=None)
    parser.add_argument("--savepath", default=None)

    args = parser.parse_args()

    unique_word = pickle_load(args.unique_word_pkl)
    vocab_dict = encode_dict(unique_word)

    if args.savepath is not None:
        print(f'Saving to {args.savepath}')
        pickle_save(vocab_dict, args.savepath)