import argparse

from src.utils.encode import encode_dict

from src.utils.general import pickle_save
from src.utils.general import pickle_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unique_word_pkl", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--vocab_size", type=int, default=1000000)
    parser.add_argument("--max_length", type=int, default=None)

    args = parser.parse_args()

    unique_word = pickle_load(args.unique_word_pkl)
    print('Encoding')
    vocab_dict = encode_dict(unique_word,
                             vocab_size=args.vocab_size,
                             max_length=args.max_length)
    print('Completed encoding')

    if args.savepath is not None:
        print(f'Saving to {args.savepath}')
        pickle_save(vocab_dict, args.savepath)
