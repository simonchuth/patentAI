import random

import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer  # noqa: F401


encoder_path = 'encoder/use4_model/'
encoder = hub.load(encoder_path)


def encode_dict(unique_words, vocab_size=1000000, max_length=50):
    vocab_dict = {}

    unique_words = list(unique_words)

    idx = list(range(0, len(unique_words)))
    random.shuffle(idx)

    count = 0
    for i in idx:
        print(f'Percentage Completed: {count/vocab_size*100}%')
        if (len(unique_words[i]) < max_length) and \
                (unique_words[i].isalpha()) and (count < vocab_size):
            vocab_dict[unique_words[i]] = encoder([unique_words[i]])
            count += 1
        elif count >= vocab_size:
            break
    return vocab_dict


def encode_data(input):

    if isinstance(input, str):
        list_input = [input]
    elif isinstance(input, list):
        list_input = input
    else:
        list_input = [str(input)]

    return encoder(list_input)
