import random

import numpy as np

from tqdm import tqdm

import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer  # noqa: F401

import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from src.utils.extract_pdf_data import extract_term_from_definition

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
            vocab_dict[unique_words[i].lower()] = \
                encoder([unique_words[i].lower()])
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


def encode_dnn_dataset(dataset, term_pattern=r'".+?"'):
    context_tensor = []
    target_tensor = []

    for app in tqdm(dataset):
        intro = app[0]
        claims = ' '.join(app[1])
        intro_tensor = encode_data(intro)
        claims_tensor = encode_data(claims)

        definitions = app[2]
        for def_entry in definitions:
            term = extract_term_from_definition(def_entry)
            term_tensor = encode_data(term)

            definition_tokens = text_to_word_sequence(def_entry)
            definition_tokens.append('STOPSTOPSTOP')

            for i in range(4, len(definition_tokens)):
                preword = ' '.join(definition_tokens[:i])
                target_word = definition_tokens[i]

                preword_tensor = encode_data(preword)
                target_word_tensor = encode_data(target_word)
                len_preword_tensor = encode_preword_len(preword)

                context = [intro_tensor,
                           claims_tensor,
                           term_tensor,
                           preword_tensor,
                           len_preword_tensor]

                context_combined = tf.concat(context, axis=1)
                context_tensor.append(context_combined)
                target_tensor.append(target_word_tensor)

    context_tensor = tf.concat(context_tensor, axis=0)
    target_tensor = tf.concat(target_tensor, axis=0)
    return context_tensor, target_tensor


def encode_attention_dataset(dataset):
    output_tensor_list = []
    for app in tqdm(dataset):
        intro = app[0]
        claims = ' '.join(app[1])

        intro_token = text_to_word_sequence(intro)
        claims_token = text_to_word_sequence(claims)

        if (len(intro_token) < 400) or (len(claims_token) < 300):
            continue
        else:
            intro_token = intro_token[:400]
            claims_token = claims_token[:300]

        definitions = app[2]
        def_tensor_list = []
        for def_entry in definitions:
            term = extract_term_from_definition(def_entry)
            def_entry_tokens = text_to_word_sequence(def_entry)
            def_entry_tokens.append('<STOP>')
            for i, token in enumerate(def_entry_tokens):
                if i < 3:
                    continue
                decoder_input_data = def_entry_tokens[i-3:i]
                decoder_target_data = def_entry_tokens[i]
                def_tensor_list.append([encode_data(term),
                                        encode_data(decoder_input_data),
                                        encode_data(decoder_target_data)])

        intro_tensor = encode_data(intro_token)
        claims_tensor = encode_data(claims_token)

        output_tensor_list.append([intro_tensor,
                                   claims_tensor,
                                   def_tensor_list])

    return output_tensor_list


def encode_preword_len(preword):
    len_preword = np.array([len(preword)/500,
                            len(preword)/1000,
                            len(preword)/1500,
                            len(preword)/2000])

    len_preword_tensor = convert_to_tensor(len_preword,
                                           dtype=tf.float32)

    len_preword_tensor = tf.reshape(len_preword_tensor, [1, 4])

    return len_preword_tensor
