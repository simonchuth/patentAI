import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer  # noqa: F401
from tqdm import tqdm

encoder_path = 'encoder/use4_model/'
encoder = hub.load(encoder_path)


def encode_dict(unique_words):
    vocab_dict = {}
    for word in tqdm(unique_words):
        vocab_dict[word] = encoder([word])
    return vocab_dict
