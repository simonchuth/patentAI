from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf

from src.model.model import DNN
from src.model.model import predict_word
from src.utils.encode import encode_data
from src.utils.general import pickle_load

model_path = 'models/model_on_data12x20.h5'
vocab_dict_path = 'pkl_files/unique_vocab/unique_dict.pkl'

model = DNN()
model.load_model(model_path)

vocab_dict = pickle_load(vocab_dict_path)
vocab_dict['STOPSTOPSTOP'] = encode_data('STOPSTOPSTOP')


class Inputs(BaseModel):
    claim: str
    intro: str
    term: str


app = FastAPI()


@app.get("/")
def root():
    return {'Status': 'Connection Established'}


@app.post("/predict")
def predict(inputs: Inputs):

    inputs = inputs.dict()

    claim = inputs['claim']
    intro = inputs['intro']
    term = inputs['term']
    pre_word = 'The term "' + term + '"'

    context = [intro, claim]
    context_tensor = encode_data(context)
    context_tensor = tf.reshape(context_tensor, [1, 1024])

    term_tensor = encode_data(term)

    # Add new word from context to dictionary
    word_from_context = (set(' '.join(context).split(' ')))
    instance_dict = vocab_dict.copy()
    print('Adding new word from context to vocabulary')
    for word in word_from_context:
        instance_dict[word.lower()] = encode_data(word.lower())

    new_word = ''
    while True:
        pre_word = pre_word + ' ' + new_word.lower()
        pre_word = pre_word.strip()
        pre_word_tensor = encode_data(pre_word)
        context_combined = tf.concat([context_tensor,
                                      term_tensor,
                                      pre_word_tensor], axis=1)
        pred_tensor = model.predict(context_combined)
        new_word = predict_word(pred_tensor, instance_dict, new_word)
        print(pre_word)
        if new_word == 'STOPSTOPSTOP':
            break

    return {'Result': pre_word}
