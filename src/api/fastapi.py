from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf

from src.model.model import DNN
from src.model.model import predict_word
from src.utils.encode import encode_data
from src.utils.encode import encode_preword_len
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

    claims = inputs['claim']
    intro = inputs['intro']
    term = inputs['term']
    preword = 'The term "' + term + '"'

    intro_tensor = encode_data(intro)
    claims_tensor = encode_data(claims)

    term_tensor = encode_data(term)

    # Add new word from context to dictionary
    intro_claims = [intro, claims]
    word_from_context = (set(' '.join(intro_claims).split(' ')))
    instance_dict = vocab_dict.copy()
    print('Adding new word from context to vocabulary')
    for word in word_from_context:
        if word.lower() not in instance_dict.keys():
            instance_dict[word.lower()] = encode_data(word.lower())

    new_word = ''
    while True:
        preword = preword + ' ' + new_word.lower()
        preword = preword.strip()
        preword_tensor = encode_data(preword)
        len_preword_tensor = encode_preword_len(preword)

        context = [intro_tensor,
                   claims_tensor,
                   term_tensor,
                   preword_tensor,
                   len_preword_tensor]

        context_combined = tf.concat(context, axis=1)

        pred_tensor = model.predict(context_combined)
        new_word = predict_word(pred_tensor, instance_dict, new_word)
        print(preword)
        if new_word == 'stopstopstop':
            break

    return {'Result': preword}
