import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Attention, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from scipy.spatial.distance import euclidean

from src.utils.general import join_path
from src.utils.general import check_mkdir


class BaseAPI:
    def __init__(self,
                 output_path,
                 es_patience,
                 callbacks):

        if output_path is None:
            callbacks = None
            final_model = None
            cb = None

        else:
            # Make directory for outputs
            check_mkdir(join_path(output_path, 'checkpoints'))
            check_mkdir(join_path(output_path, 'models'))

            checkpoint_model = join_path(output_path, ['checkpoints',
                                                       'epoch{epoch}.h5'])
            best_model = join_path(output_path, ['models', 'best_model.h5'])
            final_model = join_path(output_path, ['models', 'final_model.h5'])

            cb = []

            if 'es' in callbacks:
                cb.append(EarlyStopping(patience=es_patience,
                                        restore_best_weights=True))

            if 'checkpoint_model' in callbacks:
                cb.append(ModelCheckpoint(filepath=checkpoint_model,
                                          monitor='val_loss'))

            if 'best_model' in callbacks:
                cb.append(ModelCheckpoint(filepath=best_model,
                                          monitor='val_loss',
                                          save_best_only=True))

            if len(cb) == 0:
                cb = None

        self.params = {'batch_size': 10000,
                       'epochs': 1000,
                       'output_path': output_path,
                       'callbacks': cb,
                       'final_model': final_model}

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train,
                       batch_size=self.params['batch_size'],
                       epochs=self.params['epochs'],
                       callbacks=self.params['callbacks'],
                       validation_data=(X_val, y_val))

    def predict(self, X):

        return self.model.predict(X)

    def get_params(self):

        return self.params

    def set_params(self, update_params):
        self.params = update_dict(self.params, update_params)

    def save_model(self):

        if self.params['final_model'] is None:
            raise ValueError("Save path for final model not provided, please\
                             provide it via set_params with key 'final_model'")
        self.model.save(self.params['final_model'])

    def load_model(self, model_path):

        self.model = load_model(model_path)

    def evaluate(self, X, y):
        self.model.evaluate(X, y)


class DNN(BaseAPI):
    def __init__(self,
                 output_path=None,
                 es_patience=15,
                 input_dim=2052,
                 output_dim=512,
                 callbacks=['es, checkpoint_model, best_model']):

        BaseAPI.__init__(self, output_path, es_patience, callbacks)
        DNN_params = {'layers_num': [2048, 1024, 1024, 512],
                      'layers_act': ['tanh', 'tanh', 'tanh', 'linear'],
                      'optimizer': Adam(),
                      'loss': CosineSimilarity(),
                      'metrics': ['cosine_similarity'],
                      'input_dim': input_dim,
                      'output_dim': output_dim}
        self.set_params(DNN_params)
        self.compile_model()

    def compile_model(self):

        self.model = Sequential()

        for i, layer in enumerate(self.params['layers_num']):
            if i == 0:
                print(f"Input layer: {layer} nodes,\
                      {self.params['layers_act'][i]} activation function")
                self.model.add(Dense(layer,
                                     input_dim=self.params['input_dim'],
                                     activation=self.params['layers_act'][i],
                                     name='Input'))
            else:
                print(f"Hidden_{i}: {layer} nodes,\
                      {self.params['layers_act'][i]} activation function")
                self.model.add(Dense(layer,
                                     activation=self.params['layers_act'][i],
                                     name=f'Hidden_{i}'))

        self.model.add(Dense(self.params['output_dim'],
                             activation='linear',
                             name='Output'))

        self.model.compile(optimizer=self.params['optimizer'],
                           loss=self.params['loss'],
                           metrics=self.params['metrics'])

        self.model.summary()


def update_dict(base_dict, update_dict):

    for key in update_dict:
        base_dict[key] = update_dict[key]
    return base_dict


def predict_word(tensor, vocab_dict, previous_word, word_count):
    sorted_neigh = sorted(vocab_dict.keys(),
                          key=lambda word: euclidean(vocab_dict[word], tensor))
    for word in sorted_neigh:
        if word not in word_count.keys():
            return word.lower()
        elif (word.lower() != previous_word.lower()) and \
             (word_count[word] < 5):
            return word.lower()


class Att_model(BaseAPI):
    def __init__(self,
                 output_path=None,
                 es_patience=15,
                 callbacks=['es, checkpoint_model, best_model']):

        BaseAPI.__init__(self, output_path, es_patience, callbacks)

    def compile_model(self):

        intro_inputs = Input(batch_shape=(10, None, 512),
                             name='intro_inputs')
        claim_inputs = Input(batch_shape=(10, None, 512),
                             name='claim_inputs')
        term_inputs = Input(batch_shape=(10, 1, 512),
                            name='term_inputs')
        decoder_inputs = Input(batch_shape=(10, None, 512),
                               name='decoder_inputs')

        # Encoder stack
        lstm_intro_f = LSTM(512,
                            return_state=True,
                            return_sequences=True)
        lstm_claims_f = LSTM(512,
                             return_state=True,
                             return_sequences=True)
        lstm_intro_b = LSTM(512,
                            return_state=True,
                            go_backwards=True,
                            return_sequences=True)
        lstm_claims_b = LSTM(512,
                             return_state=True,
                             go_backwards=True,
                             return_sequences=True)

        intro_lstm_f_out, intro_f_h, intro_f_c = lstm_intro_f(intro_inputs)
        claims_lstm_f_out, claims_f_h, claims_f_c = lstm_claims_f(claim_inputs)
        intro_lstm_b_out, intro_b_h, intro_b_c = lstm_intro_b(intro_inputs)
        claims_lstm_b_out, claims_b_h, claims_b_c = lstm_claims_b(claim_inputs)

        print(intro_lstm_f_out.shape)
        print(intro_f_h.shape)
        print(intro_f_c.shape)
        print(term_inputs.shape)

        intro_f_att = Attention()([term_inputs, intro_lstm_f_out])
        claims_f_att = Attention()([term_inputs, claims_lstm_f_out])
        intro_b_att = Attention()([term_inputs, intro_lstm_b_out])
        claims_b_att = Attention()([term_inputs, claims_lstm_b_out])

        context_att = tf.concat([intro_f_att,
                                 claims_f_att,
                                 intro_b_att,
                                 claims_b_att], axis=2)

        context_h_state = tf.concat([intro_f_h,
                                     claims_f_h,
                                     intro_b_h,
                                     claims_b_h], axis=1)

        context_c_state = tf.concat([intro_f_c,
                                     claims_f_c,
                                     intro_b_c,
                                     claims_b_c], axis=1)

        encoder_states = [context_h_state, context_c_state]

        # Decoder stack
        lstm_decode = LSTM(2048, return_sequences=True, return_state=True)
        decoder_output, _, _ = lstm_decode(decoder_inputs,
                                           initial_state=encoder_states)

        dnn_input = tf.concat([context_att, decoder_output], axis=2)

        dnn_output = Dense(2048, activation='tanh', name='DNN_1')(dnn_input)
        dnn_output = Dense(2048, activation='tanh', name='DNN_2')(dnn_output)
        dnn_output = Dense(1024, activation='tanh', name='DNN_3')(dnn_output)
        dnn_output = Dense(1024, activation='tanh', name='DNN_4')(dnn_output)
        dnn_output = Dense(512, activation='linear', name='DNN_5')(dnn_output)

        self.model = Model([intro_inputs,
                            claim_inputs,
                            term_inputs,
                            decoder_inputs],
                           dnn_output)

        self.model.compile(optimizer=Adam(),
                           loss=CosineSimilarity(),
                           metrics=['cosine_similarity'])

        self.model.summary()

