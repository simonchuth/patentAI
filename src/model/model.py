import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from scipy.spatial.distance import euclidean


class BaseAPI:
    def __init__(self, output_path=None, es_patience=10):
        if output_path is None:
            callbacks = None
            final_model = None
        else:
            # Make directory for outputs
            os.mkdir(output_path + '/logs')
            os.mkdir(output_path + '/checkpoints')
            os.mkdir(output_path + '/models')

            tensorboard_logdir = output_path + '/logs'
            checkpoint_model = output_path + '/checkpoints/epoch{epoch}.h5'
            best_model = output_path + '/models/best_model.h5'
            final_model = output_path + '/models/final_model.h5'

            callbacks = [EarlyStopping(patience=es_patience),
                         ModelCheckpoint(filepath=checkpoint_model,
                                         monitor='val_loss'),
                         ModelCheckpoint(filepath=best_model,
                                         monitor='val_loss',
                                         save_best_only=True),
                         TensorBoard(log_dir=tensorboard_logdir)]

        self.params = {'batch_size': 10000,
                       'epochs': 1000,
                       'output_path': output_path,
                       'callbacks': callbacks,
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

        old_params = self.params.copy()
        self.params = update_dict(self.params, update_params)
        # Compile model again if the parameters were changed
        if old_params != self.params:
            self.compile_model()

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
                 es_patience=5,
                 input_dim=2048,
                 output_dim=512):

        BaseAPI.__init__(self, output_path, es_patience)
        DNN_params = {'layers': [1024, 1024, 512, 512],
                      'optimizer': Adam(),
                      'loss': CosineSimilarity(),
                      'metrics': ['cosine_similarity'],
                      'input_dim': input_dim,
                      'output_dim': output_dim}
        self.set_params(DNN_params)

    def compile_model(self):

        self.model = Sequential()

        for i, layer in enumerate(self.params['layers']):
            if i == 0:
                self.model.add(Dense(layer,
                                     input_dim=self.params['input_dim'],
                                     activation='relu',
                                     name='Input'))
            else:
                self.model.add(Dense(layer,
                                     activation='relu',
                                     name=f'Hidden_{i}'))

        self.model.add(Dense(self.params['output_dim'],
                             activation='relu',
                             name='Output'))

        self.model.compile(optimizer=self.params['optimizer'],
                           loss=self.params['loss'],
                           metrics=self.params['metrics'])

        self.model.summary()


def update_dict(base_dict, update_dict):

    for key in update_dict:
        base_dict[key] = update_dict[key]
    return base_dict


def predict_word(tensor, vocab_dict):
    sorted_neigh = sorted(vocab_dict.keys(),
                        key=lambda word: euclidean(vocab_dict[word], tensor))
    return sorted_neigh[0]
