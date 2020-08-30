import numpy as np

import faiss

from src.utils.general import join_path
from src.utils.general import pickle_load
from src.utils.general import softmax
from src.utils.encode import encode_data


class KnnModel:
    def __init__(self):
        self.index = None
        self.dict_idx2word = None
        self.vals = None

    def load_model(self, data_folder):
        index_path = join_path(data_folder, ['train_data', 'index.trained'])
        vals_path = join_path(data_folder, ['train_data', 'vals.trained'])
        dict_path = join_path(data_folder, ['train_data', 'dict_word2idx.pkl'])

        dict_word2idx = pickle_load(dict_path)
        self.dict_idx2word = {v: k for k, v in dict_word2idx.items()}

        self.index = faiss.read_index(index_path)

        self.vals = np.load(vals_path)

        print('Model loaded')

    def predict(self,
                input_intro,
                input_claims,
                intro_term,
                input_preword,
                k_neighbor=16):
        intro_tensor = encode_data(input_intro).numpy()
        claims_tensor = encode_data(input_claims).numpy()
        term_tensor = encode_data(intro_term).numpy()
        preword_tensor = encode_data(input_preword).numpy()

        keys_np = np.concatenate([intro_tensor,
                                  claims_tensor,
                                  term_tensor,
                                  preword_tensor], axis=1)

        distances, indices = self.index.search(keys_np, k_neighbor)
        result_text_list = self._process_faiss_output(distances, indices)
        result_text = result_text_list[0]
        sorted_result = sorted(result_text.items(),
                               key=lambda item: item[1],
                               reverse=True)
        return sorted_result

    # Auxiliary  Functions

    def _process_faiss_output(self, distances, indices):

        values_list = [self.vals[idx] for idx in indices]
        probs_list = [softmax(-dist) for dist in distances]

        result_text_list = []
        for i, values in enumerate(values_list):
            probs = probs_list[i]
            result_text = self._aggregate_probas(values, probs)
            result_text_list.append(result_text)

        return result_text_list

    def _aggregate_probas(self, values, probs):

        result_idx = {}
        dict_idx2token = self.params['dict_idx2token']

        for i, val in enumerate(values):
            if val[0] not in result_idx.keys():
                result_idx[val[0]] = probs[i]
            else:
                result_idx[val[0]] += probs[i]

        result_text = {dict_idx2token[k]: v for k, v in result_idx.items()}

        return result_text
