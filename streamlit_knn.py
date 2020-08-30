import streamlit as st

import SessionState

from src.model.knn_model import KnnModel

session_state = SessionState.get(knn_model=None,
                                 preword=None)
data_folder = '~/data/patentAI/2_bio_20yrs'

if session_state.knn_model is None:
    model = KnnModel()
    model.load_model(data_folder)
    session_state.knn_model = model

st.title('PatentAI')

intro = st.text_area('Paste the introduction here')
claim = st.text_area('Paste the claims here')
term = st.text_input('Enter the term here', 'Term')
if session_state.preword is None:
    session_state.preword = f'The term "{term}"'
preword = st.text_input('Definition:', session_state.preword)
preword = preword.strip()

if (intro != '') and (claim != '') and (term != '') and (preword != ''):
    model = session_state.knn_model
    sorted_result = model.predict(intro, claim, term, preword)
    if st.button(f'{sorted_result[0][0]}: {sorted_result[0][1]*100}%'):
        session_state.preword = session_state.preword + ' ' + sorted_result[0][0]
    if st.button(f'{sorted_result[1][0]}: {sorted_result[1][1]*100}%'):
        session_state.preword = session_state.preword + ' ' + sorted_result[1][0]
    if st.button(f'{sorted_result[2][0]}: {sorted_result[2][1]*100}%'):
        session_state.preword = session_state.preword + ' ' + sorted_result[2][0]
    if st.button(f'{sorted_result[3][0]}: {sorted_result[3][1]*100}%'):
        session_state.preword = session_state.preword + ' ' + sorted_result[3][0]
    if st.button(f'{sorted_result[4][0]}: {sorted_result[4][1]*100}%'):
        session_state.preword = session_state.preword + ' ' + sorted_result[4][0]
