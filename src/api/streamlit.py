import streamlit as st

import requests
import json

from os.path import join

st.title('Singlish Sentence Classifier')

api_base = st.text_input('Please enter path to the Sentence Classifier API',
                         'http://192.168.1.111:8000/')

try:
    connection_check = requests.get(api_base)
    st.success('Connected to Sentence Classifier API successfully')

except Exception:
    st.warning('Connection to Sentence Classifier API cannot be established, \
                please check the path')
    st.button('Refresh')
