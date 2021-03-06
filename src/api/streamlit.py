import streamlit as st

import requests
import json

from os.path import join

st.title('PatentAI')

api_base = st.text_input('Please enter path to the PatentAI API',
                         'http://192.168.1.111:8000/')

try:
    connection_check = requests.get(api_base)
    st.success('Connected to PatentAI API successfully')

except Exception:
    st.warning('Connection to PatentAI API cannot be established, \
                please check the path')
    st.button('Refresh')

intro = st.text_area('Paste the introduction here')
claim = st.text_area('Paste the claims here')
term = st.text_input('Please enter the term here')

if (intro != '') and (claim != '') and (term != ''):
    if st.button('Submit'):
        predict_api = join(api_base, 'predict')
        def_api = join(api_base, 'retrieve_def')
        inputs = {'claim': claim, 'intro': intro, 'term': term}

        definition = requests.post(def_api, data=json.dumps(inputs)).json()
        st.text(f'Examples of definition of "{term}""')
        st.write(definition['definition_list'])

        with st.spinner("Generating definition..."):
            prediction = requests.post(predict_api,
                                       data=json.dumps(inputs)).json()
        st.success('Definition generation completed')
        st.write(prediction['result'])
