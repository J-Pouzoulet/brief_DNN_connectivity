import streamlit as st
import pandas as pd
import numpy as np
import requests as rq
import json

st.title('Wanna find out what is resnet50 predicting???')    
url = st.text_input("Enter an image's url here and push the predict button!", 'https://static.independent.co.uk/2022/02/09/22/Orangutan_Baby-New_Orleans_61180.jpg')
st.image(url)

if st.button('Predict'):
    #call fast_api when opened in local uvicorn server 
    res = rq.get(f'http://127.0.0.1:8000/?url={url}').json()
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    item = res['item']
    prob = float(res['prob'])*100
    st.title(f'Resnet50 is predicting a {item} at {round(prob, 2)}%')