import streamlit as st
import time

st.header('Portfolio Optimization concepts')
my_bar = st.progress(0)

for percent_complete in range(33):
    time.sleep(0.09)
    my_bar.progress(percent_complete + 1)


st.subheader('Comming soon')
st.write('Site under construction')