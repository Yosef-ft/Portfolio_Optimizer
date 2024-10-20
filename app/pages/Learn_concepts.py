import streamlit as st
import time

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.header('Portfolio Optimization concepts')
my_bar = st.progress(0)

for percent_complete in range(33):
    time.sleep(0.09)
    my_bar.progress(percent_complete + 1)


st.subheader('Comming soon')
st.write('Site under construction')
