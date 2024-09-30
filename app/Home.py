import sys
import os

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.path.abspath('scripts'))
from portOptimizer import PortOpt

st.set_page_config(layout="wide")


st.title("Portfolio Optimizer")

with st.expander('How to Use the App'):
    st.markdown('''
    This app helps you optimize your investment portfolio for a given set of stocks.

    **Instructions:**
    1. Start by entering the stock tickers (symbols) for publicly traded companies.
       - Example: `AAPL MSFT NVDA BALL BAH BAK BANC BANF BANFP BANR BANX`.  
       *(Tip: You can copy and paste the tickers into the input field below.)*
    2. After entering the tickers, adjust the amount of money you want to allocate and click on `Optimize` to generate the optimal portfolio.
    ''')

st.subheader("Enter your prefered symbols space separated")

symbols = st.text_input("Enter you symbols")
money = st.text_input("Enter the amount of money you want to optimize", value="10000")
short = st.checkbox("Allow shorts")

if st.button('Optimize'):

    if symbols:
        if short:
            allow_short = True
        else: 
            allow_short = False
        money = float(money)
        stocks = yf.download(tickers=symbols, period="max")['Adj Close']
        portfolio_optimizer = PortOpt()
        allocation, leftover = portfolio_optimizer.budget_allocator(stocks,money ,allow_short)

        allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Amount'])
        allocation['Color'] = np.where(allocation["Amount"] <0, 'red', 'green')
        allocation['Ratio'] = allocation['Amount'] / allocation['Amount'].abs().sum()

        st.subheader("Ratios for your portfolio")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name='Net',
                x=allocation.index,
                y=allocation['Ratio'],
                marker_color=allocation['Color']))


        st.plotly_chart(fig)

        st.subheader("Ratios with your custom budget: Number of shares you should buy")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name='Net',
                    x=allocation.index,
                    y=allocation['Amount'],
                    marker_color=allocation['Color'])
        )
        st.plotly_chart(fig)


        portfolio_optimizer.plot_efficient_frontier_custom( stocks, allow_short)



    else:
        st.write("Please enter two or more symbols/tickers separated by space.")