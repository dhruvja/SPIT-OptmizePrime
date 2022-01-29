import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np

START = "2015-01-01"
END = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("TATAELXSI", "TATACONSUM", "NTPC", "SUNPHARMA")

selected_stocks = st.selectbox("Select data for prediction", stocks)
n_years = st.slider("Years of prediction", 1,4)
period = n_years*365

@st.cache
def load_data(ticker):
	data = yf.download(ticker,START, END)
	data.reset_index(inplace=True)
	return data

data = load_data(selected_stocks)

st.subheader("Raw Data")
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Close'], name="stock_close"))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()


