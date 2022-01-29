import streamlit as st
from nsetools import Nse


def app():

    st.title("Stocks Listed on NSE")
    nse = Nse()

    st.sidebar.selectbox(
        "Select Page You want to visit",
        ("Stocks List", "Prediction Info" )
    )
    gainers = nse.get_top_gainers()
    losers = nse.get_top_losers()
    # print(gainers[0])
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.subheader("Top Gainers")
    kpi3.subheader("Top Losers")
    for i in range(10):
        lowerChange = ((losers[i]['lowPrice'] - losers[i]['highPrice'])/losers[i]['lowPrice'])*100
        upperChange = ((gainers[i]['highPrice'] - gainers[i]['lowPrice'])/gainers[i]['lowPrice'])*100
        lowerChange = "{:.2f}".format(lowerChange)
        upperChange = "{:.2f}".format(upperChange)
        kpi1.metric(label=gainers[i]['symbol'], value=gainers[i]['openPrice'], delta = upperChange)
        kpi3.metric(label=losers[i]['symbol'], value=losers[i]['openPrice'], delta = lowerChange)
