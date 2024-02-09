#TO RUN YOU NEED TO RUN THE FOLLOWING IN THE TERMINAL
#pip3 or pip depended on what you have
#pip3 install streamlit, yfinance, prophet, plotly
#and then to get the web app you need to type streamlit run (location of python file, ex. /User/shivpatel/Desktop/stock_predictor.py)

import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Predictor Web App')

stock_symbol = st.text_input("Enter stock symbol (e.g., GOOG, AAPL, MSFT, etc.)", "")

if stock_symbol:  # Check if a stock symbol is entered
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(stock_symbol)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    # Show the plot with modified colors
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    # Set the color of the actual (ground truth) line to blue
    fig1.data[0].marker.color = 'red'
    # Set the color of the predicted line to red
    fig1.data[1].marker.color = 'red'
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
else:
    st.write("Enter a stock symbol to see the forecast.")
