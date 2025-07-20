import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Predictor")

ticker = st.text_input("Enter stock symbol (e.g. INFY.NS, TCS.NS)", "INFY.NS")

if st.button("Analyze"):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = stock.history(period="1y")

        # Plot chart
        st.subheader(f"{info['shortName']} - Closing Price (1 Year)")
        plt.figure(figsize=(10, 4))
        plt.plot(data['Close'])
        plt.title("Closing Price")
        st.pyplot(plt)

        # Show key metrics
        st.subheader("📊 Key Stats")
        st.write(f"**52 Week High:** ₹{info['fiftyTwoWeekHigh']}")
        st.write(f"**52 Week Low:** ₹{info['fiftyTwoWeekLow']}")
        st.write(f"**PE Ratio:** {info.get('trailingPE', 'N/A')}")
        st.write(f"**Market Cap:** ₹{info.get('marketCap', 'N/A')}")

        # Predict next price
        st.subheader("🤖 Price Prediction (Next Day)")
        df = data[['Close']].dropna()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        X = np.array(df['Close']).reshape(-1, 1)
        y = df['Target']

        model = LinearRegression()
        model.fit(X, y)
        next_price = model.predict([[df['Close'].iloc[-1]]])[0]
        st.success(f"Predicted Next Close Price: ₹{round(next_price, 2)}")

        # Benchmark comparison
        st.subheader("📉 Compare with NIFTY 50")
        nifty = yf.Ticker("^NSEI").history(period="1y")
        plt.figure(figsize=(10, 4))
        plt.plot(data['Close'], label=ticker)
        plt.plot(nifty['Close'], label='NIFTY 50')
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error: {str(e)}")
