import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import timedelta

st.title("📈 Stock Price Prediction using LSTM")

# Input UI
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, INFY.NS)", value="AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.button("Predict"):
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for this ticker or date range.")
    else:
        # Extract Close prices
        df = pd.DataFrame(data["Close"])
        df.columns = ["Close"]
        st.subheader("Raw Closing Prices")
        st.line_chart(df)

        # Normalize and prepare training data
        dataset = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        prediction_days = 60
        x_train, y_train = [], []

        for i in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i - prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

        # Prepare input for prediction
        last_60_days = df[-prediction_days:].values
        last_60_scaled = scaler.transform(last_60_days)
        X_test = [last_60_scaled]
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)

        # Show predicted price for next day
        next_day = end_date + timedelta(days=1)
        st.subheader("📊 Prediction Result")
        st.write(f"**Predicted Closing Price for {ticker} on {next_day}:** ₹{predicted_price[0][0]:.2f}")

        # Plot last known and predicted point
        fig, ax = plt.subplots()
        actual = df[-30:]["Close"].tolist()
        predicted = [np.nan]*29 + [predicted_price[0][0]]
        ax.plot(actual + [np.nan], label="Actual (last 30 days)", color='blue')
        ax.plot(predicted, label="Predicted (next day)", color='red', linestyle='--', marker='o')
        ax.set_title(f"{ticker} Price Prediction for Next Day")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        st.success("✅ Prediction complete.")
