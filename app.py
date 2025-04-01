import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# API CoinGecko pour récupérer l'historique des prix
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bittensor/market_chart"

def get_tao_history(days=365):
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(COINGECKO_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        st.error(f"Erreur {response.status_code} : Impossible de récupérer les données.")
        return None

# Préparation des données pour le LSTM
def prepare_data(df, window_size=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_price"] = scaler.fit_transform(df["price"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df["scaled_price"].iloc[i : i + window_size].values)
        y.append(df["scaled_price"].iloc[i + window_size])

    return np.array(X), np.array(y), scaler

# Entraînement du modèle LSTM
def train_lstm(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model

# Prédictions des prix futurs
def predict_future_prices(model, df, scaler, days=7):
    last_sequence = df["scaled_price"].iloc[-7:].values.reshape(1, 7, 1)
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(prediction)[0][0]
        future_prices.append(future_price)

        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices

# Calcul des indicateurs MACD et RSI
def compute_macd_rsi(df):
    df["EMA12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

# Interface Streamlit
st.title("📈 TAO Predictor - Prédictions sur 7 et 30 jours")

df = get_tao_history()
if df is not None:
    df = compute_macd_rsi(df)

    # Sélection des indicateurs
    show_macd = st.checkbox("Afficher MACD")
    show_rsi = st.checkbox("Afficher RSI")

    # Bouton pour afficher les prévisions sur 7 jours
    if st.button("📊 Afficher les prévisions sur 7 jours"):
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler, days=7)

        # Graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=8, freq="D")[1:]
        plt.plot(future_dates, future_prices, label="Prédictions 7 jours", linestyle="dashed", color="red")

        if show_macd:
            plt.plot(df["timestamp"], df["MACD"], label="MACD", linestyle="dotted", color="purple")
            plt.plot(df["timestamp"], df["Signal"], label="Signal MACD", linestyle="dotted", color="orange")

        if show_rsi:
            plt.twinx()
            plt.plot(df["timestamp"], df["RSI"], label="RSI", linestyle="dashed", color="green")
            plt.ylim(0, 100)

        plt.xlabel("Date")
        plt.ylabel("Prix en USD")
        plt.title("📈 Prédiction du prix TAO sur 7 jours")
        plt.legend()
        st.pyplot(plt)

    # Bouton pour afficher les prévisions sur 30 jours
    if st.button("📊 Afficher les prévisions sur 30 jours"):
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler, days=30)

        # Graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=31, freq="D")[1:]
        plt.plot(future_dates, future_prices, label="Prédictions 30 jours", linestyle="dashed", color="green")

        if show_macd:
            plt.plot(df["timestamp"], df["MACD"], label="MACD", linestyle="dotted", color="purple")
            plt.plot(df["timestamp"], df["Signal"], label="Signal MACD", linestyle="dotted", color="orange")

        if show_rsi:
            plt.twinx()
            plt.plot(df["timestamp"], df["RSI"], label="RSI", linestyle="dashed", color="green")
            plt.ylim(0, 100)

        plt.xlabel("Date")
        plt.ylabel("Prix en USD")
        plt.title("📈 Prédiction du prix TAO sur 30 jours")
        plt.legend()
        st.pyplot(plt)
