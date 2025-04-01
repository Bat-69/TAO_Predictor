import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 📌 Initialisation des variables d'état (important pour éviter que le graphique disparaisse)
if "show_macd" not in st.session_state:
    st.session_state.show_macd = False
if "show_rsi" not in st.session_state:
    st.session_state.show_rsi = False

# ✅ Fonction pour récupérer l'historique des prix TAO
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

# ✅ Fonctions pour calculer MACD et RSI
def calculate_macd(df):
    df["EMA_12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# ✅ Préparation des données pour le modèle LSTM
def prepare_data(df, window_size=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_price"] = scaler.fit_transform(df["price"].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df["scaled_price"].iloc[i : i + window_size].values)
        y.append(df["scaled_price"].iloc[i + window_size])

    return np.array(X), np.array(y), scaler

# ✅ Création et entraînement du modèle LSTM
def train_lstm(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    return model

# ✅ Prédiction des prix futurs
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

# 📌 Boutons pour afficher les prévisions
if st.button("📊 Afficher les prévisions sur 7 jours"):
    df = get_tao_history()
    if df is not None:
        df = calculate_macd(df)
        df = calculate_rsi(df)

        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler, days=7)

        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=8, freq="D")[1:]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        ax1.plot(future_dates, future_prices, label="Prédictions 7 jours", linestyle="dashed", color="red")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Prix en USD")
        ax1.set_title("📈 Prédiction du prix TAO sur 7 jours")
        ax1.legend()

        # 📌 Ajout de cases à cocher (avec gestion d'état)
        if st.checkbox("📈 Afficher MACD (7 jours)", value=st.session_state.show_macd):
            st.session_state.show_macd = not st.session_state.show_macd
        if st.checkbox("📊 Afficher RSI (7 jours)", value=st.session_state.show_rsi):
            st.session_state.show_rsi = not st.session_state.show_rsi

        # ✅ Affichage du MACD sur le même graphique
        if st.session_state.show_macd:
            ax2 = ax1.twinx()
            ax2.plot(df["timestamp"], df["MACD"], label="MACD", color="purple", alpha=0.6)
            ax2.plot(df["timestamp"], df["Signal_Line"], label="Signal Line", color="orange", alpha=0.6)
            ax2.set_ylabel("MACD")
            ax2.legend(loc="upper left")

        st.pyplot(fig)  # Affichage du graphique principal

        # ✅ RSI sur un graphique séparé
        if st.session_state.show_rsi:
            fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
            ax_rsi.plot(df["timestamp"], df["RSI"], label="RSI", color="green")
            ax_rsi.axhline(70, linestyle="--", color="red")
            ax_rsi.axhline(30, linestyle="--", color="blue")
            ax_rsi.set_title("RSI - Indice de Force Relative (7 jours)")
            ax_rsi.legend()
            st.pyplot(fig_rsi)  # Affichage du graphique RSI
    else:
        st.error("Erreur : Impossible d'afficher les prévisions.")
