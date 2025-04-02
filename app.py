import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # Librairie pour les indicateurs techniques
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# 📌 API CoinGecko pour récupérer l'historique des prix
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bittensor/market_chart"

def get_tao_history(days=365):
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    try:
        response = requests.get(COINGECKO_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        else:
            st.error(f"❌ Erreur {response.status_code} : {response.json()}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"⏳ Erreur de connexion : {e}")
        return None

# 📌 Ajout des indicateurs techniques
def add_technical_indicators(df):
    df["SMA_10"] = ta.trend.sma_indicator(df["price"], window=10)
    df["EMA_10"] = ta.trend.ema_indicator(df["price"], window=10)
    df["RSI"] = ta.momentum.rsi(df["price"], window=14)
    df["MACD"] = ta.trend.macd(df["price"])
    df["Bollinger_High"] = ta.volatility.bollinger_hband(df["price"])
    df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["price"])
    
    # Remplacer les valeurs NaN par la moyenne des colonnes
    df.fillna(df.mean(), inplace=True)
    
    return df

# 📌 Normalisation et préparation des données
def prepare_data(df, window_size=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = ["price", "SMA_10", "EMA_10", "RSI", "MACD", "Bollinger_High", "Bollinger_Low"]
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
    
    X, y = [], []
    for i in range(len(df_scaled) - window_size):
        X.append(df_scaled.iloc[i : i + window_size].values)
        y.append(df_scaled.iloc[i + window_size]["price"])

    return np.array(X), np.array(y), scaler

# 📌 Création et entraînement du modèle LSTM
def train_lstm(X, y):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    # 🔥 Entraînement du modèle
    model.fit(X, y, epochs=25, batch_size=16, verbose=1)
    return model

# 📌 Prédiction des prix futurs
def predict_future_prices(model, df, scaler, days=30):
    feature_cols = ["price", "SMA_10", "EMA_10", "RSI", "MACD", "Bollinger_High", "Bollinger_Low"]
    last_sequence = df[feature_cols].iloc[-7:].values.reshape(1, 7, len(feature_cols))  
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(np.append(prediction, np.zeros(len(feature_cols) - 1)).reshape(1, -1))[0][0]
        future_prices.append(future_price)

        # Mise à jour de la séquence
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices

# 📌 Interface Streamlit
st.title("📈 TAO Predictor - Prédiction avec Indicateurs Techniques")

if st.button("🚀 Entraîner le modèle LSTM"):
    df = get_tao_history()
    if df is not None:
        df = add_technical_indicators(df)
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y)

        st.success("✅ Modèle entraîné avec succès !")

# 📌 Prédiction et affichage
if st.button("📊 Afficher les prévisions sur 30 jours"):
    df = get_tao_history()
    if df is not None:
        df = add_technical_indicators(df)
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y)
        future_prices = predict_future_prices(model, df, scaler, days=30)

        # Graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=31, freq="D")[1:]
        plt.plot(future_dates, future_prices, label="Prédictions 30 jours", linestyle="dashed", color="green")

        plt.xlabel("Date")
        plt.ylabel("Prix en USD")
        plt.title("📈 Prédiction TAO avec indicateurs techniques")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Erreur : Impossible d'afficher les prévisions.")
