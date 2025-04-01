import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Titre de l'application
st.title("📈 TAO Predictor - Prédiction à 7 et 30 jours")

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

# Normalisation des prix et préparation des données pour le modèle
def prepare_data(df, window_size=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_price"] = scaler.fit_transform(df["price"].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df["scaled_price"].iloc[i : i + window_size].values)
        y.append(df["scaled_price"].iloc[i + window_size])

    return np.array(X), np.array(y), scaler

# Création et entraînement du modèle LSTM
def train_lstm(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    # Entraînement du modèle
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)
    return model

# Fonction pour évaluer le modèle avec le MSE
def evaluate_model(model, X, y, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(y_actual, predictions)
    return mse

# Fonction pour prédire les prix futurs
def predict_future_prices(model, df, scaler, days=30):
    last_sequence = df["scaled_price"].iloc[-7:].values.reshape(1, 7, 1)  
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(prediction)[0][0]
        future_prices.append(future_price)

        # Mise à jour de la séquence
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices

# Bouton pour entraîner le modèle
if st.button("🚀 Entraîner le modèle LSTM"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)

        # 🆕 Calcul et affichage de la performance
        mse = evaluate_model(model, X.reshape(-1, X.shape[1], 1), y, scaler)
        st.write(f"📊 **Performance du modèle (MSE) : {mse:.4f}**")

        st.write("✅ Modèle entraîné avec succès !")
    else:
        st.error("Erreur : Impossible d'entraîner le modèle.")

# Bouton pour afficher les prévisions sur 7 jours
if st.button("📊 Afficher les prévisions sur 7 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler, days=7)

        # Graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=8, freq="D")[1:]
        plt.plot(future_dates, future_prices, label="Prédictions 7 jours", linestyle="dashed", color="red")

        plt.xlabel("Date")
        plt.ylabel("Prix en USD")
        plt.title("📈 Prédiction du prix TAO sur 7 jours")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Erreur : Impossible d'afficher les prévisions.")

# Bouton pour afficher les prévisions sur 30 jours
if st.button("📊 Afficher les prévisions sur 30 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler, days=30)

        # Graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=31, freq="D")[1:]
        plt.plot(future_dates, future_prices, label="Prédictions 30 jours", linestyle="dashed", color="green")

        plt.xlabel("Date")
        plt.ylabel("Prix en USD")
        plt.title("📈 Prédiction du prix TAO sur 30 jours")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Erreur : Impossible d'afficher les prévisions.")
