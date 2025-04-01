import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 📌 API CoinGecko pour récupérer les données historiques du prix de TAO
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

# 📏 Fonction de normalisation et préparation des données pour le LSTM
def prepare_data(df, window_size=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_price"] = scaler.fit_transform(df["price"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df["scaled_price"].iloc[i : i + window_size].values)
        y.append(df["scaled_price"].iloc[i + window_size])

    return np.array(X), np.array(y), scaler

# 📏 Fonction pour évaluer la performance du modèle
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)

    # Convertir les prédictions en valeurs réelles
    predictions_real = scaler.inverse_transform(
        np.hstack([predictions, np.zeros((len(predictions), 1))])
    )[:, 0]

    y_test_real = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 1))])
    )[:, 0]

    mse = mean_squared_error(y_test_real, predictions_real)
    return mse

# 📌 Fonction pour créer et entraîner le modèle LSTM
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

# 📌 Fonction de prédiction sur X jours
def predict_future_prices(model, df, scaler, days=7):
    last_sequence = df["scaled_price"].iloc[-7:].values.reshape(1, 7, 1)
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(prediction)[0][0]
        future_prices.append(future_price)

        # Mise à jour de la séquence pour la prochaine prédiction
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices

# 🌟 Interface Streamlit 🌟
st.title("📈 TAO Predictor - Prédiction du prix de Bittensor (TAO)")

# 📊 Charger l'historique des prix
if st.button("📊 Charger l'historique des prix"):
    df = get_tao_history()
    if df is not None:
        st.write("✅ Données chargées avec succès !")
        st.line_chart(df.set_index("timestamp")["price"])
    else:
        st.error("Erreur : Impossible de récupérer les données.")

# 🔄 Préparer les données
if st.button("🔄 Préparer les données"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        st.write(f"✅ Données préparées avec {X.shape[0]} échantillons.")
    else:
        st.error("Erreur : Impossible de préparer les données.")

# 🚀 Entraîner et évaluer le modèle
if st.button("🚀 Entraîner et évaluer le modèle LSTM"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)

        # ✅ Évaluation du modèle après entraînement
        mse = evaluate_model(model, X.reshape(-1, X.shape[1], 1), y, scaler)
        st.write(f"📉 **Erreur quadratique moyenne (MSE) : {mse:.4f}**")

        st.write("✅ Modèle entraîné et évalué avec succès !")
    else:
        st.error("Erreur : Impossible d'entraîner le modèle.")

# 📊 Prédire le prix à 7 jours
if st.button("📊 Afficher les prévisions sur 7 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler, days=7)

        # Création du graphique
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

# 📊 Prédire le prix à 30 jours
if st.button("📊 Afficher les prévisions sur 30 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler, days=30)

        # Création du graphique
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
