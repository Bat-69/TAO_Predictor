import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Titre de l'application
st.title("📈 TAO Predictor - Prédiction à 7 jours")

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

# Bouton pour récupérer les données
if st.button("📊 Charger l'historique des prix"):
    df = get_tao_history()
    if df is not None:
        st.write("✅ Données chargées avec succès !")
        st.line_chart(df.set_index("timestamp")["price"])
    else:
        st.error("Erreur : Impossible de récupérer les données.")
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Normalisation des prix
def prepare_data(df, window_size=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_price"] = scaler.fit_transform(df["price"].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df["scaled_price"].iloc[i : i + window_size].values)
        y.append(df["scaled_price"].iloc[i + window_size])

    return np.array(X), np.array(y), scaler

# Bouton pour préparer les données
if st.button("🔄 Préparer les données"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        st.write(f"✅ Données préparées avec {X.shape[0]} échantillons.")
    else:
        st.error("Erreur : Impossible de préparer les données.")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fonction pour créer et entraîner le modèle LSTM
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

# Bouton pour entraîner le modèle
if st.button("🚀 Entraîner le modèle LSTM"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        st.write("✅ Modèle entraîné avec succès !")
    else:
        st.error("Erreur : Impossible d'entraîner le modèle.")
# Fonction pour prédire le prix de TAO dans 7 jours
def predict_future_prices(model, df, scaler, days=30):
    last_sequence = df["scaled_price"].iloc[-7:].values.reshape(1, 7, 1)  # Prendre les 7 derniers jours
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(prediction)[0][0]
        future_prices.append(future_price)

        # Mettre à jour la séquence pour la prochaine prédiction
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices
# Bouton pour prédire le prix futur
if st.button("🔮 Prédire le prix dans 7 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_price = predict_future_price(model, df, scaler)
        st.write(f"📈 **Prix prédit dans 7 jours : {future_price:.2f} USD**")
    else:
        st.error("Erreur : Impossible de prédire le prix.")
import matplotlib.pyplot as plt

# Bouton pour prédire et afficher le graphique
if st.button("📊 Afficher les prévisions sur 7 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], 1), y)
        future_prices = predict_future_prices(model, df, scaler)

        # Création du graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=7, freq="D")
        plt.plot(future_dates, future_prices, label="Prédictions", linestyle="dashed", color="red")

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
