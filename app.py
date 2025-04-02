import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Titre de l'application
st.title("📈 TAO Predictor - Prédiction à 7 et 30 jours")

# API CoinGecko pour récupérer l'historique des prix
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bittensor/market_chart"

def get_tao_history(days=365):  # CoinGecko limite à 365 jours pour les comptes gratuits
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bittensor/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    
    try:
        response = requests.get(COINGECKO_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])
            if not prices:
                st.error("⚠️ Aucune donnée reçue. Réessaie plus tard.")
                return None
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        else:
            st.error(f"❌ Erreur {response.status_code} : {response.json()}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"⏳ Erreur de connexion : {e}")
        return None

# Fonction pour calculer le RSI (Relative Strength Index)
def compute_rsi(data, window=14):
    delta = data["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# Normalisation des prix et préparation des données pour le modèle
def prepare_data(df, window_size=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_price"] = scaler.fit_transform(df["price"].values.reshape(-1, 1))
    
    # Ajout du RSI
    df["rsi"] = compute_rsi(df)
    df["rsi_scaled"] = scaler.fit_transform(df["rsi"].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[["scaled_price", "rsi_scaled"]].iloc[i : i + window_size].values)
        y.append(df["scaled_price"].iloc[i + window_size])

    return np.array(X), np.array(y), scaler

# Sauvegarde et chargement du modèle
MODEL_PATH = "lstm_tao_model.pkl"

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Création et entraînement du modèle LSTM optimisé
def train_lstm(X, y, epochs=50, batch_size=32):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    save_model(model)
    return model

# Fonction pour prédire les prix futurs
def predict_future_prices(model, df, scaler, days=30):
    last_sequence = df[["scaled_price", "rsi_scaled"]].iloc[-7:].values.reshape(1, 7, 2)
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(prediction)[0][0]
        future_prices.append(future_price)

        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices

# Bouton pour entraîner le modèle
epochs = st.slider("Nombre d'epochs", min_value=10, max_value=100, value=50, step=10)
batch_size = st.slider("Taille du batch", min_value=8, max_value=64, value=32, step=8)

if st.button("🚀 Entraîner le modèle LSTM"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        
        model = load_model()
        if model is None:
            model = train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y, epochs, batch_size)
        
        mse = mean_squared_error(scaler.inverse_transform(y.reshape(-1, 1)), scaler.inverse_transform(model.predict(X.reshape(-1, X.shape[1], X.shape[2]))))
        st.metric(label="📊 Performance du Modèle (MSE)", value=f"{mse:.4f}")
        st.write("✅ Modèle entraîné avec succès !")
    else:
        st.error("Erreur : Impossible d'entraîner le modèle.")

# Affichage des prévisions
def plot_predictions(df, future_prices, days):
    future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=days + 1, freq="D")[1:]
    
    lower_bound = np.array(future_prices) * 0.95
    upper_bound = np.array(future_prices) * 1.05

    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
    plt.plot(future_dates, future_prices, label=f"Prédictions {days} jours", linestyle="dashed", color="red")
    plt.fill_between(future_dates, lower_bound, upper_bound, color='gray', alpha=0.2, label="Intervalle de confiance")
    
    plt.xlabel("Date")
    plt.ylabel("Prix en USD")
    plt.title(f"📈 Prédiction du prix TAO sur {days} jours")
    plt.legend()
    st.pyplot(plt)

if st.button("📊 Afficher les prévisions sur 7 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = load_model() or train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y)
        future_prices = predict_future_prices(model, df, scaler, days=7)
        plot_predictions(df, future_prices, 7)

if st.button("📊 Afficher les prévisions sur 30 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = load_model() or train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y)
        future_prices = predict_future_prices(model, df, scaler, days=30)
        plot_predictions(df, future_prices, 30)
