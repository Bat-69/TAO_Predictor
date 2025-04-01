import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 📌 API CoinGecko pour récupérer les prix
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bittensor/market_chart"

def get_tao_history(days=365):
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(COINGECKO_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        prices = data["prices"]
        volumes = data["total_volumes"]
        
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = [v[1] for v in volumes]

        return df
    else:
        st.error(f"Erreur {response.status_code} : Impossible de récupérer les données.")
        return None

# 📈 Calcul des indicateurs techniques
def calculate_indicators(df):
    df["SMA_14"] = df["price"].rolling(window=14).mean()
    short_ema = df["price"].ewm(span=12, adjust=False).mean()
    long_ema = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BB_Mid"] = df["price"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Mid"] + (df["price"].rolling(window=20).std() * 2)
    df["BB_Lower"] = df["BB_Mid"] - (df["price"].rolling(window=20).std() * 2)

    return df.dropna()

# 🏗 Préparation des données pour le modèle LSTM
def prepare_data(df, window_size=14):
    df = calculate_indicators(df)
    features = ["price", "volume", "MACD", "RSI", "BB_Upper", "BB_Lower"]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i : i + window_size])
        y.append(scaled_data[i + window_size, 0])  # On prédit le prix futur
    
    return np.array(X), np.array(y), scaler

# 🎯 Création et entraînement du modèle LSTM
def train_lstm(X, y):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=25, batch_size=16, verbose=1)
    return model

# 🔮 Prédiction des prix futurs
def predict_future_prices(model, df, scaler, days=7):
    df = calculate_indicators(df)
    features = ["price", "volume", "MACD", "RSI", "BB_Upper", "BB_Lower"]
    
    scaled_data = scaler.transform(df[features])
    last_sequence = scaled_data[-14:].reshape(1, 14, len(features))
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(
            np.hstack([prediction, np.zeros((1, len(features) - 1))])
        )[0][0]
        future_prices.append(future_price)

        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices

# 🏠 Interface utilisateur Streamlit
st.title("📈 TAO Predictor - Optimisation du modèle")

if st.button("🚀 Entraîner le modèle et afficher les prédictions"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X, y)
        future_prices = predict_future_prices(model, df, scaler)

        # 📊 Graphique des prédictions
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=len(future_prices), freq="D")
        ax.plot(future_dates, future_prices, label="Prédictions", linestyle="dashed", color="red")

        ax.set_xlabel("Date")
        ax.set_ylabel("Prix en USD")
        ax.set_title("📈 Prédiction du prix TAO sur 7 jours")
        ax.legend()
        st.pyplot(fig)
        
        st.write("✅ **Modèle entraîné avec succès et prévisions affichées !**")
    else:
        st.error("Erreur : Impossible de récupérer les données.")
