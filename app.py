import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 📌 API CoinGecko pour récupérer l'historique des prix et volumes
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
        
        # Ajout du volume de trading
        df["volume"] = [v[1] for v in volumes]

        return df
    else:
        st.error(f"Erreur {response.status_code} : Impossible de récupérer les données.")
        return None

# 📈 Calcul des indicateurs techniques
def calculate_indicators(df):
    df["SMA_14"] = df["price"].rolling(window=14).mean()  # Moyenne Mobile 14 jours

    # MACD
    short_ema = df["price"].ewm(span=12, adjust=False).mean()
    long_ema = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bandes de Bollinger
    df["BB_Mid"] = df["price"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Mid"] + (df["price"].rolling(window=20).std() * 2)
    df["BB_Lower"] = df["BB_Mid"] - (df["price"].rolling(window=20).std() * 2)

    return df

# 📌 Interface Streamlit
st.title("📊 TAO Predictor - Analyse Technique")

# Bouton pour charger les données et afficher les indicateurs
if st.button("📊 Charger les données et afficher les indicateurs"):
    df = get_tao_history()
    if df is not None:
        df = calculate_indicators(df)

        # Affichage des indicateurs sur le graphique
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Courbe des prix
        ax1.plot(df["timestamp"], df["price"], label="Prix TAO", color="blue")
        ax1.plot(df["timestamp"], df["BB_Upper"], linestyle="dashed", color="gray", label="Bollinger Upper")
        ax1.plot(df["timestamp"], df["BB_Lower"], linestyle="dashed", color="gray", label="Bollinger Lower")
        ax1.set_ylabel("Prix en USD")
        ax1.legend(loc="upper left")

        # 📈 Ajout du MACD
        ax2 = ax1.twinx()
        ax2.plot(df["timestamp"], df["MACD"], label="MACD", color="red")
        ax2.plot(df["timestamp"], df["MACD_Signal"], label="MACD Signal", color="green")
        ax2.set_ylabel("MACD")
        ax2.legend(loc="upper right")

        plt.title("📊 Evolution du prix avec MACD & Bandes de Bollinger")
        st.pyplot(fig)

        # Affichage du RSI
        fig, ax3 = plt.subplots(figsize=(12, 3))
        ax3.plot(df["timestamp"], df["RSI"], label="RSI", color="purple")
        ax3.axhline(70, linestyle="dashed", color="red", label="Seuil surachat")
        ax3.axhline(30, linestyle="dashed", color="green", label="Seuil survente")
        ax3.set_ylabel("RSI")
        ax3.legend()
        plt.title("📈 RSI Indicator")
        st.pyplot(fig)
        
        st.write("✅ **Indicateurs calculés et affichés avec succès !**")
    else:
        st.error("Erreur : Impossible de récupérer les données.")
