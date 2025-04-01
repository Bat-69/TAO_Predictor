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
