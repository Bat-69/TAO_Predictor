import streamlit as st
import requests

# Titre de l'application
st.title("📈 TAO Predictor")
st.write("Bienvenue sur l'application de prédiction du prix de TAO.")

# Clé API CoinMarketCap (remplace par la tienne)
API_KEY = "2f778dc5-205e-4a98-9b05-ad192fbc5e46"
URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

# Récupérer le prix actuel de TAO
def get_tao_price():
    headers = {"X-CMC_PRO_API_KEY": API_KEY}
    params = {"symbol": "TAO", "convert": "USD"}
    response = requests.get(URL, headers=headers, params=params)
    data = response.json()
    
    try:
        price = data["data"]["TAO"]["quote"]["USD"]["price"]
        return round(price, 2)
    except KeyError:
        return "Erreur : Données introuvables"

# Affichage du prix en direct
if st.button("Obtenir le prix TAO"):
    price = get_tao_price()
    st.write(f"💰 Prix actuel de TAO : **${price}**")
