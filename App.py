import streamlit as st

# Titre de l'application
st.title("📈 TAO Predictor")
st.write("Bienvenue sur l'application de prédiction du prix de TAO.")

# Ajout d'un champ de texte
user_input = st.text_input("Entrez un symbole de crypto (ex: TAO) :", "TAO")

# Bouton de soumission
if st.button("Prédire le prix"):
    st.write(f"Prédiction en cours pour {user_input}...")
