import streamlit as st
import requests

# Exécuter --->  python -m streamlit run Python4DS/streamlit.py


st.title("Téléchargement des dernières parties Lichess")

# Choix
username = st.text_input("Pseudonyme", value="EricRosen")
token = st.text_input("Token", value="lip_IolWV17XjlvaNCEvqeXd", type="password")
output_file = st.text_input("Nom du fichier de sortie", value="games.pgn")



# analyse_personnelle.py
def download_lichess_pgn(username, token, output_file):
    """
    Télécharge toutes les parties d'un utilisateur au format PGN et les sauvegarde dans un fichier.

    Args:
        username (str): Nom d'utilisateur Lichess.
        token (str): Jeton d'accès Lichess.
        output_file (str): Chemin où sauvegarder le fichier PGN.

    """
    url = f'https://lichess.org/api/games/user/{username}'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/x-chess-pgn'  
    }
    params = {
        'max': 3000,  
        'clocks': 1,        
        'evals': 1,
        'Opening':1,
        'pgnInJson': False, 
    }
    response = requests.get(url, headers=headers, params=params, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Erreur lors du téléchargement : {response.status_code}")
    

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    st.success(f"Fichier PGN téléchargé et sauvegardé sous {output_file}")



if st.button("Télécharger"):
    if username != "" and token != "" and output_file != "":
        try:
            download_lichess_pgn(username, token, output_file)
        except Exception as error_text:
            st.error(f"Erreur : {error_text}")
    else:
        st.warning("Veuillez remplir tous les champs avant de continuer.")
