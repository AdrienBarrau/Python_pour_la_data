import requests
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
    
    print(f"Fichier PGN téléchargé et sauvegardé sous {output_file}")


username = 'EricRosen'
token = 'lip_IolWV17XjlvaNCEvqeXd'
output_file = 'games.pgn' 
download_lichess_pgn(username, token, output_file)
