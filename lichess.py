import requests

import json


def fetch_games(username, max_games=100):
    # URL de l'API Lichess pour récupérer les parties
    url = f"https://lichess.org/api/games/user/{username}"
    headers = {
        'Accept': 'application/x-ndjson',
    }

    params = {
        'max': max_games,
        'pgnInJson': True,  # Pour recevoir les PGN annotés
        'withTimestamps': True,  # Pour obtenir les temps par coup
        'clocks': 1,
        'evals': 1,  # Inclure les évaluations après chaque coup
        'opening': 1,
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return response.text

def save_games_to_pgn(games_data, filename):
    with open(filename, 'w') as f:
        f.write(games_data)
'''
def main():
    username = input("Entrez le nom d'utilisateur de Lichess: ")  #EricRosen
    max_games = int(input("Entrez le nombre maximum de parties à récupérer: "))

    print(f"Récupération des parties pour l'utilisateur {username}...")
    games_data = fetch_games(username, max_games)

    # Enregistrement des parties dans un fichier PGN
    filename = f"{username}_games.pgn"
    save_games_to_pgn(games_data, filename)

    print(f"Les parties ont été sauvegardées dans le fichier {filename}")

if __name__ == "__main__":
    main()
'''


def tab_info_games(username, max_games):          #renvoi une tableau tab tel a que tab[i] est un tableau contenant toutesles infos intéressantes de la partie numéro i (dans l'ordre): le type de la partie (souvent blitz ou bullet), le nom d'utilisateur jouant les blancs, le nom d'utilisateur jouant les noirs, le vainqueur, la manière de gagner, elo blanc, elo noir, ouverture, les coups, les evaluations avant chaque coups, les temps  
  games_data=fetch_games(username, max_games)
  print(games_data)
  tab = []

  for line in games_data.splitlines():
        game = json.loads(line)  # Convertir chaque ligne JSON en dictionnaire

        game_info = {
            'type': game.get('speed', 'Unknown'),
            'white_player': game['players']['white']['user']['name'] if 'user' in game['players']['white'] else 'Anonymous',
            'black_player': game['players']['black']['user']['name'] if 'user' in game['players']['black'] else 'Anonymous',
            'winner': game.get('winner', 'draw'),
            'termination': game.get('status', 'Unknown'),
            'white_elo': game['players']['white'].get('rating', 'Unknown'),
            'black_elo': game['players']['black'].get('rating', 'Unknown'),
            'opening': game['opening'].get('name', 'Unknown') if 'opening' in game else 'Unknown',
            'moves': game.get('moves', '').split(),
            'evaluations': [analysis.get('eval', 'N/A') for analysis in game.get('analysis', [])],
            'clocks': game.get('clocks', [])
        }
        
        tab.append(game_info)
    
  return tab

tab_info_games("EricRosen",5)
