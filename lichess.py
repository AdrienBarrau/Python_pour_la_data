import requests
import json
from datetime import datetime

def get_lichess_games(username, max_games=100):
    # URL de l'API de Lichess 
    url = f"https://lichess.org/api/games/user/{username}"

  
    params = {
        'max': max_games, 
        'moves': 1,  
        'pgnInJson': 1, 
        'clocks': 1,  
        'evals': 0,  
        'opening': 1,  
        'perfType': 'blitz',  
    }

    headers = {
        'Accept': 'application/x-ndjson'  
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Erreur: Impossible de récupérer les données pour {username}")
        return []

    games = response.text.splitlines()

    games_data = []

    for game in games:
  
        game_json = json.loads(game)

        timestamp = game_json.get('createdAt', 0) // 1000  # Convertir le timestamp en secondes
        game_date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        status = game_json.get('status')
        if status == 'mate':
            termination = 'Checkmate'
        elif status == 'resign':
            termination = 'Resignation'
        elif status == 'stalemate':
            termination = 'Stalemate'
        elif status == 'timeout':
            termination = 'Timeout'
        elif status == 'draw':
            termination = 'Draw'
        elif status == 'outoftime':
            termination = 'Opponent ran out of time'
        else:
            termination = 'Other'

       #params importants
        game_info = {
            'game_id': game_json.get('id'),
            'date': game_date,
            'winner': game_json.get('winner', 'draw'),  # gagnant: "white", "black" ou "draw"
            'opponent': game_json.get('players', {}).get('black' if game_json['players']['white']['user']['name'] == username else 'white', {}).get('user', {}).get('name', 'Unknown'),
            'result': 'won' if game_json.get('winner') == game_json['players']['white']['user']['name'] else 'lost' if game_json.get('winner') else 'draw',
            'moves': game_json.get('moves', '').split(),
            'time_per_move': game_json.get('clocks', []),
            'opening': game_json.get('opening', {}).get('name', 'Unknown'),
            'game_mode': game_json.get('speed', 'Unknown'),  # Mode de jeu : blitz, bullet, etc.
            'termination': termination,  # Raison de la fin de partie
        }

        games_data.append(game_info)

    return games_data


username = "Hikaru1"  
games = get_lichess_games(username, max_games=100)


for i, game in enumerate(games, start=1):
    print(f"Partie {i}:")
    print(f"  ID: {game['game_id']}")
    print(f"  Date: {game['date']}")
    print(f"  Adversaire: {game['opponent']}")
    print(f"  Résultat: {game['result']}")
    print(f"  Raison de la fin: {game['termination']}")
    print(f"  Mode de jeu: {game['game_mode']}")
    print(f"  Ouverture: {game['opening']}")
    print(f"  Coups: {', '.join(game['moves'])}")
    print(f"  Temps par coup: {game['time_per_move']}")
    print("-" * 40)

