import requests
import json
import pandas as pd
import statistics as sta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
class LichessAPI:
    """
    Classe pour interagir avec l'API de Lichess et gérer les parties d'un utilisateur.
    """

    API_URL = "https://lichess.org/api/games/user/"

    @staticmethod
    def fetch_games(username, max_games=200):
        headers = {'Accept': 'application/x-ndjson',}
        params = {'max': max_games,'pgnInJson': True,'withTimestamps': True,'clocks': 1,'evals': 1,'opening': 1,}
        response = requests.get(f"{LichessAPI.API_URL}{username}", headers=headers, params=params)
        response.raise_for_status()
        return response.text

class ChessGame:
    """
    Classe pour représenter une partie d'échecs et ses détails.
    """
    def __init__(self, game_data, username):
        self.type = game_data.get('speed', 'Unknown')
        self.white_player = game_data['players']['white']['user']['name'] if 'user' in game_data['players']['white'] else 'Anonymous'
        self.black_player = game_data['players']['black']['user']['name'] if 'user' in game_data['players']['black'] else 'Anonymous'
        self.winner = game_data.get('winner', 'draw')
        self.termination = game_data.get('status', 'Unknown')
        self.white_elo = game_data['players']['white'].get('rating', 'Unknown')
        self.black_elo = game_data['players']['black'].get('rating', 'Unknown')
        self.opening = game_data.get('opening', {}).get('name', 'Unknown')
        self.moves = game_data.get('moves', '').split()
        self.evaluations = [analysis.get('eval', 'N/A') for analysis in game_data.get('analysis', [])]
        self.clocks = game_data.get('clocks', [])
        self.username = username

    def is_white(self):
        """Retourne True si l'utilisateur est le joueur blanc."""
        return self.username == self.white_player

    def user_evaluations(self):
        """Retourne les évaluations pour les coups de l'utilisateur uniquement."""
        indices = slice(0, None, 2) if self.is_white() else slice(1, None, 2)
        return self.evaluations[indices]

    def user_clocks(self):
        """Retourne les temps pour les coups de l'utilisateur uniquement."""
        indices = slice(0, None, 2) if self.is_white() else slice(1, None, 2)
        return self.clocks[indices]

class ChessPlayer:
    """
    Classe pour représenter un joueur et ses parties.
    """
    def __init__(self, username, max_games=200):
        self.username = username
        self.max_games = max_games
        self.games = self._fetch_games()

    def _fetch_games(self):
        """Récupère et transforme les données des parties en objets ChessGame."""
        raw_data = LichessAPI.fetch_games(self.username, self.max_games)
        games = []
        for line in raw_data.splitlines():
            game_data = json.loads(line)
            games.append(ChessGame(game_data, self.username))
        return games

    def to_dataframe(self):
        """Transforme les données des parties en DataFrame Pandas."""
        records = []
        for game in self.games:
            record = {
                'type': game.type,
                'white_player': game.white_player,
                'black_player': game.black_player,
                'winner': game.winner,
                'termination': game.termination,
                'white_elo': game.white_elo,
                'black_elo': game.black_elo,
                'opening': game.opening,
                'move_count': len(game.moves),
                'user_evaluations': game.user_evaluations(),
                'user_clocks': game.user_clocks(),
            }
            records.append(record)
        return pd.DataFrame(records)



# Exemple d'utilisation
if __name__ == "__main__":
    username = "EricRosen"
    player = ChessPlayer(username, max_games=20)

    # Convertir les données des parties en DataFrame
    df = player.to_dataframe()
    #print(df.head())
    #print(df.iloc[0])
    

# regressions lineaire sur la perte en centipions au ième coup.
def lin_reg_blitz(n,i):
  tab_evals=[]
  tab_clocks=[]
  
  parties_blitz=df[df['type']=='blitz']
  for j in range (min(n, len(parties_blitz))):

    if i < len(parties_blitz.iloc[j]['user_evaluations']) and i < len(parties_blitz.iloc[j]['user_clocks']):

      tab_evals=tab_evals+[[parties_blitz.iloc[j]['user_evaluations'][i]]]
      tab_clocks=tab_clocks+[ parties_blitz.iloc[j]['user_clocks'][i]]
  
     

  reg=LinearRegression()
  reg.fit(tab_evals,tab_clocks)
  return reg.coef_,reg.intercept_

lin_reg_blitz(19,30)
