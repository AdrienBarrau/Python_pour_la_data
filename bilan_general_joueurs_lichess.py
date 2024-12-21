import pandas as pd
import matplotlib.pyplot as plt
!pip install chess
import chess.pgn
class ChessGame:

    def __init__(self, game, username):
        self.white_player = game.headers.get("White", "Unknown")
        self.black_player = game.headers.get("Black", "Unknown")
         # Gestion des valeurs Elo invalides
        try:
            self.white_elo = int(game.headers.get("WhiteElo", 0)) if game.headers.get("WhiteElo") and game.headers.get("WhiteElo").isdigit() else None
        except ValueError:
            self.white_elo = None

        try:
            self.black_elo = int(game.headers.get("BlackElo", 0)) if game.headers.get("BlackElo") and game.headers.get("BlackElo").isdigit() else None
        except ValueError:
            self.black_elo = None
        self.result = game.headers.get("Result", "Unknown")
        self.termination = game.headers.get("Termination", "Unknown")
        self.opening = game.headers.get("Opening", "Unknown")
        self.time_control = game.headers.get("TimeControl", "Unknown")
        self.moves, self.evaluations, self.clocks = self._extract_moves(game)
        self.move_count = len(self.moves)
        
        self.game_type = self._classify_game_type(self.time_control)  # bullet (le plus rapide), blitz, rapid ou classique (le plus lent)
        self.username = username


    def _extract_moves(self, game):
        moves = []
        evaluations = []  #tab des evaluations
        clocks = []  #tab du temps restant

        for node in game.mainline():
            move = node.move.uci()
            moves.append(move)

            if "eval" in node.comment:
                eval_str = node.comment.split("[%eval ")[1].split("]")[0]  
                try:
                    evaluations.append(float(eval_str))
                except ValueError:
                    evaluations.append(None)
            else:
                evaluations.append(None)

            # Récupérer le temps restant
            if "clk" in node.comment:
                clk_str = node.comment.split("[%clk ")[1].split("]")[0]
                clocks.append(clk_str)
            else:
                clocks.append(None)

        return moves, evaluations, clocks

    def _classify_game_type(self, time_control):
        """
        Classe le type de la partie en fonction du TimeControl.
        Exemples de time_control :
        - "300+5" : 5 minutes avec incrément 5 sec (blitz)
        - "60+0" : 1 minute sans incrément (bullet)
        - "1800+0" : 30 minutes sans incrément (rapid)
        """
        if not time_control or time_control == "-":
            return "Unknown"

        try:
            base, increment = time_control.split("+")
            base = int(base)  # Temps de base en secondes
            increment = int(increment)  # Incrément

            total_time = base + (40 * increment)
            if total_time < 180:  # 180 secondes=3 minutes
                return "Bullet"
            elif total_time <= 600:
              return "Blitz"
            elif total_time <= 1800:
              return "Rapid"
            else:
              return "Classical"   #plus de 30 minutes
        except ValueError:
            return "Unknown"

def fetch_games_from_pgn(pgn_file_path, username, max_games=1000):

    games_list = []


    with open(pgn_file_path) as pgn_file:
        for _ in range(max_games):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break


            chess_game = ChessGame(game, username=username)


            games_list.append({           # d'autres variables peuvent s'ajouter
                "White": chess_game.white_player,
                "Black": chess_game.black_player,
                "WhiteElo": chess_game.white_elo,
                "BlackElo": chess_game.black_elo,
                "Result": chess_game.result,
                "Termination": chess_game.termination,
                "Opening": chess_game.opening,
                "TimeControl": chess_game.time_control,
                "MoveCount": chess_game.move_count,
                "Moves": chess_game.moves,
                "evaluations": chess_game.evaluations,
                "clocks":chess_game.clocks,
                "game_type":chess_game.game_type
            })

    return pd.DataFrame(games_list)




def statistiques_descriptives(df):


  results_count = df["Result"].value_counts()
  print("\nRépartition des résultats :")
  print(results_count)


  plt.figure(figsize=(6, 6))
  plt.pie(results_count, labels=results_count.index, autopct='%1.1f%%', colors=["lightgreen", "lightcoral", "lightblue"])
  plt.title("Répartition des résultats (Blancs, Noirs, Nulles)")
  plt.show()



  average_moves = df["MoveCount"].mean()
  print(f"\nNombre moyen de coups par partie : {average_moves:.2f}")


  plt.figure(figsize=(8, 5))
  plt.hist(df["MoveCount"], bins=20, color="skyblue", edgecolor="black")
  plt.title("Distribution du nombre de coups par partie")
  plt.xlabel("Nombre de coups")
  plt.ylabel("Fréquence")
  plt.show()



  df = df.rename(columns={"opening": "Opening", "result": "Result", "winner": "Winner"})
  df["Result"] = df["Result"].replace({"1-0": "White", "0-1": "Black", "1/2-1/2": "Draw"})

  top_openings = df["Opening"].value_counts().head(20)
  opening_win_rate = (
        df[df["Result"] != "Draw"]
        .groupby("Opening")["Result"]
        .value_counts(normalize=True)
        .unstack()
    )


  opening_win_rate = opening_win_rate.loc[top_openings.index]


  opening_win_rate.plot(
        kind="bar", stacked=True, figsize=(12, 6), color=["lightgreen", "salmon"]
    )
  plt.title("Pourcentage de Victoires et Défaites par Ouverture (Top 20)")
  plt.ylabel("Pourcentage")
  plt.xlabel("Ouverture")
  plt.xticks(rotation=45)
  plt.legend(["Victoire Blancs", "Victoire Noirs"])
  plt.tight_layout()
  plt.show()
  print(df["game_type"])
  game_type_counts = df["game_type"].value_counts()
  print("\nNombre de parties par type :")
  print(game_type_counts)

  # Visualisation sous forme de diagramme
  plt.figure(figsize=(8, 6))
  game_type_counts.plot(kind="bar", color=["lightblue", "lightgreen", "lightcoral", "gold"])
  plt.title("Nombre de parties par type")
  plt.xlabel("Type de partie")
  plt.ylabel("Nombre de parties")
  plt.xticks(rotation=0)
  plt.show()



if __name__ == "__main__":
    df_games = fetch_games_from_pgn(pgn_file_path,username, max_games=1000)  #toutes les parties lichess sur un mois
    df_games_perso=fetch_games_from_pgn("/content/games.pgn",username,max_games=1000)   #parties d un certain utilisateur
    #print(df_games.head())

# FILTRATION DU DATAFRAME

elo_min = 1600
elo_max =1800

# Ajoutez un filtre, celui la garde les parties ou le elo des DEUX JOUEURS est entre elo_min, elo_max
filtered_df_elo = df_games[
    (df_games["WhiteElo"] >= elo_min) & (df_games["WhiteElo"] <= elo_max) &
    (df_games["BlackElo"] >= elo_min) & (df_games["BlackElo"] <= elo_max)
]

filtered_df_blitz = df_games[df_games["game_type"] == "Blitz"]
filtered_df_bullet = df_games[df_games["game_type"] == "Bullet"]
filtered_evaluated_perso = df_games_perso[df_games_perso["evaluations"].apply(lambda x: isinstance(x, list) and any(eval is not None for eval in x))]
#statistiques_descriptives(df_games)
statistiques_descriptives(df_games_perso)
#statistiques_descriptives(filtered_df_blitz)
#statistiques_descriptives(filtered_df_bullet)
statistiques_descriptives(filtered_evaluated_perso)
