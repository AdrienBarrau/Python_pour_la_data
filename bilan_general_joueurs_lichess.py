import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
!pip install chess
import chess.pgn
!pip install zstandard
import zstandard as zstd
import statistics as sta
import datetime
import time


compressed_file = "/content/lichess_db_standard_rated_2013-01.pgn.zst"   # Chemin des fichiers
decompressed_file = "/content/lichess_db_standard_rated_2013-01.pgn"


with open(compressed_file, 'rb') as compressed:   # Décompression du zst
    with open(decompressed_file, 'wb') as decompressed:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(compressed, decompressed)

pgn_file_path=decompressed_file


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

def fetch_games_from_pgn(pgn_file_path, username, max_games=10000):

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

  centipawn_losses = []
  max_moves = 0

  for _, row in df.iterrows():
        evaluations = row["evaluations"]
        if evaluations and isinstance(evaluations, list):
            losses = []
            for i in range(1, len(evaluations)):
                if evaluations[i] is not None and evaluations[i - 1] is not None:
                    loss = abs(evaluations[i] - evaluations[i - 1])
                    losses.append(loss)
                else:
                    losses.append(None)  # Si une évaluation manque, ignorer ce coup
            centipawn_losses.append(losses)
            max_moves = max(max_moves, len(losses))

    # Calculer la perte moyenne par numéro de coup
  losses_per_move = [[] for _ in range(max_moves)]
  for losses in centipawn_losses:
      for move_idx, loss in enumerate(losses):
          if loss is not None:
              losses_per_move[move_idx].append(loss)


  average_losses = [np.mean(move_losses) if len(move_losses) > 0 else None for move_losses in losses_per_move]

  plt.figure(figsize=(12, 6))
  plt.plot(range(1, len(average_losses) + 1), average_losses, marker="o", color="blue")
  plt.title("Perte moyenne de centipions par numéro de coup")
  plt.xlabel("Numéro de coup")
  plt.ylabel("Perte moyenne de centipions")
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def plays_white(data):
    White=[]
    for i in data['White']:
        if i==username:
            White.append(1)
        else:
            White.append(0)
    return White

def won (data) :
    Won=[]
    for i in range(len(data['Result'])):
        if data['Plays_white'][i]==1:
            Won.append(int([*data['Result'][i]][0]))
        else:
            Won.append(int([*data['Result'][i]][2]))
    return Won

def convert_clock(list_clock):
    new_clock=[]
    for i in list_clock :
        x = time.strptime(i.split(',')[0],'%H:%M:%S')
        new_clock.append(datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
    return new_clock

def add_variables_perso (data): 
    # Retrieve only the chosen player evaluations and clocks
    their_times=[]
    their_evals=[]
    for i in range(len(data['White'])):
        if data['White'][i]==1 :
            their_times.append(convert_clock(data['clocks'][i][::2]))
            their_evals.append(data['evaluations'][i][::2])
        else :
            their_times.append(convert_clock(data['clocks'][i][1::2]))
            their_evals.append(data['evaluations'][i][1::2])
    #initiate new variables
    m_o_r=[]
    m_m_1_r=[]
    m_m_2_r=[]
    m_e_r=[]
    m_o_t=[]
    m_m_1_t=[]
    m_m_2_t=[]
    m_e_t=[]
    Len_game=[] 
    for i in range(len(data['White'])):
        n= len(their_evals[i])
        if n>26 : 
            indices=[(0,6), (6,16), (16,26),(26,n)] #split the moves in four categories : opening (5 moves), middle game 1 (10 moves), middle game 2 (10 moves), end game (The remaining moves) 
            means_ev=[sta.mean(their_evals[i][a:b]) for a,b in indices] #compute for each slice the mean rating of the move
            means_time=[sta.mean(their_times[i][a:b]) for a,b in indices] #compute for each slice the mean time taken for the move
            Len_game.append("Long") #Get the length of the game based on the number of moves
        elif n>16 :
            indices=[(0,6), (6,16),(16,n)]
            means_ev=[sta.mean(their_evals[i][a:b]) for a,b in indices]
            means_time=[sta.mean(their_times[i][a:b]) for a,b in indices]
            means_time.append('NaN') #NaN is added if there wasn't enough moves in the game to complete the last slices
            means_ev.append('NaN')
            Len_game.append("Medium")
        elif len(ev)>6 :
            indices=[(0,6), (6,n)]
            means_ev=[sta.mean(their_evals[i][a:b]) for a,b in indices]
            means_time=[sta.mean(their_times[i][a:b]) for a,b in indices]
            means_time+=['NaN','NaN']
            means_ev+=['NaN','NaN']
            Len_game.append("Short")
        else : #If the game had less than 6 moves the analysis in terms of opening, middle game, is not useful : nothing is computed
            means_ev=['Nan' for k in range(4)]
            means_time=['Nan' for k in range(4)]
            Len_game.append("Too Short")
        m_o_r.append(means_ev[0]) #Then the computed measures are added
        m_m_1_r.append(means_ev[1])
        m_m_2_r.append(means_ev[2])
        m_e_r.append(means_ev[3])
        m_o_t.append(means_time[0])
        m_m_1_t.append(means_time[1])
        m_m_2_t.append(means_time[2])
        m_e_t.append(means_time[3])

    data["length_game"]=Len_game
    data["mean_opening_rating"]=m_o_r #the new colums are added to the data set
    data["mean_middle1_rating"]=m_m_1_r
    data["mean_middle2_rating"]=m_m_2_r
    data["mean_end_rating"]=m_e_1_r
    data["mean_opening_time"]=m_o_t
    data["mean_middle1_time"]=m_m_1_t
    data["mean_middle2_time"]=m_m_2_t
    data["mean_end_time"]=m_e_1_t

if __name__ == "__main__":
    df_games = fetch_games_from_pgn(pgn_file_path,username, max_games=10000)
    df_games_perso=fetch_games_from_pgn("/content/games.pgn",username,max_games=1000)
    df_games_perso['Plays_white']= plays_white(df_games_perso)
    df_games_perso['Won']= won(df_games_perso)
    #add_variables_perso(df_games_perso)
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
filtered_evaluated= df_games[df_games["evaluations"].apply(lambda x: isinstance(x, list) and any(eval is not None for eval in x))]
filtered_evaluated_perso = df_games_perso[df_games_perso["evaluations"].apply(lambda x: isinstance(x, list) and any(eval is not None for eval in x))]


#statistiques_descriptives(df_games)
#statistiques_descriptives(filtered_evaluated)
#statistiques_descriptives(df_games_perso)
#statistiques_descriptives(filtered_df_blitz)
#statistiques_descriptives(filtered_df_bullet)
#statistiques_descriptives(filtered_evaluated_perso)
