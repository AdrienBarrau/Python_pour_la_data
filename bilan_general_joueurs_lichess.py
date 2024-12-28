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
import statsmodels.api as sm
import streamlit as st


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

        Events=important_events(self.moves)
        self.white_queen_taken_bool=main_events(Events,'W','Q')[0]
        self.white_queen_taken_move=main_events(Events,'W','Q')[1]
        self.black_queen_taken_bool=main_events(Events,'B','Q')[0]
        self.black_queen_taken_move=main_events(Events,'B','Q')[1]
        self.white_castling_bool=main_events(Events,'W','C')[0]
        self.white_castling_move=main_events(Events,'W','C')[1]
        self.black_castling_bool=main_events(Events,'B','C')[0]
        self.black_castling_move=main_events(Events,'B','C')[1]
        if self.white_queen_taken_bool and self.black_queen_taken_bool :
            self.queen_exchange_bool= abs(self.black_queen_taken_move-self.white_queen_taken_move)<2
        else : self.queen_exchange_bool= False
        if self.queen_exchange_bool :
            self.queen_exchange_move=max(self.white_queen_taken_move,self.black_queen_taken_move)
        else : self.queen_exchange_move='NaN'

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
                "game_type":chess_game.game_type,
                "white_queen_taken_bool":chess_game.white_queen_taken_bool,
                "white_queen_taken_move":chess_game.white_queen_taken_move,
                "black_queen_taken_bool":chess_game.black_queen_taken_bool,
                "black_queen_taken_move":chess_game.black_queen_taken_move,
                "white_castling_bool":chess_game.white_castling_bool,
                "white_castling_move":chess_game.white_castling_move,
                "black_castling_bool":chess_game.black_castling_bool,
                "black_castling_move":chess_game.black_castling_move,
                "queen_exchange_bool":chess_game.queen_exchange_bool,
                "queen_exchange_move":chess_game.queen_exchange_move
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

def plays_white(data): #this function takes the dataframe of the games of one player and gives back a column composed of 0s and 1s : 1 if the game was won
    White=[]
    for i in data['White']:
        if i==username:
            White.append(1)
        else:
            White.append(0)
    return White

def won (data) : #this function takes the dataframe of the games of one player and gives back a column composed of 0s and 1s : 1 if the player played white
    Won=[]
    for i in range(len(data['Result'])):
        if data['Plays_white'][i]==1:
            Won.append(int([*data['Result'][i]][0]))
        else:
            Won.append(int([*data['Result'][i]][2]))
    return Won

def convert_clock(list_clock): #this function converts a chess clock in format '00:00:59" into an integer that is the number of seconds
    new_clock=[]
    for i in list_clock :
        x = time.strptime(i.split(',')[0],'%H:%M:%S')
        new_clock.append(datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
    return new_clock


def add_variables_perso (data): #this function takes the dataframe of the games of one player and adds a few columns to the data set, namely the mean evaluations and seconds used at relevant moments of the game
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
    for i in range(len(data['White'])): #compute the variables
        try : 
            First_eval=their_evals[i][0]
        except IndexError :
            their_evals[i]=[None]
        if (First_eval is None) :
            means_ev=['NaN' for k in range(4)]
            n= len(their_times[i])
            if n>26 : 
                indices=[(0,6), (6,16), (16,26),(26,n)] #split the moves in four categories : opening (5 moves), middle game 1 (10 moves), middle game 2 (10 moves), end game (The remaining moves) 
                means_time=[sta.mean(their_times[i][a:b]) for a,b in indices] #compute for each slice the mean time taken for the move
                Len_game.append("Long") #Get the length of the game based on the number of moves
            elif n>16 :
                indices=[(0,6), (6,16),(16,n)]
                means_time=[sta.mean(their_times[i][a:b]) for a,b in indices]
                means_time.append('NaN') #NaN is added if there wasn't enough moves in the game to complete the last slices
                Len_game.append("Medium")
            elif n>6 :
                indices=[(0,6), (6,n)]
                means_time=[sta.mean(their_times[i][a:b]) for a,b in indices]
                means_time+=['NaN','NaN']
                Len_game.append("Short")
            else : #If the game had less than 6 moves the analysis in terms of opening, middle game, is not useful : nothing is computed
                means_time=['NaN' for k in range(4)]
                Len_game.append("Too Short")
        else :
            their_evals_clean_i=[]
            for j in their_evals[i] :
                if j is None :
                    their_evals_clean_i.append(0)
                else :
                    their_evals_clean_i.append(j)
            n= len(their_evals_clean_i)
            if n>26 : 
                indices=[(0,6), (6,16), (16,26),(26,n)] #split the moves in four categories : opening (5 moves), middle game 1 (10 moves), middle game 2 (10 moves), end game (The remaining moves) 
                means_ev=[sta.mean(their_evals_clean_i[a:b]) for a,b in indices] #compute for each slice the mean rating of the move
                means_time=[sta.mean(their_times[i][a:b]) for a,b in indices] #compute for each slice the mean time taken for the move
                Len_game.append("Long") #Get the length of the game based on the number of moves
            elif n>16 :
                indices=[(0,6), (6,16),(16,n)]
                means_ev=[sta.mean(their_evals_clean_i[a:b]) for a,b in indices]
                means_time=[sta.mean(their_times[i][a:b]) for a,b in indices]
                means_time.append('NaN') #NaN is added if there wasn't enough moves in the game to complete the last slices
                means_ev.append('NaN')
                Len_game.append("Medium")
            elif n>6 :
                indices=[(0,6), (6,n)]
                means_ev=[sta.mean(their_evals_clean_i[a:b]) for a,b in indices]
                means_time=[sta.mean(their_times[i][a:b]) for a,b in indices]
                means_time+=['NaN','NaN']
                means_ev+=['NaN','NaN']
                Len_game.append("Short")
            else : #If the game had less than 6 moves the analysis in terms of opening, middle game, is not useful : nothing is computed
                means_ev=['NaN' for k in range(4)]
                means_time=['NaN' for k in range(4)]
                Len_game.append("Too Short")
    
        m_o_r.append(means_ev[0]) #Then the computed measures are added
        m_m_1_r.append(means_ev[1])
        m_m_2_r.append(means_ev[2])
        m_e_r.append(means_ev[3])
        m_o_t.append(means_time[0])
        m_m_1_t.append(means_time[1])
        m_m_2_t.append(means_time[2])
        m_e_t.append(means_time[3])
    Enemy_queen_move=[] #initiate new variables
    Enemy_castling_move=[]
    Enemy_castling_dummy=[]
    Player_queen_move=[]
    Player_castling_move=[]
    Player_castling_dummy=[]
    Elo_diff=[]
    for i in range(len(data['White'])): #splits the games info into variables specific to the player considered or their opponent
        if data['White'][i]==1 :
            Elo_diff.append(data['WhiteElo'][i]-data['BlackElo'][i])
            Enemy_queen_move.append(data['black_queen_taken_move'][i])
            Enemy_castling_move.append(data['black_castling_move'][i])
            Player_queen_move.append(data['white_queen_taken_move'][i])
            Player_castling_move.append(data['white_castling_move'][i])
            if data["white_castling_bool"][i]:
                Player_castling_dummy.append(1)
            else :
                Player_castling_dummy.append(0)
            if data["black_castling_bool"][i]:
                Enemy_castling_dummy.append(1)
            else :
                Enemy_castling_dummy.append(0)
        else :
            Elo_diff.append(data['BlackElo'][i]-data['WhiteElo'][i])
            Player_queen_move.append(data['black_queen_taken_move'][i])
            Player_castling_move.append(data['black_castling_move'][i])
            Enemy_queen_move.append(data['white_queen_taken_move'][i])
            Enemy_castling_move.append(data['white_castling_move'][i])
            if data["black_castling_bool"][i]:
                Player_castling_dummy.append(1)
            else :
                Player_castling_dummy.append(0)
            if data["white_castling_bool"][i]:
                Enemy_castling_dummy.append(1)
            else :
                Enemy_castling_dummy.append(0)
    L_g_ts=[] #initiate new variables
    L_g_s=[]
    L_g_m=[]
    L_g_l=[]
    for i in Len_game : #Len game is transformed into four dummies
        if i=='Too Short':
            L_g_ts.append(1)
            L_g_s.append(0)
            L_g_m.append(0)
            L_g_l.append(0)
        elif i=='Short':
            L_g_ts.append(0)
            L_g_s.append(1)
            L_g_m.append(0)
            L_g_l.append(0)
        elif i=='Medium':
            L_g_ts.append(0)
            L_g_s.append(0)
            L_g_m.append(1)
            L_g_l.append(0)
        elif i=='Long':
            L_g_ts.append(0)
            L_g_s.append(0)
            L_g_m.append(0)
            L_g_l.append(1)
    queen_exchange_dummy=[]

    for i in data["queen_exchange_bool"]:
        if i :
            queen_exchange_dummy.append(1)
        else :
            queen_exchange_dummy.append(0)
    data["Player_castling_dummy"]=Player_castling_dummy #the new colums are added to the data set
    data["Enemy_castling_dummy"]=Enemy_castling_dummy
    data["queen_exchange_dummy"]=queen_exchange_dummy
    data["Enemy_queen_taken_move"]=Enemy_queen_move
    data["Enemy_castling_move"]=Enemy_castling_move
    data["Player_queen_taken_move"]=Player_queen_move
    data["Player_castling_move"]=Player_castling_move
    data["Elo_diff"]=Elo_diff
    data["length_game_too_short"]=L_g_ts
    data["length_game_short"]=L_g_s
    data["length_game_medium"]=L_g_m
    data["length_game_long"]=L_g_l
    data["mean_opening_rating"]=m_o_r
    data["mean_middle1_rating"]=m_m_1_r
    data["mean_middle2_rating"]=m_m_2_r
    data["mean_end_rating"]=m_e_r
    data["mean_opening_time"]=m_o_t
    data["mean_middle1_time"]=m_m_1_t
    data["mean_middle2_time"]=m_m_2_t
    data["mean_end_time"]=m_e_t

def convert(a,b) : #this function takes a position on the board in format 'a4' and converts it to a tuple composed of two integers
    if a == 'a' : return (11,int(b))
    elif a == 'b' : return (12,int(b))
    elif a == 'c' : return (13,int(b))
    elif a == 'd' : return (14,int(b))
    elif a == 'e' : return (15,int(b))
    elif a == 'f' : return (16,int(b))
    elif a == 'g' : return (17,int(b))
    elif a == 'h' : return (18,int(b))

def important_events (moves) : #This function takes a sequence of moves in UCI format, simulates the game and lists all important events in the game (a piece taken or castling)
    Initial_board=[(a,b) for a in [11,12,13,14,15,16,17,18] for b in [1,2,3,4,5,6,7,8]] #creates the initial empty board of positions
    Pieces=[('WRa',5),('WPa',1),0,0,0,0,('BPa',1),('BRa',5), #a piece is described by a tuple : a string with its color 'W', its type 'R' and its column of start if it needs to be differentiated 'a' ; and an integer with its "value"
    ('WNb',3),('WPb',1),0,0,0,0,('BPb',1),('BNb',3), 
    ('WBc',3),('WPc',1),0,0,0,0,('BPc',1),('BBc',3),
    ('WQ',9),('WPd',1),0,0,0,0,('BPd',1),('BQ',9),
    ('WK',0),('WPe',1),0,0,0,0,('BPe',1),('BK',0),
    ('WBf',3),('WPf',1),0,0,0,0,('BPf',1),('BBf',3),
    ('WNg',3),('WPg',1),0,0,0,0,('BPg',1),('BNg',3),
    ('WRh',5),('WPh',1),0,0,0,0,('BPh',1),('BRh',5)] # lists all pieces in the game
    Board={}
    k=0
    for j in Initial_board : # Assignate in a dictionnary a piece or a 0 (items) for each position (keys)
        Board[j]=Pieces[k]
        k+=1
    number_move=1
    Events=[]
    for i in moves :
        uci_list=[*i]
        initial=convert(uci_list[0],uci_list[1])
        final=convert(uci_list[2],uci_list[3]) #converts the uci format into our format of starting and end position for a moving piece
        if Board[final]!=0 : #If there is already a piece in the final position, that piece and the move are added to the Events list
            Events.append([Board[final],number_move])
        Board[final]=Board[initial]
        Board[initial]=0
        if i== 'e1g1' and number_move<30 : #If there is castling the rook also needs to be moved, and the castling is added to the Events list
            initial=convert('h','1')
            final=convert('f','1')
            Board[final]=Board[initial]
            Board[initial]=0
            Events.append([('WCS',0),number_move]) #CS stands for 'Castling' 'Small'
        elif i== 'e1c1' and number_move<30:
            initial=convert('a','1')
            final=convert('d','1')
            Board[final]=Board[initial]
            Board[initial]=0
            Events.append([('WCG',0),number_move]) #CG stands for 'Castling' 'Great'
        elif i== 'e8g8' and number_move<30:
            initial=convert('h','8')
            final=convert('f','8')
            Board[final]=Board[initial]
            Board[initial]=0
            Events.append([('BCS',0),number_move])
        elif i== 'e8c8' and number_move<30:
            initial=convert('a','8')
            final=convert('d','8')
            Board[final]=Board[initial]
            Board[initial]=0
            Events.append([('BCG',0),number_move])
        number_move+=1
    return Events

def main_events (events,color,piece) : #This function takes a list of events in a game (computed by important events), the color as a string 'W', the type of piece as a string 'Q', and gives back a tuple
    #first a boolean of wether or not that piece was taken during the game (or the castling made), then either an integer (if that piece was taken) that stands for the move at which it happened; or a string 'NaN' if the piece was not taken
    for i in events:
        if [*i[0][0]][0]==color and [*i[0][0]][1]==piece :
            return (True,i[1])
    return (False,'NaN')

username='EricRosen'

if __name__ == "__main__":
    df_games = fetch_games_from_pgn(pgn_file_path,username, max_games=10000)
    df_games_perso=fetch_games_from_pgn("/content/games.pgn",username,max_games=1000)
    df_games_perso['Plays_white']= plays_white(df_games_perso)
    df_games_perso['Won']= won(df_games_perso)
    add_variables_perso(df_games_perso)

def significant_predictors(username,max_games=2000): #This function takes the username of a Lichess player, and the number of their games you want to consider
    #It runs a logistic regression on a few variables of interest, and gives back a list of the significant predicors, and wether or not they increase this username's Lichess player chance of winning (positive predictors) or losing (negative predictor)
    data=fetch_games_from_pgn("/home/onyxia/Python_pour_la_data/games.pgn",username,max_games)
    data['Plays_white']= plays_white(data)
    data['Won']= won(data)
    add_variables_perso(data)
    Intercept=[1 for i in range(len(data["Plays_white"]))]
    data["Intercept"]=Intercept
    variables=['Intercept',"length_game_short","length_game_medium","length_game_long","queen_exchange_dummy","Enemy_castling_dummy","Player_castling_dummy"]
    columns=data[['Won']+variables]
    columns=columns.dropna(how='any')
    y=columns.iloc[:,0].to_numpy()
    all_variables_columns=[i for i in range(1,len(variables)+1)]
    print(all_variables_columns)
    x=columns.iloc[:,all_variables_columns].to_numpy()
    model = sm.Logit(y,x)
    result=model.fit()
    positive_predictors=[]
    negative_predictors=[]
    for i in range(1,len(variables)):
        if result.pvalues[i]<0.05 and result.params[i]<0:
            negative_predictors.append(variables[i])
        elif result.pvalues[i]<0.05 and result.params[i]>0:
            positive_predictors.append(variables[i])
    print(result.summary())
    return positive_predictors, negative_predictors



# APP STREAMLIT
# Exécuter --->  python -m streamlit run Python4DS/predictors_streamlit.py
st.title("Analyse des prédicteurs significatifs")

username = st.text_input("Pseudonyme Lichess")
max_games = st.number_input("Nombre de parties à considérer pour l'analyse", min_value=1, max_value=2000, value=100, step=1)

dic = {
    "length_game_short": "une partie rapide",
    "length_game_medium": "une partie de durée intermédiaire",
    "length_game_long": "une partie longue",
    "queen_exchange_dummy": "un échange de dames",
    "Enemy_castling_dummy": "un roque de la part de votre adversaire",
    "Player_castling_dummy": "un roque de votre part"
}

if st.button("Analyser"):
    if username != "":
        try:
            positive_predictors, negative_predictors = significant_predictors(username, max_games)

            st.subheader("Résultats")

            text = f"Au regard des {max_games} parties considérées dans cette analyse, les conclusions sont les suivantes :\n"
            if positive_predictors:
                french_pos_pred = ", ".join([dic[p] for p in positive_predictors if p in dic])
                text += f"- :green[**Prédicteurs positifs :**] {french_pos_pred} augmentent vos chances de victoire.\n"
            else:
                text += "- Aucun prédicteur positif significatif.\n"

            if negative_predictors:
                french_neg_pred = ", ".join([dic[p] for p in negative_predictors if p in dic])
                text += f"- :red[**Prédicteurs négatifs :**] {french_neg_pred} diminuent vos chances de victoire.\n"
            else:
                text += "- Aucun prédicteur négatif significatif."
            
            st.write(text)
        except Exception as error:
            st.error(f"Erreur : {error}")
    else:
        st.warning("Entrer un pseudonyme Lichess.")



# affichage et filtrations



elo_min =1700
elo_max =2000

#Ajoutez un filtre, celui la garde les parties ou le elo des DEUX JOUEURS est entre elo_min, elo_max

filtered_df_elo = df_games[(df_games["WhiteElo"] >= elo_min) & (df_games["WhiteElo"] <= elo_max) &(df_games["BlackElo"] >= elo_min) & (df_games["BlackElo"] <= elo_max)]

filtered_df_blitz = df_games[df_games["game_type"] == "Blitz"]
filtered_df_bullet = df_games[df_games["game_type"] == "Bullet"]
filtered_evaluated= df_games[df_games["evaluations"].apply(lambda x: isinstance(x, list) and any(eval is not None for eval in x))]
filtered_evaluated_perso = df_games_perso[df_games_perso["evaluations"].apply(lambda x: isinstance(x, list) and any(eval is not None for eval in x))]

filtered_df_elo_blitz=filtered_df_elo[filtered_df_elo["game_type"] == "Blitz"]
filtered_df_elo_bullet=filtered_df_elo[filtered_df_elo["game_type"] == "Bullet"]
#print(df_games["evaluations"])
#print(filtered_evaluated_perso["evaluations"][12])

#statistiques_descriptives(df_games)
statistiques_descriptives(filtered_df_bullet)
statistiques_descriptives(filtered_df_blitz)

statistiques_descriptives(filtered_df_elo_blitz)
statistiques_descriptives(filtered_df_elo_bullet)


#statistiques_descriptives(filtered_evaluated)
#statistiques_descriptives(df_games_perso)
#statistiques_descriptives(filtered_df_blitz)
#statistiques_descriptives(filtered_df_bullet)
#statistiques_descriptives(filtered_evaluated_perso)

print(len(filtered_evaluated)/len(df_games))
print(len(filtered_evaluated_perso)/len(df_games_perso))

