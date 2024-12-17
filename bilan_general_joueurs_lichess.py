class ChessGame:

    def __init__(self, game, username=None):
        self.white_player = game.headers.get("White", "Unknown")
        self.black_player = game.headers.get("Black", "Unknown")
        self.white_elo = int(game.headers.get("WhiteElo", 0)) if game.headers.get("WhiteElo") else None
        self.black_elo = int(game.headers.get("BlackElo", 0)) if game.headers.get("BlackElo") else None
        self.result = game.headers.get("Result", "Unknown")
        self.termination = game.headers.get("Termination", "Unknown")
        self.opening = game.headers.get("Opening", "Unknown")
        self.time_control = game.headers.get("TimeControl", "Unknown")
        self.moves = self._extract_moves(game)
        self.move_count = len(self.moves)
        self.evaluations = [analysis.get('eval', 'N/A') for analysis in game.headers.get('analysis', [])]
        self.clocks = game.headers.get('clocks', [])
        self.username = username

    def _extract_moves(self, game):
        """Extrait les coups de la partie sous forme de liste."""
        return [move.uci() for move in game.mainline_moves()]
    

def fetch_games_from_pgn(pgn_file_path, username=None, max_games=10000):
    """
    Extrait les parties d'un fichier PGN et retourne un DataFrame.

    Args:
        pgn_file_path (str): Chemin vers le fichier PGN.
        username (str): Nom de l'utilisateur pour un filtrage optionnel.
        max_games (int): Nombre maximum de parties à importer.

    Returns:
        pd.DataFrame: Un DataFrame contenant les informations des parties.
    """
    games_list = []
    
    # Ouvrir et lire le fichier PGN
    with open(pgn_file_path) as pgn_file:
        for _ in range(max_games):
            game = chess.pgn.read_game(pgn_file)
            if game is None:  # Fin du fichier
                break

            # Créer un objet ChessGame
            chess_game = ChessGame(game, username=username)
            
            # Ajouter les informations de la partie à la liste
            games_list.append({
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
                "clocks":chess_game.clocks
            })
    
    # Convertir en DataFrame
    return pd.DataFrame(games_list)

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin du fichier PGN
    pgn_file_path = "/lichess_db_standard_rated_2014-09.pgn"

    # Importer les parties dans un DataFrame
    df_games = fetch_games_from_pgn(pgn_file_path, max_games=10000)

    # Afficher les premières lignes du DataFrame
    print("Aperçu des données :")
    print(df_games.head())

    # Filtrer uniquement les parties blitz
    #df_blitz = df_games[df_games["TimeControl"].str.contains("blitz", case=False, na=False)]  il faudra preciser une fourchette de temps
  

elo_min = 1600
elo_max =1800

# Filtrer les parties où les deux joueurs sont dans la tranche définie
filtered_df = df_games[
    (df_games["WhiteElo"] >= elo_min) & (df_games["WhiteElo"] <= elo_max) &
    (df_games["BlackElo"] >= elo_min) & (df_games["BlackElo"] <= elo_max)
]




def statistiques_descriptives(df):

  # Répartition des résultats
  results_count = df["Result"].value_counts()
  print("\nRépartition des résultats :")
  print(results_count)

  # Camembert des résultats
  plt.figure(figsize=(6, 6))
  plt.pie(results_count, labels=results_count.index, autopct='%1.1f%%', colors=["lightgreen", "lightcoral", "lightblue"])
  plt.title("Répartition des résultats (Blancs, Noirs, Nulles)")
  plt.show()


  # Moyenne des coups par partie
  average_moves = df["MoveCount"].mean()
  print(f"\nNombre moyen de coups par partie : {average_moves:.2f}")

    
  # Histogramme des nombres de coups
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

statistiques_descriptives(filtered_df)

print(len(filtered_df))

