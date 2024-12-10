import pandas as pd
import matplotlib.pyplot as plt

def plot_avg_reflection_time(df):
    # Filtrer les parties blitz
    parties_blitz = df[df['type'] == 'blitz']

    # Initialiser un dictionnaire pour stocker les temps par coup
    reflection_times = {}

    # Parcourir les parties
    for _, row in parties_blitz.iterrows():
        user_clocks = row['user_clocks']  # Temps pour chaque coup

        # Calculer le temps de réflexion pour chaque coup (différence entre les temps successifs)
        for i in range(len(user_clocks) - 1):
            time_taken = user_clocks[i] - user_clocks[i + 1]

            # Ne conserver que les valeurs positives
            if time_taken >= 0:
                if i not in reflection_times:
                    reflection_times[i] = []
                reflection_times[i].append(time_taken)

    # Calculer les temps moyens pour chaque coup
    avg_times = {k: sum(v) / len(v) for k, v in reflection_times.items() if len(v) > 0}

    # Préparer les données pour l'affichage
    moves = list(avg_times.keys())
    avg_reflection = list(avg_times.values())

    # Générer l'histogramme
    plt.figure(figsize=(12, 6))
    plt.bar(moves, avg_reflection, color='skyblue', edgecolor='black')
    plt.xlabel("Numéro du coup")
    plt.ylabel("Temps de réflexion moyen (secondes)")
    plt.title("Temps de réflexion moyen par numéro de coup (Blitz)")
    plt.xticks(moves)
    plt.tight_layout()
    plt.show()

# Appel de la fonction pour tracer l'histogramme
plot_avg_reflection_time(df)




import pandas as pd
import matplotlib.pyplot as plt

def plot_win_percentage(df):
    """
    Affiche un camembert avec les pourcentages de victoires, défaites et nulles.
    """
    # Calculer les pourcentages
    total_games = len(df)
    wins = len(df[df['winner'] == 'white']) + len(df[df['winner'] == 'black'])
    draws = total_games - wins

    # Préparer les données
    labels = ['Victoires', 'Défaites', 'Nulles']
    sizes = [wins, total_games - wins - draws, draws]
    colors = ['lightgreen', 'salmon', 'lightblue']

    # Générer le camembert
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title("Pourcentage de Victoires, Défaites et Nulles")
    plt.axis('equal')
    plt.show()

def plot_white_black_win_percentage(df):
    """
    Affiche un camembert avec les pourcentages de victoires avec les blancs et les noirs.
    """
    # Calculer les pourcentages
    white_wins = len(df[df['winner'] == 'white'])
    black_wins = len(df[df['winner'] == 'black'])
    total_wins = white_wins + black_wins

    # Préparer les données
    labels = ['Blancs', 'Noirs']
    sizes = [white_wins / total_wins * 100, black_wins / total_wins * 100]
    colors = ['white', 'black']

    # Générer le camembert
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title("Pourcentage de Victoires par Couleur")
    plt.axis('equal')
    plt.show()

def plot_opening_stats(df):
    """
    Affiche un histogramme des ouvertures les plus fréquentes et leurs résultats.
    """
    # Calculer les statistiques par ouverture
    openings = df['opening'].value_counts().head(10)  # Top 10 ouvertures
    openings_win_rate = df[df['winner'] != 'draw'].groupby('opening')['winner'].value_counts(normalize=True).unstack()

    # Filtrer les ouvertures fréquentes
    openings_win_rate = openings_win_rate.loc[openings.index]

    # Générer un histogramme
    openings_win_rate.plot(kind='bar', stacked=True, figsize=(12, 6), color=['lightgreen', 'salmon'])
    plt.title("Pourcentage de Victoires et Défaites par Ouverture (Top 10)")
    plt.ylabel("Pourcentage")
    plt.xlabel("Ouverture")
    plt.xticks(rotation=45)
    plt.legend(["Victoire Blancs", "Victoire Noirs"])
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
plot_win_percentage(df)
plot_white_black_win_percentage(df)
plot_opening_stats(df)





