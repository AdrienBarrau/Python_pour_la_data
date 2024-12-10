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
