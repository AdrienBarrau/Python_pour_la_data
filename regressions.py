from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def preprocess_time_by_color(df, i, max_time_per_move=180):
    """
    Prétraite les données en séparant les temps des blancs et des noirs.

    Args:
        df (pd.DataFrame): DataFrame contenant les parties.
        i (int): Index du coup à analyser (0 pour le premier coup).
        max_time_per_move (float): Temps maximal autorisé pour jouer un coup (en secondes).

    Returns:
        pd.DataFrame: DataFrame avec les colonnes 'time_used' et 'centipawn_loss'.
    """
    processed_data = []

    for _, row in df.iterrows():
        clocks = row["clocks"]
        evaluations = row["evaluations"]

        # Vérifier si les données sont valides
        if (
            isinstance(clocks, list)
            and isinstance(evaluations, list)
            and len(clocks) > i
            and len(evaluations) > i
        ):
            try:
                if i % 2 == 0:  # Coup des blancs
                    current_time = clocks[i]
                    previous_time = clocks[i - 2] if i - 2 >= 0 else None
                else:  # Coup des noirs
                    current_time = clocks[i]
                    previous_time = clocks[i - 2] if i - 2 >= 0 else None

                # Convertir les horloges en secondes
                current_time_sec = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(current_time.split(":"))))
                previous_time_sec = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(previous_time.split(":")))) if previous_time else None

                # Calcul du temps utilisé
                time_used = previous_time_sec - current_time_sec if previous_time_sec else None

                diff_eval=abs(evaluations[i]-evaluations[i-1])
                # Filtrer les valeurs absurdes
                if time_used is not None and 0 <= time_used <= max_time_per_move:
                    processed_data.append({
                        "time_used": time_used,
                        "centipawn_loss": diff_eval
                    })

            except Exception as e:
                # Ignorer les erreurs dues à des formats incorrects
                continue

    return pd.DataFrame(processed_data)

def perform_regression_on_ith_move(df, i):
    """
    Effectue une régression linéaire sur les données du iᵉ coup.

    Args:
        df (pd.DataFrame): DataFrame contenant les parties.
        i (int): Index du coup à analyser (0 pour le premier coup).

    Returns:
        None
    """
    # Prétraiter les données pour le iᵉ coup
    processed_df = preprocess_time_by_color(df, i)

    if processed_df.empty:
        print("Pas de données valides pour le coup", i)
        return

    # Supprimer les lignes contenant des NaN
    processed_df = processed_df.dropna(subset=["time_used", "centipawn_loss"])

    if processed_df.empty:
        print("Pas de données après suppression des NaN pour le coup", i)
        return

    # Diviser les données
    X = processed_df[["time_used"]].values

    y = processed_df["centipawn_loss"].values

    # Vérifier qu'il y a suffisamment de données pour la régression
    if len(X) < 2:
        print("Pas assez de données pour effectuer une régression pour le coup", i)
        return

    # Séparer en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle de régression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Affichage des résultats
    print("Erreur quadratique moyenne :", mean_squared_error(y_test, y_pred))
    print("Coefficient de détermination (R²) :", r2_score(y_test, y_pred))

    # Visualisation
    plt.scatter(X, y, color="blue", label="Données réelles")
    plt.plot(X_test, y_pred, color="red", label="Prédictions (Régression)")
    plt.xlabel("Temps mis (secondes)")
    plt.ylabel("Perte de pions")
    plt.title(f"Régression entre le temps mis et la perte de centipions ({i+1}ᵉ coup)")
    plt.legend()
    plt.show()





filtered_evaluated_perso_df_blitz = filtered_evaluated_perso[filtered_evaluated_perso["game_type"] == "Blitz"]


perform_regression_on_ith_move(filtered_perso_blitz, i=2)

#print(filtered_evaluated_df_blitz[["clocks", "evaluations"]].head())

print(len(filtered_evaluated_perso_df_blitz))
print(len(filtered_perso_blitz))

#perform_regression_on_ith_move(filtered_evaluated_df_blitz, i=30)


#perform_regression_on_ith_move(filtered_evaluated_perso_df_blitz, i=10)
#processed_df = preprocess_for_ith_move_time_used(filtered_evaluated_perso_df_blitz, i=4)


