import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Fonction statistiques_descriptives

def statistiques_descriptives(df):
    st.subheader("Statistiques descriptives")

    if st.button("Répartition des résultats"):
        results_count = df["Result"].value_counts()
        st.write("Répartition des résultats")
        # print(results_count)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(results_count, labels=results_count.index, autopct='%1.1f%%', colors=["lightgreen", "lightcoral", "lightblue"])
        ax.set_title("Répartition des résultats (Blancs, Noirs, Nulles)")
        st.pyplot(fig)

    if st.button("Distribution du nombre de coups par partie"):
        average_moves = df["MoveCount"].mean()
        st.write(f"Nombre moyen de coups par partie : {average_moves:.2f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["MoveCount"], bins=20, color="skyblue", edgecolor="black")
        ax.set_title("Distribution du nombre de coups par partie")
        ax.set_xlabel("Nombre de coups")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)

    df = df.rename(columns={"opening": "Opening", "result": "Result", "winner": "Winner"})
    df["Result"] = df["Result"].replace({"1-0": "White", "0-1": "Black", "1/2-1/2": "Draw"})

    if st.button("Victoires par ouverture"):
        top_openings = df["Opening"].value_counts().head(20)
        opening_win_rate = (
            df[df["Result"] != "Draw"]
            .groupby("Opening")["Result"]
            .value_counts(normalize=True)
            .unstack()
        )
        opening_win_rate = opening_win_rate.loc[top_openings.index]

        fig, ax = plt.subplots(figsize=(12, 6))
        opening_win_rate.plot(
            kind="bar", stacked=True, ax=ax, color=["lightgreen", "salmon"]
        )
        ax.set_title("Pourcentage de Victoires et Défaites par Ouverture (Top 20)")
        ax.set_ylabel("Pourcentage")
        ax.set_xlabel("Ouverture")
        ax.legend(["Victoire Blancs", "Victoire Noirs"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    if st.button("Nombre de parties par type"):
        game_type_counts = df["game_type"].value_counts()
        st.write("Nombre de parties par type")
        st.write(game_type_counts)

        fig, ax = plt.subplots(figsize=(8, 6))
        game_type_counts.plot(kind="bar", ax=ax, color=["lightblue", "lightgreen", "lightcoral", "gold"])
        ax.set_title("Nombre de parties par type")
        ax.set_xlabel("Type de partie")
        ax.set_ylabel("Nombre de parties")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    if st.button("Perte moyenne de centipions par numéro de coup"):
        centipawn_losses = []
        max_moves = 0

        for _, row in df.iterrows():
            evaluations = row.get("evaluations", [])
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

        losses_per_move = [[] for _ in range(max_moves)]
        for losses in centipawn_losses:
            for move_idx, loss in enumerate(losses):
                if loss is not None:
                    losses_per_move[move_idx].append(loss)

        average_losses = [np.mean(move_losses) if len(move_losses) > 0 else None for move_losses in losses_per_move]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(1, len(average_losses) + 1), average_losses, marker="o", color="blue")
        ax.set_title("Perte moyenne de centipions par numéro de coup")
        ax.set_xlabel("Numéro de coup")
        ax.set_ylabel("Perte moyenne de centipions")
        ax.grid(True)
        st.pyplot(fig)




# Affichage de l'interface

st.title("Statistiques descriptives sur les parties")

file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    statistiques_descriptives(df)
else:
    st.warning("Veuillez télécharger un fichier CSV d'abord.")
