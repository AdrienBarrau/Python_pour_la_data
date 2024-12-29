Projet de programmation 2024

Adrien Barrau, Etienne Chastel, Alban Géron


Sujet

Les meilleur.e.s joueu.se.r.s d'échecs, en amont de parties importantes réalisent souvent des analyses de la concurrence. Il s'agit d'abord de se renseigner sur les pratiques les plus communes au sein de la communauté des joueu.se.r.s d'échec. D'avoir en tête les ouvertures pratiquées, les caractéristiques typiques des parties réalisé par les joueu.ses.rs. Il s'agit ensuite de se renseigner sur les habitudes d'un.e joueu.se.r en particulier, en amont d'une partie, pour se préparer au mieux. Cette analyse de la concurrence est essentielle pour se tenir au fait des tendances actuelles et pour se préparer au mieux. Elle est néanmoins coûteuse en temps, pour recueillir et anaylser des données sur un grand nombre de joueu.se.r.s. 


Projet

Ce projet se propose donc, pour les échecs, d'automatiser ces différents aspects de l'analyse de la concurrence. Pouvons-nous fournir aux joueu.se.r.s des analyses des parties d'échecs des autres joueu.se.r.s et de leurs adversaires ?


Données utilisées

Les données utilisées sont celles de la plateforme Lichess (https://database.lichess.org), qui fournit des données sur toutes les parties jouées en ligne sur leur site. 


Statistiques utilisées

Au cours de ce projet nous produisons différents type de statistiques.
Nous fournissons d'abord des statistiques descriptives sur l'ensemble des parties jouées. Le code peut fonctionner sur tout type de base de donnée importée depuis Lichess. Néanmoins, a titre d'exemple, nous proposons ici de nous intéresser aux parties jouées en septembre 2014. L'objectif est de produire des graphiques et résultats utiles autour de quelques variables clés calculées à partir des données brutes. Cela permet d'être mieux renseigné.e sur l'ensemble des parties.
Nous fournissons ensuite des statistiques inférentielles sur l'ensemble des parties jouées par un.e joueu.se.r spécifique de Lichess. A titre d'exemple nous nous intéressons ici aux parties du joueur 'EricRosen'. Le modèle choisi est un modèle logistique : nous regardons quelles variables produisent un effet significatif sur le resultat d'une partie : la victoire ou non du joueur considéré. Cela permet d'obtenir les meilleurs prédicteurs de victoire ou de défaite de ce joueur, et donc d'adapter sa stratégie pour maximiser ses propres chances de victoire face à ce joueur.


Navigation

Pour naviguer dans ce projet il faut d'abord se rendre sur le lien suivant : https://database.lichess.org/#standard_games et importer le fichier "lichess_db_standard_rated_2014-09.pgn.zst". Il suffit ensuite d'executer toutes les cellules du notebook.
