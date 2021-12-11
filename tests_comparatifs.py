# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:24:26 2020

@author: alexandre
"""

import numpy as np
import matplotlib.pyplot as plt
import greedy_v as grd
import colonie_de_fourmis as col
import test_heuristiques as th
import GSA as gsa
import time


if __name__ == "__main__":
    mean_performances = np.zeros((9, 5))
    # pour i nb de villes de 3 à 11
    n_generated_problems = 30
    for i in range(3, 12):
        performances = np.zeros((n_generated_problems, 5))
        # on a un tableau de n_generated_problems lignes (une par pb)
        # 5 colonnes (les temps d'exec et erreur relatives par méthodes)

        # random problem de taille i et résolution
        for j in range(n_generated_problems):
            villes, matrice = th.random_cities(i)

            pb = col.Antcolonie(matrice, 30, 10, 10, 60, 0.3, 500)
            _, _, min_costs, _, ant_col_t = pb.start()
            performances[j][0] = ant_col_t

            _, mini, greedy_t = grd.greedy_ville(matrice)
            performances[j][1] = (np.min(min_costs) - mini) / mini
            performances[j][2] = greedy_t

            start = time.time()
            gsol = gsa.gsaMOD(3, matrice, gsa.function, 500, 3)
            gsa_t = time.time() - start
            performances[j][3] = gsa_t
            sol = gsa.function(gsol.get_Globalbest(), matrice)
            performances[j][4] = (sol - mini) / mini

        mean_performances[i-3][:] = np.mean(performances, axis=0)

    """
    Analyse du temps d'execution
    """
    plt.plot(list(range(3, 12)), mean_performances[:, 0], color='r', label="colonie de fourmis")
    plt.plot(list(range(3, 12)), mean_performances[:, 2], color='b', label="glouton")
    plt.plot(list(range(3, 12)), mean_performances[:, 3], color='g', label="gravitational search")
    plt.suptitle("Temps d'exécution moyen en fonction du nombre de villes\nChaque échantillon est une disposition aléatoire de villes")
    plt.legend(loc='upper left')
    plt.show()

    """
    Analyse de l'erreur relative
    """
    plt.plot(list(range(3, 12)), mean_performances[:, 1], color='r', label="colonie de fourmis")
    plt.plot(list(range(3, 12)), mean_performances[:, 4], color='g', label="gravitational search")
    plt.suptitle("Erreur relative moyenne en fonction du nombre de villes")
    plt.legend(loc='upper left')
    plt.show()

    """
    Comparaison des solutions optimales locales entre gsa et AntColonie
    Dépends beaucoups des hyper-paramètres
    """
    # un pb (une liste de coordonnees x y de ville et la matrice des distances entre elles)
    # ici on choisit 15 villes créés aléatoirement
    (villes, matrice) = th.random_cities(15)
    n_generated_problems = 100
    performances = np.zeros((n_generated_problems, 2))

    for j in range(n_generated_problems):
        pb = col.Antcolonie(matrice, 30, 10, 10, 60, 0.3, 500)
        (iterations, minSols, minCosts, allCosts, dtAntCol) = pb.start()
        performances[j][0] = np.min(minCosts)
        gsol = gsa.gsaMOD(3, matrice, gsa.function, 500, 3)
        sol = gsa.function(gsol.get_Globalbest(), matrice)
        performances[j][1] = sol

    """
    répartition des coûts minimaux à chaque exécutions
    """
    plt.hist(performances[:, 0])
    plt.suptitle('Histogramme des meilleures solutions de la colonie de fourmis sur 100 exécutions')
    plt.show()
    plt.hist(performances[:, 1])
    plt.suptitle('Histogramme des meilleures solutions de GSA sur 100 exécutions')
    plt.show()

    plt.boxplot([performances[:, 0], performances[:, 1]])
    plt.suptitle('Boxplot des meilleures solutions trouvées sur 100 exécutions')
    plt.gca().xaxis.set_ticklabels(['colonie de fourmis', 'gravitational search'])
    plt.show()
