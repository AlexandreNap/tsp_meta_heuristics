# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:24:26 2020

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import greedy_v as grd
import colonie_de_fourmis as col
import test_heuristiques as th
import GSA as gsa
import time


tab_res = np.zeros((9,5))
#pour i nb de villes de 3 à 11
for i in range(3,12):
    temp = np.zeros((10,5))
    #on a un tableau de 10 lignes (une par pb)
    #5 colonnes (les temps d'exec et erreur relatives)
    
    #pour j (de 0 a 9) un pb aleatoire
    for j in range(10):
        (villes,matrice) = th.random_cities(i)
        
        pb = col.Antcolonie(matrice,30,10,10,60,0.3,500) 
        (iterations, minSols, minCosts, allCosts, dtAntCol) = pb.start() 
        
        #temps de Ant Colonie
        temp[j][0] = dtAntCol
        
        (sol, mini, dtGlouton) = grd.greedy_ville(matrice)
        
        #erreur relative de Ant Colonie
        temp[j][1] = (np.min(minCosts) - mini) / mini
        
        #temps de glouton
        temp[j][2] = dtGlouton
        
        start = time.time()
        gsol = gsa.gsaMOD(3, matrice, gsa.function, 500, 3)  
        dtGSA = time.time() - start
        
        #temps de gsa
        temp[j][3] = dtGSA
        
        #erreur relative de gsa
        sol = gsa.function(gsol.get_Globalbest(), matrice)
        temp[j][4] = (sol - mini) / mini
    
    #moyenne des différentes données de temps sur l'axe 0 (les lignes donc une valeur moyenne par colonne)
    tab_res[i-3][:] = np.mean(temp, axis=0) 
    #on a un tableau de 9 lignes (une par taille de pb)
    #5 colonnes (les temps d'exec moyens et erreur relatives moyennes)
    #on mets à jour la i-3 eme ligne
    
    
"""
Analyse du temps d'execution
"""
plt.plot(list(range(3,12)),tab_res[:, 0],color='r', label="colonie de fourmis")
plt.plot(list(range(3,12)),tab_res[: ,2],color='b', label="glouton")
plt.plot(list(range(3,12)),tab_res[:, 3],color='g', label="gravitational search")
plt.suptitle("Temps d'exécution moyen en fonction du nombre de villes\nChaque échantillon est une disposition aléatoire de villes")
plt.legend(loc='upper left')
plt.show()



"""
Analyse de l'erreur relative
"""
plt.plot(list(range(3,12)),tab_res[:,1],color='r', label="colonie de fourmis")
plt.plot(list(range(3,12)),tab_res[:,4],color='g', label="gravitational search")
plt.suptitle("Erreur relative moyenne en fonction du nombre de villes")
plt.legend(loc='upper left')
plt.show()



"""
Comparaison des performances entre gsa et AntColonie
"""
#un tableau de 100 lignes (une par execution du pb) et 2 colonnes (les deux couts minimaux)
temp = np.zeros((100,2))

#un pb (une liste de coordonnees x y de ville et la matrice des distances entre elles)
#ici on choisit 15 villes créés aléatoirement 
(villes,matrice) = th.random_cities(15)

for j in range(100):    
    
    pb = col.Antcolonie(matrice,30,10,10,60,0.3,500) 
    (iterations, minSols, minCosts, allCosts, dtAntCol) = pb.start()       
    #le coût min de AntColonie
    temp[j][0] = np.min(minCosts)

    
    gsol = gsa.gsaMOD(3, matrice, gsa.function, 500, 3)      
    sol = gsa.function(gsol.get_Globalbest(), matrice)
    #le coût min de gsa
    temp[j][1] = sol
       
    
    print(j)
    
"""
répartition des coûts minimaux à chaque exécutions
"""
plt.hist(temp[:,0])
plt.suptitle('Répartitions des meilleurs solutions de la colonie de fourmis sur 100 exécutions')
plt.show()
plt.hist(temp[:,1])
plt.suptitle('Répartitions des meilleurs solutions de GSA sur 100 exécutions')
plt.show()

"""
boites à moustaches des coûts minimaux à chaque executions
"""
plt.boxplot([temp[:,0],temp[:,1]])
plt.suptitle('Boites à moustache des meilleurs solutions trouvés par chaque méthode sur 100 exécutions')
plt.gca().xaxis.set_ticklabels(['colonie de fourmis', 'gravitational search'])
plt.show()