# -*- coding: utf-8 -*-
from termcolor import colored

def calcul_distance(chemin, matrice_distances) :
 
    distance_trajet = 0
    for i in range (len(chemin)-1) :
        distance_trajet += matrice_distances[chemin[i]][chemin[i+1]]
    
    return distance_trajet

def afficher_parcours(chemin) :

    trajet = ""
    for indice_ville in chemin :
        trajet += indice_ville + " - "
    print(trajet[:-2])

def greedy_ville(dist) :
    
    from itertools import permutations 
    import time

    # Les distances par les routes
    distances_routes = dist
    ville_depart = 0    
    indice_ville = ville_depart
    indices = list(range(dist.shape[0]))
    indices.remove(indice_ville)
    
    perm = permutations(indices) 
    
    solution = []
    minimum = 10000
    somme = 0
    
    start_time = time.time()
    for i in list(perm) :  
        somme += 1
        
        chemin = [indice_ville] + list(i) + [indice_ville]
        """
        print(chemin)
        print(calcul_distance(chemin, distances_routes))
        print('\n')
        """
        dist = calcul_distance(chemin, distances_routes)
        if  dist < minimum :
            minimum = calcul_distance(chemin, distances_routes)
            solution = chemin
        
    interval = time.time() - start_time
    print("La solution optimale en partant de", ville_depart, "est le trajet :")
    print(colored(solution, 'green'))
    print("La distance est de :", colored(minimum, 'blue'), "km")
    print ('Total time in seconds :', colored(interval, 'red'))
    
    print("Nombre de chemins testés :", somme)
    return (solution, minimum, interval)