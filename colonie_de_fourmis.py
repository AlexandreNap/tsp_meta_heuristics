# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:49:01 2020

@author: alexandre
"""
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time
from termcolor import colored



class Antcolonie:
    """
        cost est la matrice carrée de couts
        m est le nombre de fourmis
        alpha l'intensité, a quel point on favorise les phéromones
        beta la visibilité,à quel point on favorise la ville la plus proche
        q est une constante  dans le calcul de la quantité de phéromones déposée
        p est la proportion de phéromones qui s'évapore à chaque tour
        tmax est le nombre d'itérations
    """
    def __init__(self, costs, m, alpha, beta, q, p, tmax):
        self.costs = costs        
        self.n = np.shape(costs)[0]  
        self.nu = np.ones((self.n, self.n)) / self.costs
        
        self.phero = np.ones((self.n, self.n))#np.zeros((self.n, self.n))
        self.deltaPhero = np.zeros((self.n, self.n))
        
        self.ants=[]
		
        #paramètres du pb:
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.p = p
        self.m = m
        
        for i in range(0,self.m):
            self.ants.append(Ant(self))
        
        self.tmax = tmax
        
        
    """
    depose les pheromones sur la route entre A et B
"""    
    def drop_phero(self, cityA, cityB, length):
        self.deltaPhero[cityA, cityB] += self.q / length
       
        
    """
        Evaporation des phéromones sur tous les chemins 
        et ajouts des pheromones déposées
"""    
    def update_phero(self):
        self.phero = (1 - self.p) * self.phero + self.deltaPhero
        self.deltaPhero = np.zeros((self.n, self.n))
        
       
    """calcule la prochaine ville selon la ville actuelle, les prochaines
    villes possibles, les parametres du problème
    """
    def next_city_rule(self, i, nextCities):
        probas =[]
        p = 0.0
            
        total = 0
        #on construit un vecteur de répartition de probabilité de choisir
        #chaque ville
        #p[i]= P(uniform(0,1)<p[i])
        #si p[i] < uniform(0,1) < p[i+1] alors on choisira la i eme ville
        for j in nextCities:
            p += self.phero[i, j] ** self.alpha * self.nu[i, j] ** self.beta
            total += p
            probas.append(p)
                
        #on normalise le vecteur pour que tout soit entre 0 et 1
        if (p==0):#il faut tenir compte du debut ou les pheromones sont nulles
            #mais on peut aussi choisir de ne pas les initialiserà zero
            probas = np.arange(1, len(nextCities) + 1) / len(nextCities)
        else :
            probas = probas / p
            
        
        #il restes plus qu'a tirer aléatoirement rand en 0 et 1 et a voir dans
        #quelle partie de [0,1] il est par rapport aux répartitions de chaques
        #villes
        rand = rd.uniform(0,1)        
        j = 0
       
        # print(str(probas))
        # print(str(rand))
        
        while j < len(nextCities) and rand > probas[j] : 
            j += 1
        #print(j)
            
        return nextCities[j]        
    
    def reset_ants(self):
        for ant in self.ants:
            ant.reset()
		
    """
        Boucle de lancement de l'algorithme
    """            
    def start(self):
        minCosts = []
        allCosts = np.zeros((self.tmax,self.m))
        iterations = []
        minSols = []
        start_time = time.time()
        
        for i in range(0, self.tmax): 
            self.reset_ants()
                
            for ant in self.ants:
                ant.new_path()
                
            self.update_phero() 
            
            cmin = min(ant.lengthPath for ant in self.ants)
            #print("itération : " + str(i) + " sol min : " + str(cmin))             
            minCosts.append(cmin)            
            for ant in self.ants:
                if(ant.lengthPath == cmin):
                    solm = ant.path
                    break
            minSols.append(solm)
            
            iterations.append(i)
           
            #print(solm)
            
            for j in range(0,self.m):
                allCosts[i][j] = self.ants[j].lengthPath
          
        interval = time.time()-start_time 
        """
        affichage des réssultats de la méthode
        """
        
        print("La solution optimale en partant de 0 est le trajet :\n",
              colored(minSols[np.argmin(minCosts)], 'green'))     
        print('La distance est de : ', colored(np.min(minCosts), 'blue'), 'km')
        print('Total time in seconds :', colored(interval, 'red'))
        print('itération de la meilleure solution : ', np.argmin(minCosts))
        print('nombre d\'itérations : ', iterations[-1]+1)
        #évolution du meilleur coût
        plt.plot(minCosts, color='r') 
        plt.suptitle("Coût optimal à chaque itération")
        plt.show()
        
        #évolution du coût moyen 
        means = []
        for i in iterations :
            means.append(np.mean(allCosts[i]))
        plt.scatter(iterations, means, color='b')
        plt.suptitle("Coût moyen des m solutions à chaque itération")
        plt.show()
        
        
       
        
        return (iterations, minSols, minCosts, allCosts, interval)
       
"""
    Classe fourmis
"""
class Ant:
    def __init__(self, problem):
        self.problem = problem
        self.path = []
        self.lengthPath = 0
    
    def reset(self):
        self.path = []
        self.lengthPath = 0
		
    """
    la fourmis depose les pheromones le long de son trajet
    """
    def drop_path_phero(self):        
        for i in range(0, self.problem.n - 1):
            self.problem.drop_phero(self.path[i], self.path[i + 1],
                                    self.lengthPath)
            
        self.problem.drop_phero(self.path[-1], self.path[0], self.lengthPath)
        
    
    """
    ajoute la ville au chemin de la fourmis 
    ainsi que le coût du trajet jusqu'a cette nouvelle ville si il y avait
    déja une ville dans le chemin
    """
    def add_to_path(self, city):
        if (self.path != []):
            self.lengthPath +=  self.problem.costs[self.path[-1],city]        
        self.path.append(city) 
        
        
    def new_path(self):        
        self.add_to_path(0)
        #on part de la première ville 0
                
        for i in range(0, self.problem.n-1):
            
            #print(str(i) + " : " + str(self.path))
            
            #on regarde alors les autres villes possibles depuis la ieme
            nextCities = []
            for j in range(0, self.problem.n):
                if j not in self.path:
                    nextCities.append(j)
            
            #print(str(nextCities))    
            
            #on calcule avec les parametres du modele et en tirant
            #aleatoirement quelle ville choisir
            nextCity = self.problem.next_city_rule(i, nextCities)            
            self.add_to_path(nextCity)

        #on ajoute le dernier coûts de retour au debut
        self.lengthPath += self.problem.costs[self.path[-1], self.path[0]]
        
        self.drop_path_phero()
            
 