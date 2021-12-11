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
        
        self.phero = np.ones((self.n, self.n))  # np.zeros((self.n, self.n))
        self.delta_phero = np.zeros((self.n, self.n))
        
        self.ants = []

        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.p = p
        self.m = m
        
        for i in range(0, self.m):
            self.ants.append(Ant(self))
        
        self.tmax = tmax
        
    def drop_phero(self, city_a, city_b, length):
        self.delta_phero[city_a, city_b] += self.q / length

    def update_phero(self):
        self.phero = (1 - self.p) * self.phero + self.delta_phero
        self.delta_phero = np.zeros((self.n, self.n))

    def choose_next_city(self, i, possible_cities):
        probas = []
        p = 0.0
        total = 0
        # on construit un vecteur de répartition de probabilité de choisir
        # chaque ville
        # p[i]= P(uniform(0,1)<p[i])
        # si p[i] < uniform(0,1) < p[i+1] alors on choisira la i eme ville
        for j in possible_cities:
            p += self.phero[i, j] ** self.alpha * self.nu[i, j] ** self.beta
            total += p
            probas.append(p)
                
        # on normalise le vecteur pour que tout soit entre 0 et 1
        if p == 0:
            probas = np.arange(1, len(possible_cities) + 1) / len(possible_cities)
        else:
            probas = probas / p

        rand = rd.uniform(0, 1)
        j = 0
        while j < len(possible_cities) and rand > probas[j]:
            j += 1
        return possible_cities[j]
    
    def reset_ants(self):
        for ant in self.ants:
            ant.reset()

    def start(self):
        min_costs = []
        all_costs = np.zeros((self.tmax, self.m))
        iterations = []
        min_sols = []
        start_time = time.time()
        
        for i in range(0, self.tmax): 
            self.reset_ants()
            for ant in self.ants:
                ant.new_path()
            self.update_phero()
            best_cost = min(ant.path_length for ant in self.ants)
            min_costs.append(best_cost)
            for ant in self.ants:
                if(ant.path_length == best_cost):
                    best_sol = ant.path
                    break
            min_sols.append(best_sol)
            iterations.append(i)
            for j in range(0, self.m):
                all_costs[i][j] = self.ants[j].path_length

        interval = time.time() - start_time

        print("La solution optimale en partant de 0 est le trajet :\n",
              colored(min_sols[np.argmin(min_costs)], 'green'))
        print('La distance est de : ', colored(np.min(min_costs), 'blue'), 'km')
        print('Total time in seconds :', colored(interval, 'red'))
        print('itération de la meilleure solution : ', np.argmin(min_costs))
        print('nombre d\'itérations : ', iterations[-1] + 1)

        plt.plot(min_costs, color='r')
        plt.suptitle("Coût optimal à chaque itération")
        plt.show()

        means = []
        for i in iterations:
            means.append(np.mean(all_costs[i]))
        plt.scatter(iterations, means, color='b')
        plt.suptitle("Coût moyen des m solutions à chaque itération")
        plt.show()

        return iterations, min_sols, min_costs, all_costs, interval
       

class Ant:
    def __init__(self, problem):
        self.problem = problem
        self.path = []
        self.path_length = 0
    
    def reset(self):
        self.path = []
        self.path_length = 0

    def drop_path_phero(self):        
        for i in range(0, self.problem.n - 1):
            self.problem.drop_phero(self.path[i], self.path[i + 1],
                                    self.path_length)
        self.problem.drop_phero(self.path[-1], self.path[0], self.path_length)

    def add_to_path(self, city):
        if (self.path != []):
            self.path_length += self.problem.costs[self.path[-1], city]
        self.path.append(city) 

    def new_path(self):        
        self.add_to_path(0)
        for i in range(0, self.problem.n-1):
            possible_cities = []
            for j in range(0, self.problem.n):
                if j not in self.path:
                    possible_cities.append(j)

            next_city = self.problem.choose_next_city(i, possible_cities)
            self.add_to_path(next_city)
        self.path_length += self.problem.costs[self.path[-1], self.path[0]]
        self.drop_path_phero()
