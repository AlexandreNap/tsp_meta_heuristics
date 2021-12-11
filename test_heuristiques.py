# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:50:02 2020

@author: alexandre
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import greedy_v as grd
import colonie_de_fourmis as col
import GSA as gsa
import time
from termcolor import colored


def plot(points, path: list, titre=""):
    x = []
    y = []
    for i in range(0, points.shape[0]):
        x.append(points[i][0])
        y.append(points[i][1])        
    plt.figure("solution", figsize=(16, 9))
    plt.plot(x, y, 'co')
    
    for i in range(0, len(points)):
        plt.annotate('(%s)' % i, xy=(x[i], y[i]))
    
    for k in range(1, len(path)):
        i = path[k - 1]
        j = path[k]        
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r',
                  length_includes_head=True)
        
    plt.arrow(x[path[-1]], y[path[-1]], x[path[0]] - x[path[-1]],
              y[path[0]] - y[path[-1]], color='r', length_includes_head=True)
    plt.suptitle(titre)   
    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.show()

         
inf = 9999999999


def distance(x1, y1, x2, y2):
    res = np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return res 


def random_cities(n, repartition=0):
    cities = np.zeros((n, 2))
    if repartition == 1:
        for i in range(n):
            cities[i, 0] = rd.uniform(i * 100/n, (i + 1) * 100/n)
            cities[i, 1] = rd.uniform(i * 100/(4 * n), (i + 1) * 100/n)
    else:
        for i in range(n):
            cities[i, 0] = rd.uniform(0, 100)
            cities[i, 1] = rd.uniform(0, 100)
    distances = np.zeros((n, n)) + np.eye(n) * inf
    for i in range(0, n):
        for j in range(i+1, n):
            distances[i][j] = distance(cities[i][0], cities[i][1],
                                       cities[j][0], cities[j][1])
            distances[j][i] = distances[i][j]
    return cities, distances
 

"""
un exemple de matrice où on connait la sol optimale
"""       
"""
matr = np.array([[inf, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, inf, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, inf, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, inf, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, inf, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, inf, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, inf, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, inf, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, inf, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 1, inf]])

for i in range(0, 10):
    for j in range(i + 1, 10):
        if (matr[i, j]==0):
            matr[i, j] = rd.uniform(1,8)
            matr[j, i] = matr[i, j]
    """       
"""

mat =np.array([[0, 207, 843, 831, 785, 545, 853, 891, 749, 706, 615, 863, 312],
      [200, 0, 644, 664, 585, 485, 654, 698, 556, 646, 422, 803, 245],
      [842, 646, 0, 1080, 593, 1122, 499, 1005, 972, 1280, 809, 1437, 882],
      [831, 661, 1079, 0, 576, 296, 704, 107, 112, 306, 272, 463, 520],
      [783, 584, 591, 575, 0, 749, 136, 559, 466, 775, 411, 932, 678],
      [544, 485, 1122, 295, 750, 0, 856, 545, 303, 169, 398, 326, 243],
      [853, 654, 499, 704, 136, 856, 0, 688, 594, 903, 490, 1060, 785],
      [891, 695, 1089, 108, 544, 414, 672, 0, 146, 424, 308, 581, 648],
      [749, 553, 971, 111, 467, 304, 595, 145, 0, 314, 166, 471, 538],
      [707, 648, 1282, 308, 778, 173, 907, 425, 316, 0, 476, 199, 406],
      [614, 418, 809, 273, 412, 397, 518, 307, 165, 472, 0, 629, 422],
      [862, 803, 1437, 463, 933, 327, 1061, 580, 471, 198, 630, 0, 561],
      [308, 246, 883, 530, 681, 243, 788, 647, 538, 404, 424, 561, 0]])

mat= mat + np.eye(13)*inf
"""

if __name__ == "__main__":
    n = 10
    
    cities, mat = random_cities(n, 0)
    # mat = matr
     
    start = time.time()
    gsol = gsa.gsaMOD(3, mat, gsa.function, 500, 3)
    interval = time.time() - start
    solGs = gsol.get_Globalbest() - 1

    print('\n\n\n')
    print('\n' + 50*'-')
    print('méthode GSA : ')
    print(50*'-')
    
    print("La solution optimale en partant de 0 est le trajet :")
    print(colored(solGs, 'green'))
    print("La distance est de :", colored(gsa.function(gsol.get_Globalbest(),
                                                       mat), 'blue'), "km")
    print('Total time in seconds :', colored(interval, 'red'))
    print('nombre d\'itérations : ', 500)

    print('\n' + 50*'-')
    print('méthode colonie de fourmis : ')
    print(50*'-')
    pb = col.Antcolonie(mat, 30, 10, 10, 60, 0.3, 500)
    iterations, minSols, minCosts, allCosts, dt = pb.start()
    # plot(cities, minSols[0])
    # plot(cities, minSols[-1])
    solF = minSols[np.argmin(minCosts)]
    print(50 * '_')
    
    print('\n' + 50*'-')
    print('méthode gloutonne : ')
    print(50*'-')
    solGr = grd.greedy_ville(mat)[0]
    title = "GSA \nMeilleur trajet trouvé"
    plot(cities, solGs, title)
    
    title = "Colonie de fourmis \nMeilleur trajet trouvé de toutes les itérations"
    plot(cities, solF, title)
    
    title = "Glouton \nMeilleur trajet trouvé"
    plot(cities, solGr, title)
