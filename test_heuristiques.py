# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:50:02 2020

@author: alexa
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import greedy_v as grd
import colonie_de_fourmis as col
import GSA as gsa
import time
from termcolor import colored

"""
cette fonction plot permet d'afficher les points et le chemin les reliant
"""
def plot(points, path: list, titre= ""):
    x = []
    y = []
    for i in range(0,points.shape[0]):
        x.append(points[i][0])
        y.append(points[i][1])        
    plt.figure("solution")    
    plt.plot(x, y, 'co')
    
    for i in range(0,len(points)):                                       
        plt.annotate('(%s)' % i,xy=(x[i],y[i]))
    
    for k in range(1, len(path)):
        i = path[k - 1]
        j = path[k]        
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r',
                  length_includes_head=True)
        
    plt.arrow(x[path[-1]], y[path[-1]], x[path[0]] - x[path[-1]]
              , y[path[0]] - y[path[-1]], color='r', length_includes_head=True)
    plt.suptitle(titre)   
    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.show()
   
"""
"""
         
inf=9999999999

"""
cette fonction creer n villes placees aleatoirement et creer la matrice
de distance entre chaque villes
on donne en entrée une repartition des villes. Par defaut elle est uniforme
sur le carée 100*100
""" 
def distance(x1,y1,x2,y2):
    res = np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return res 

def random_cities(n, repartition = 0):
    cities = np.zeros((n,2))
    
    if (repartition == 1):
        for i in range(n):
            cities[i,0]=rd.uniform(i*100/n,(i+1)*100/n)
            cities[i,1]=rd.uniform(i*100/(4*n),(i+1)*100/n)
    else:
        for i in range(n):
            cities[i,0]=rd.uniform(0,100)
            cities[i,1]=rd.uniform(0,100)    
    
    distances = np.zeros((n, n)) + np.eye(n) * inf      
    
    for i in range(0, n):
        for j in range(i+1, n):
            distances[i][j] = distance(cities[i][0], cities[i][1],
                                       cities[j][0], cities[j][1])
            distances[j][i] = distances[i][j]
            
    return (cities, distances)
 



#plot(random_cities(5)[0], [1, 2, 0, 4, 0, 3, 1])


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

if (__name__ == "__main__"):
    
    """
    ------------------------------------------------------------------------------
                    Test des méthodes
    ------------------------------------------------------------------------------
    """
    n=10
    
    (villes,matrice) = random_cities(n, 0)
    
    #matrice = matr 
     
    start = time.time()
    gsol = gsa.gsaMOD(3, matrice, gsa.function, 500, 3)  
    interval = time.time() - start
    solGs = gsol.get_Globalbest() - 1#car utilise pas la même numérotation
    print('\n\n\n')
    print('_________________________________________________________________')
    print('\n-----------------------------------------------------------------')
    print('méthode GSA : ')
    print('-----------------------------------------------------------------\n')
    
    print("La solution optimale en partant de 0 est le trajet :")#0 au lieu de 1 
    print(colored(solGs, 'green'))
    print("La distance est de :", colored(gsa.function(gsol.get_Globalbest(),
                                               matrice), 'blue'), "km")
    print ('Total time in seconds :', colored(interval, 'red'))
    print('nombre d\'itérations : ', 500)
    
    
    
    print('_________________________________________________________________')
        
    print('\n-----------------------------------------------------------------')
    print('méthode colonie de fourmis : ')
    print('-----------------------------------------------------------------\n')
    pb = col.Antcolonie(matrice,30,10,10,60,0.3,500)    
    
    (iterations, minSols, minCosts, allCosts, dt) = pb.start()       
    
    #plot(villes, minSols[0])
    #plot(villes, minSols[-1])
    solF = minSols[np.argmin(minCosts)]
    
    
    
    print('_________________________________________________________________')
    
    print('\n-----------------------------------------------------------------')
    print('méthode gloutonne : ')
    print('-----------------------------------------------------------------\n')
    solGr = grd.greedy_ville(matrice)[0]
    
    
    
    print('_________________________________________________________________')


    titre = "GSA \nMeilleur trajet trouvé"
    plot(villes, solGs, titre)
    
    titre = "Colonie de fourmis \nMeilleur trajet trouvé de toutes les itérations"
    plot(villes, solF, titre)
    #titre = "Colonie de fourmis \nPire des meilleurs trajets trouvés à chaque itération"
    #plot(villes, minSols[np.argmax(minCosts)], titre)
    
    titre = "Glouton \nMeilleur trajet trouvé"
    plot(villes, solGr, titre)



