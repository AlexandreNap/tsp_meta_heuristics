# -*- coding: utf-8 -*-
from math import exp
import numpy as np
import logging, sys
import random
import matplotlib.pyplot as plt

logger = logging.getLogger()
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger.disabled = False
logger.debug('Hello World !')

class Planete(object):

    def __init__(self):
        """on veut memoriser les differentes positions des planetes en vue d'eventuellement les visualiser"""
        self.Positions = []
        self.Globalbest = []

    def set_agents(self, agentsActuels):
        self.Positions.append([list(i) for i in agentsActuels])

    def set_Globalbest(self, Globalbest):
        self.Globalbest = np.array(Globalbest)

    def get_agents(self):
        """Retourne un historique de tous les agents (planetes) de l'algorithme (sous forme de liste)"""
        return self.Positions

    def get_Globalbest(self):
        """Retourne les K meilleures positions de l'algorithme (sous forme de liste)"""
        return self.Globalbest 
    
class gsaMOD(Planete): # POO : heritage

    def __init__(self, nbPlanetes, matrice_couts ,function, iteration, G0=3, K0=1):

        # nbPlanetes est le nombre de planetes (constant tout le long du programme). Une planete est unr liste de villes = une trajectoire.
        # matrice_couts est ce qui détermine le nombre de villes (ce code est adapté au probleme du voyageur de commerce = TSP).
        # fonction est la fonction qui calcule le cout d'un circuit décrit par une "planete" (qui est une liste de villes = une trajectoire). Les couts peuvent etre penalisés afin de répondre aux contraintes du TSP.
        
        super(gsaMOD, self).__init__()

        self.PosX = np.array([np.random.permutation(len(matrice_couts[0])) for i in range(nbPlanetes)]) # agents actuels.
        # on va repositionner la ville numero 1 en premiere position puis l'ajouter a la fin en tant que derniere position pour etre sur de chercher dans un circuit hamiltonien.
        self.PosX = np.array([np.delete(pos, np.where(pos == 0)) for pos in self.PosX])
        self.PosX = np.array([np.insert(pos, 0, 0) for pos in self.PosX])
        self.PosX = np.array([[val+1 for val in pos] for pos in self.PosX]) # les villes vont de 1 a nbVilles et ne commencent pas a 0.
        self.PosX = np.array([np.append(pos, 1) for pos in self.PosX]) # on rajoute la ville 1 comme ville d'arrivee.
        #self.set_agents(self.PosX) # on ajoute les agents de départ à l'historique.
        print('INITIALISATION : ',self.PosX)
        
        fitness = np.array([function(x, matrice_couts) for x in self.PosX])
        InitActualBest = self.PosX[np.array([function(x, matrice_couts) for x in self.PosX]).argmin()] # planete actuellement la plus proche de la solution
        self.set_Globalbest(InitActualBest)
        print('INITIALISATION fitness = ',str(fitness))
        print('INITIALISATION self.get_Globalbest() = ',str(self.get_Globalbest()))
        print('INITIALISATION function(self.get_Globalbest(), matrice_couts) = ',str(function(self.get_Globalbest(), matrice_couts)))
                
        #K = K0
        #indices_Kbest = np.argpartition(fitness, K) # the first k elements will be the k-smallest elements
        #Kbest = self.PosX[indices_Kbest[:K],:]
        
        # Initialisation du vecteur vitesse de la même taille que les positions.
        velocity = np.array([[0 for k in range(len(matrice_couts[0])+1)] for i in range(nbPlanetes)])

        for t in range(iteration):

            randI = np.random.random((nbPlanetes, len(matrice_couts[0])+1))
            randJ = np.random.random((1, nbPlanetes))[0]

            # attention ici car si on a min(fitness) = max(fitness), on aura des divisions par zéro !!
            m = np.array([(function(x, matrice_couts) - max(fitness)) /
                         (min(fitness) - max(fitness)) for x in self.PosX])
            M = np.array([i / sum(m) for i in m])
            G = G0 / exp(0.01 * t)
            
            a = np.array([sum([randJ[j] * G * M[j] * (self.PosX[j] - self.PosX[i]) / (
                                       np.linalg.norm(self.PosX[i] - self.PosX[j]) + 0.001)
                               for j in range(nbPlanetes)])
                               for i in range(nbPlanetes)])

            velocity = randI * velocity + np.array([a[i] for i in range(nbPlanetes)])
            inv_max_velocity = np.array([1/max(abs(vit)) for vit in velocity])
            velocity = np.array([abs(vit * inv_max_vit) for inv_max_vit,vit in zip(inv_max_velocity,velocity)])

            indvp = [] # indices_villes_a_permuter
            for vit, pos in zip(np.ndenumerate(velocity) , np.ndenumerate(self.PosX)):
                if 0.99<=vit[1]<=1.01 : # ca fonctionne bien # recupere les indices ou la vitesse normalisee est egale a 1.
                    #print(vit[0], vit[1], pos[0], pos[1])
                    indvp.append([pos[0][0],pos[0][1]])
            
            for i in range(len(indvp)) : # on effectue un total de 'len(indvp)' "permutations".
                temp = self.PosX[indvp[i][0],indvp[i][1]]
                while (self.PosX[indvp[i][0],indvp[i][1]] == temp) : # on cherche a changer la ville de cette position, ce serait inefficace de la laisser inchangée. 
                    indice_subit_permutation = np.random.randint(1, len(matrice_couts[0]), 1) # 1 et nombreVilles car on ne veut pas changer la ville de depart ni celle d'arrivee (donc le l'indice 0 et l'indice le plus haut ne subit pas de permutation).
                    self.PosX[indvp[i][0],indvp[i][1]] = self.PosX[indvp[i][0],indice_subit_permutation]
                    self.PosX[indvp[i][0],indice_subit_permutation] = temp

            #self.set_agents(self.PosX) # on ajoute les agents a l'historique.
            ActualBest = self.PosX[np.array([function(x, matrice_couts) for x in self.PosX]).argmin()]
            if (function(ActualBest, matrice_couts) < function(self.get_Globalbest(), matrice_couts)):
                self.set_Globalbest(ActualBest)





def function(x, mat_costs) : # fonction qui calcule la distance totale du trajet !! function prend toujours un seul parametre mais la nature de celui-ci peut changer
    cost = 0
    dernier_indice = len(x)-1
    dejarencontre = np.array([])
    dejarencontre = np.append(dejarencontre, x[dernier_indice])
    for i in range(dernier_indice):
        cout = mat_costs[int(x[i]-1),int(x[i+1]-1)]
        if (i != 0) :
            dejarencontre = np.append(dejarencontre, x[i])
        else :
            if (x[0] != x[dernier_indice]):  # penalite pour forcer a avoir la meme ville au depart et a l'arrivee.
                cost += 10000
                logger.info('on n\'a pas la meme ville en depart et en arrivee = %s.',str([int(x[0]),int(x[dernier_indice])]))
            if (x[dernier_indice] != 1):  # penalite pour forcer a avoir une ville en particulier de depart et d'arrivee.
                cost += 10000
                logger.info('la derniere ville n\'est pas la bonne = %d (au lieu de 1).',int(x[dernier_indice]))
        if (cout!=0): # on emprunte un chemin qui  existe et on change de ville.
            if (i+1 != dernier_indice) and (x[i+1] in dejarencontre): # penalite pour interdire de retourner dans une ville deja visitee.
                cost += 30000
                logger.info('Ville deja traversee = %d.',int(x[i+1]))
            else:
                cost += cout
        else: # penalite pour interdire de rester dans la meme ville.
            cost += 10000
            logger.info('on ne peut pas aller de la ville A à la ville B = %s.',str([int(x[i]),int(x[i+1])]))
    return cost


"""
logger.disabled = True
mat_costs = np.array([[0, 4, 6, 0, 0],
                          [4, 0, 3, 7, 4],
                          [6, 3, 0, 4, 2],
                          [0, 7, 4, 0, 8],
                          [0, 4, 2, 8, 0]])
    
execut1 = gsaMOD(3, mat_costs, function, 500, 3) # 3 planetes, 5 villes, 6 iterations
#logger.disabled = False
print(execut1.get_agents(),' <= agents')
print(execut1.get_Globalbest(),' => cout = ', function(execut1.get_Globalbest(), mat_costs))
"""
