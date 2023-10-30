# DE algorithm with Binomial Crossover
# Reference: Differential Evolution: A Survey of the State-of-the-Art

import numpy as np
import random
import matplotlib.pyplot as plt
import Test_Functions
from Latin import latin

class DE():
    def __init__(self, lb, ub):
        self.CXPB = 1
        self.F = 0.9
        self.NGEN = 100
        self.popsize = 100
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

    def pop_init(self):
        x = latin(self.popsize, self.dim, self.lb, self.ub)
        return x

    def mutation(self, pop):
        '''
        donor vector: V
        V = X_r1 + F * (X_r2 - X_r3)
        '''
        population = pop.copy()
        mutation_pop = np.zeros([population.shape[0], population.shape[1]])
        # 对第i个个体，生成donor vector
        for i in range(len(pop)):
            ## Mutation
            x_i = population[i].copy()
            r1, r2, r3 = random.sample(
                np.delete(np.arange(0, len(pop)), i).tolist(), 3)
            X_r1 = population[r1, :]
            X_r2 = population[r2, :]
            X_r3 = population[r3, :]
            V = x_i + random.random() * (X_r1 - x_i) + self.F * (X_r2 - X_r3)
            mutation_pop[i, :] = V
        return mutation_pop

    def Crossover(self, pop, mutation_pop):
        '''
        binomial crossover
        '''
        Variation_pop = np.zeros([pop.shape[0], pop.shape[1]])
        for i in range(len(pop)):
            geninfo1 = pop[i, :].copy()
            geninfo2 = mutation_pop[i, :]
            newoff = geninfo1
            j_rand = random.randint(0, len(geninfo1) - 1)
            for j in range(len(geninfo1)):
                if random.random() <= self.CXPB or j_rand == j:
                    newoff[j] = geninfo2[j]
            newoff = np.clip(newoff, self.lb, self.ub)
            Variation_pop[i, :] = newoff
        return Variation_pop

    def selection(self, individuals, Variation_individuals):
        offspring = np.zeros([individuals.shape[0], individuals.shape[1]])
        for i in range(len(individuals)):
            fitness1 = individuals[i, -1]
            fitness2 = Variation_individuals[i, -1]
            if fitness1 <= fitness2:
                offspring[i, :] = individuals[i, :]
            else:
                offspring[i, :] = Variation_individuals[i, :]
        return offspring

    def selectBest(self, offspring):
        index_min = np.argmin(offspring[:, -1])
        ind_best = offspring[index_min]
        return ind_best

if __name__ == "__main__":
    lb = [-5.12] * 10
    ub = [5.12] * 10
    minimum = 10e6
    f = Test_Functions.ellipsoid

    DE = DE(lb, ub)

    fitness_list = []
    pop = latin(DE.popsize, DE.dim, lb, ub)
    for i in range(DE.NGEN):
        mutation_pop = DE.mutation(pop)
        Variation_pop = DE.Crossover(pop, mutation_pop)
        individuals = np.concatenate((pop, f(pop)[:, np.newaxis]), axis = 1)
        Variation_individuals = np.concatenate((Variation_pop, f(Variation_pop)[:, np.newaxis]), axis = 1)
        offspring = DE.selection(individuals, Variation_individuals)
        Best_individuals = DE.selectBest(offspring)
        pop = offspring[:, :-1]
        if Best_individuals[-1] <= minimum:
            minimum = Best_individuals[-1]
            best_ind = Best_individuals[:-1]
        fitness_list.append(minimum)
    print('The candidate is ', np.around(best_ind, 3))
    print('The minimum is %.3f' % minimum)

    plt.figure(figsize = (12, 8), dpi = 800)
    plt.plot(np.arange(0, DE.NGEN), fitness_list)

