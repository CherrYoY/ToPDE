import numpy as np
import random
from Latin import latin
from DE_current_to_rand_1_bin import DE

def fitness1(pop):
    '''
    -x1 + x2 - 5 <= 0
    x1 ^ 2 + 5 * x2 ^2 - 100 <= 0
    x1 * x2 - 10 <= 0
    - x1 * x2 - 4 < = 0
    '''
    x = pop.copy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
    cv1 = - x[:, 0] + x[:, 1] - 5
    cv2 = x[:, 0] ** 2 + 5 * x[:, 1] ** 2 -100
    cv3 = x[:, 0] * x[:, 1] - 10
    cv4 = -x[:, 0] * x[:, 1] - 4

    cv1_filtered = cv1.clip(min = 0)
    cv2_filtered = cv2.clip(min = 0)
    cv3_filtered = cv3.clip(min = 0)
    cv4_filtered = cv4.clip(min = 0)

    CV = cv1_filtered + cv2_filtered + cv3_filtered + cv4_filtered
    return CV

class ToPDE():
    def __init__(self, popsize, dim, lb, ub):
        self.NP = 5000
        self.numConstrained_point = popsize
        self.dim = dim
        self.cluster_n = 10
        self.NS = int(np.round(self.NP/self.cluster_n))
        self.lb = lb
        self.ub = ub
    
    def normalized(self, geninfo):
        x = geninfo.copy()
        for i in range(len(x)):
            x[i] = (x[i] - self.lb[i])/(self.ub[i] - self.lb[i])
        return x

    def find_replacement_point(self, pop):
        index_list = [] # 按顺序记录第i个欧式距离(第i个近的数据样本)处，距离最小时，在filtered_distance_matrix处的index
        x = pop.copy()
        distance_matrix = []
        if x.ndim == 1:
            x = x.reshape(1, -1)
        normal_x = np.apply_along_axis(self.normalized, 1, x)
        for i in range(len(normal_x)):
            normal_Euc_distance = np.linalg.norm((normal_x - normal_x[i]), axis = 1, ord = 2)
            rank_normal_Euc_distance = normal_Euc_distance[np.argsort(normal_Euc_distance)] # 将该数据样本的欧式距离按照升序排列
            distance_matrix.append(rank_normal_Euc_distance) # 此处记录的是每一个样本到其余样本的normalized Euclidean，并且将其进行了排序，用于对比
        distance_matrix = np.array(distance_matrix).reshape(len(x), len(x))
        filtered_distance_matrix = distance_matrix.copy() # 将要比较的数据样本添加到filtered_distance_matrix

        for i in range(1, distance_matrix.shape[1]): #
            index_replacement = np.where(filtered_distance_matrix[:, i] == np.min(filtered_distance_matrix[:, i]))[0] #返回第i个欧式距离等于最小值的数据样本的序号
            if len(index_replacement) == 1: # 如果返回的序号个数只有一个，那么说明该数据样本对应的第i个欧式距离是最近的，那么这个数据样本点就是要删除的点
                if i == 1: # 如果正好在第二列就没必要反向搜索
                    return index_replacement 
                for j in range(1, i): # 开始利用index_list进行反向搜索index
                    index_replacement = index_list[-j][index_replacement]
                return index_replacement
            else:
                filtered_distance_matrix = filtered_distance_matrix[index_replacement].copy()
                index_list.append(index_replacement)

    def fitness2(self, pop):
        x = pop.copy()
        min_distance_list = []
        if x.ndim == 1:
            x = x.reshape(1, -1)
        normal_x = np.apply_along_axis(self.normalized, 1, x)
        for i in range(len(normal_x)):
            normal_Euc_distance = np.linalg.norm((normal_x - normal_x[i]), axis = 1, ord = 2)
            index_min = np.argsort(normal_Euc_distance)[1] #index为[0]的个体时自己，除自己外的个体最近的点是[1]
            min_distance = normal_Euc_distance[index_min]
            min_distance_list.append(min_distance)
            fitness2 = min(min_distance_list)
        return fitness2

    def fitness1(self, pop):
        '''
        -x1 + x2 - 5 <= 0
        x1 ^ 2 + 5 * x2 ^2 - 100 <= 0
        x1 * x2 - 10 <= 0
        - x1 * x2 - 4 < = 0
        '''
        x = pop.copy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        cv1 = - x[:, 0] + x[:, 1] - 5
        cv2 = x[:, 0] ** 2 + 5 * x[:, 1] ** 2 -100
        cv3 = x[:, 0] * x[:, 1] - 10
        cv4 = -x[:, 0] * x[:, 1] - 4

        cv1_filtered = cv1.clip(min = 0)
        cv2_filtered = cv2.clip(min = 0)
        cv3_filtered = cv3.clip(min = 0)
        cv4_filtered = cv4.clip(min = 0)

        CV = cv1_filtered + cv2_filtered + cv3_filtered + cv4_filtered
        return CV

    def pop_init(self):
        flag = True
        cout = 0
        while(flag == True):
            population = latin(self.NP, self.dim, self.lb, self.ub)
            pop = population.copy()
            tem_pop = []
            feasible_sample_list = []
            r = np.array(self.lb) + np.random.random(self.dim) * (np.array(self.ub)  - np.array(self.lb)) 
            normal_r = self.normalized(r)

            for i in range(self.cluster_n):
                normal_pop = np.apply_along_axis(self.normalized, 1, pop)
                normal_Euc_distance_1 = np.linalg.norm((normal_pop - normal_r), axis = 1, ord = 2)
                z = pop[np.argmin(normal_Euc_distance_1)] 
                normal_z = self.normalized(z)
                normal_Euc_distance_2 = np.linalg.norm((normal_pop - normal_z), axis = 1, ord = 2)
                index_subpop = np.argsort(normal_Euc_distance_2)[:self.NS]
                index_restpop = np.delete(np.arange(0, len(pop)), index_subpop)
                sub_pop = pop[index_subpop]
                pop = pop[index_restpop].copy()
                de = DE(self.lb, self.ub)
                mutation_pop = de.mutation(sub_pop)
                Variation_pop = de.Crossover(sub_pop, mutation_pop)
                individuals = np.concatenate((sub_pop, self.fitness1(sub_pop)[:, np.newaxis]), axis = 1)
                Variation_individuals = np.concatenate((Variation_pop, self.fitness1(Variation_pop)[:, np.newaxis]), axis = 1)
                updated_subind = de.selection(individuals, Variation_individuals)
                tem_pop.append(updated_subind)
            for i in range(self.cluster_n):
                num_zero = np.count_nonzero(tem_pop[i][:, -1] == 0) 
                if num_zero <= self.numConstrained_point/self.cluster_n:
                    cout = cout + 1
                    flag = True
                    break
                else:
                    flag = False
                index_zero = np.where(tem_pop[i][:, -1] == 0)[0]
                index_feasible = random.sample(index_zero.tolist(), int(self.numConstrained_point/self.cluster_n))
                feasible_sample = tem_pop[i][index_feasible]
                feasible_sample_list.append(feasible_sample)
        Constrained_point = np.array(feasible_sample_list).reshape(self.numConstrained_point, self.dim + 1)
        print('Phase 1 is completed')
        Pre_population = Constrained_point[:, :-1].copy()
        FF2 = self.fitness2(Pre_population)
        k = 0
        Flag = True
        while (Flag == True):
            mutation_Prepop = de.mutation(Pre_population)
            Offspring_Prepop = de.Crossover(Pre_population, mutation_Prepop)
            index_Offspring_feasible = np.where(self.fitness1(Offspring_Prepop) == 0)[0]
            Offspring_feasible = Offspring_Prepop[index_Offspring_feasible]   
            duplicated_index_list = []
            for i in range(Pre_population.shape[0]):
                for j in range(Offspring_feasible.shape[0]):
                    gene_sum = 0
                    for l in range(Offspring_feasible.shape[1]):
                        gene_sum = abs(Offspring_feasible[j][l] - Pre_population[i][l]) + gene_sum
                    if gene_sum < 0.1:
                        duplicated_index = j
                        duplicated_index_list.append(duplicated_index)
            Offspring_feasible = np.delete(Offspring_feasible, duplicated_index_list, axis = 0)
            num_Offspring_feasible = Offspring_feasible.shape[0]
            if num_Offspring_feasible == 0:
                # print('num_Offspring_feasible is zero')
                continue            
            for i in range(num_Offspring_feasible):
                Q = Pre_population.copy() 
                refPoint = Offspring_feasible[i, :].reshape(1, -1)
                Offspring_Prepop = np.concatenate((Pre_population, refPoint), axis = 0) 
                index_delet = self.find_replacement_point(Offspring_Prepop)
                temp_pop = np.delete(Offspring_Prepop, index_delet, axis = 0)
                FF2_temp = self.fitness2(temp_pop) 
                FF2_original = self.fitness2(Q)
                if FF2_temp > FF2_original:
                    Pre_population = temp_pop.copy()
                    k = 0
                else:
                    Pre_population = Q.copy()
                    k = k + 1
            if k > 500:
                ob_pop = Pre_population.copy()
                break
        print('Phase 2 is completed')
        return ob_pop


