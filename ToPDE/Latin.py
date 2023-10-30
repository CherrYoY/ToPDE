import numpy as np
def latin(N, D, lower_bound, upper_bound):
    '''
    N - The size of the sample data
    D - No.of Decision Variables
    '''
    d = 1.0 / N #d 声明interval间隔距离
    result = np.empty([N, D])# 声明样本存放空间
    temp = np.empty([N])
    for i in range(D): #在第一个维度上
        for j in range(N): #
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size=1)[0] # [0]的作用是选取narry中的数字
        np.random.shuffle(temp) #随机排列temp中的数值
        for j in range(N):
            result[j, i] = temp[j]
        if np.any(lower_bound[i] > upper_bound[i]):
            print('Range error')
            return None
        np.add(np.multiply(result[:, i], (upper_bound[i] - lower_bound[i]), out=result[:, i]), lower_bound[i], out=result[:, i])
    return result