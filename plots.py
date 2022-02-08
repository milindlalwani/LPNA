import random
import math
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt

p = 2564217863 # A large, random prime number
k_4 = np.random.randint(-2**31, 2**31, 4)
k_2 = np.random.randint(-2**31, 2**31, 2)

def kwise_hash(k, x):
    return np.polyval(k, x) % p

def h(x, k):
    return kwise_hash(k_2, x) % k

def s(x):
    i = (kwise_hash(k_4, x) % 2)
    return [1., -1.][i]

def generate_count_sketch(k, n):
    S = np.zeros((k, n))
    for col in range(n):
        row = h(col, k)
        S[row, col] = s(col)
    return S
    
def embedding_matrix(A, m, n): 
    random_vector = [random.randrange(-math.inf, math.inf) for i in range(n)]
    random_embedding_matrix = [m][n]
    for i in range(m): 
        for j in range(n): 
            random_embedding_matrix[i][j] = random.randrange(-math.inf, math.inf)


sketch_sizes = [50, 100, 150, 200, 250, 300]
A = np.random.rand(3, 6)
b = np.random.rand(3)

list_relative_error = []
print("error is", error)
for s in sketch_sizes: 
    n = np.random.rand(5, 10)
    k = np.random.rand(3)
    S = generate_count_sketch(n, k)
    x = linalg.lstsq(S, A, b)
    x_ref = linalg.lstsq(A, b)
    relative_error = numpy.linalg.norm(x_ref-x*)/numpy.linalg.norm(x_ref)
    list_relative_error.append(relative_error)
plt.plot(sketch_sizes, list_relative_error)




