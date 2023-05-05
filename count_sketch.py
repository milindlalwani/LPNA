"""
This file contains functions for generating a count sketch using 4-wise and 2-wise independent hash functions.
"""

import random
import math
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

# A large, random prime number
P = 2564217863

# Random coefficients for 4-wise and 2-wise independent hash functions
RANDOM_COEFFS_4 = np.random.randint(-2**31, 2**31, 4)
RANDOM_COEFFS_2 = np.random.randint(-2**31, 2**31, 2)

def polynomial_hash(coefficients, x):
    """
    Returns the result of a polynomial hash function with the given coefficients and input x.
    """
    return np.polyval(coefficients, x) % P

def two_wise_hash(x):
    """
    Returns the result of a 2-wise independent hash function for the input x.
    """
    return polynomial_hash(RANDOM_COEFFS_2, x) % P

def sign(x):
    """
    Returns +1 or -1 with equal probability.
    """
    i = (polynomial_hash(RANDOM_COEFFS_4, x) % 2)
    return [1., -1.][i]

def generate_count_sketch(k, n):
    """
    Generates a count sketch with k rows and n columns.
    """
    sketch = np.zeros((k, n))
    for col in range(n):
        row = two_wise_hash(col) % k
        sketch[row, col] = sign(col)
    return sketch
