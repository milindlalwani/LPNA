import numpy as np


def lstsq(A, b):
    """
    Solve the least squares problem

    inputs
    ------
    A: an input matrix with size m x n 
    b: an input vector with size m
    outputs:
    a vector x such that x = argmin_x||Ax-b||_2.
    """
    m, n = A.shape
    assert m >= n
    # TODO: implementation here
    raise NotImplementedError


def lstsq_residual(A, x, b):
    """
    return the residual norm, ||Ax-b||_2.
    """
    # TODO: implementation here
    raise NotImplementedError


def sketch_lstsq(S, A, b):
    """
    Solve the sketched least squares problem

    inputs
    ------
    S: an embedding matrix
    A: an input matrix with size m x n 
    b: an input vector with size m
    outputs:
    a vector x such that x = argmin_x||SAx-Sb||_2.
    """
    # TODO: implementation here,
    # You can form A2 = SA and b2 = Sb, and call lstsq(A2, b2).
    raise NotImplementedError


def embedding_builder(s, m):
    """
    build an s by m matrix of i.i.d. Normal random variables. 
    Each element in the matrix has mean 0 and variance 1/s.
    """
    raise NotImplementedError


m = 300
n = 10
sketch_sizes = [50, 100, 150, 200, 250, 300]
A = np.random.rand(m,n)
b = np.random.rand(m)
x = lstsq(A, b)
error = lstsq_residual(A, x, b)
print("error is", error)
for s in sketch_sizes:
    S = embedding_builder(s, m)
    x = sketch_lstsq(S, A, b)
    error = lstsq_residual(A, x, b)
    print("sketching error with sketch size ", s, " is", error)
