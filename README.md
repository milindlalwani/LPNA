# Intro
In this report, we analyze the performance of two different embedding methods: Gaussian embedding and CountSketch embedding. We compare the accuracy of these methods in solving an underdetermined linear system, where the number of equations is less than the number of unknowns. Specifically, we consider the problem of finding the solution to the equation Ax=b, where A is a randomly generated m-by-n matrix, b is a random vector of length m, and x is the unknown vector of length n. We compare the relative error of the solution obtained using the embeddings to that obtained using the exact solution. We investigate the effect of the sketch size on the relative error for both embeddings. We also compare the performance of the embeddings with different random seeds.

# Methods
We implement Gaussian embedding and CountSketch embedding in Python using NumPy, SciPy, and Matplotlib libraries. For Gaussian embedding, we generate an s-by-m matrix of i.i.d. normal random variables, where each element in the matrix has mean 0 and variance 1/s. For CountSketch embedding, we use the algorithm described in the paper "Finding Frequent Items in Data Streams" by Muthukrishnan (2005) to generate a k-by-m sketch matrix S. We use the following hash functions:

kwise_hash(k, x) returns the hash value of x using the coefficients in k. We use this hash function to compute h(x,k) as kwise_hash(k_2, x) % k, which maps an index in the range [0,m-1] to an index in the range [0,k-1]. We also use kwise_hash(k_4, x) % 2 to compute the sign of an element in the sketch matrix.
To solve the underdetermined linear system, we use the lstsq function from the NumPy linalg library to find the least-squares solution to the equation Ax=b. We then calculate the relative error of the solution obtained using the embeddings to that obtained using the exact solution as ```(||Ax-b|| - ||Ax_ref - b||) / (||Ax_ref - b||).```

# Results
We first compare the performance of Gaussian embedding and CountSketch embedding with respect to the sketch size. We use the same values of m=3000 and n=100 for both embeddings and vary the sketch size s over the range [550, 2000]. We calculate the relative error of the solution for each value of s and plot the results on a log-log scale. The results are shown in Figure 1.

Figure 1: Relative error vs sketch size for Gaussian and CountSketch embeddings

We observe that both embeddings produce accurate solutions for small sketch sizes. However, as the sketch size increases, the relative error increases for both embeddings. Gaussian embedding produces better results than CountSketch embedding for small sketch sizes, but as the sketch size increases beyond 1000, CountSketch embedding outperforms Gaussian embedding.
