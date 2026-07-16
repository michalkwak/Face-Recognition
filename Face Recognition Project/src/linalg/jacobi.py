import numpy as np

def jacobi_eigenvalue(A, n, max_iter):
    pass

def off_diagonal_norm(A):
    """Returns the sum of squares of all off-diagonal elements of A."""
    n = A.shape[0]
    off_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                off_sum += A[i, j] ** 2
    return off_sum
