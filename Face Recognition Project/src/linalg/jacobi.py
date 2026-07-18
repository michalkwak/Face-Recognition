import numpy as np
from math import sqrt

def jacobi_eigenvalue(A, tol, max_iter):
    '''
    Diagonalizes a symmetric matrix using the Jacobian method, that is, each iteration, 
    it finds the largest off-diagonal element and applies
    a rotation that zeroes it out. Repeats until all off-diagonal
    elements are less than "tol".

    Inputs: 
        symmetric matrix A
        the largest allowed off-diagonal element threshold
        maximum iterations

    Returns:
        eigenvalues: vector of eigenvalues
        eigenvectors: matrix where each column eigenvectors[:, i] is the eigenvector for eigenvalues[i]
        number of iterations performed
    '''
    pass

def max_off_diagonal(A, n):
    '''Returns the largest off-diagonal element of A[p, q] where k < l'''
    max_val = 0.0
    k, l = 0, 1

    for i in range(n-1):
        for j in range(i+1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                k, l = i, j
    return max_val, k, l

def rotate(A, p, n, k, l):
    '''Rotate A in place to eliminate A[p, q]. Accumulate the rotations to p '''
    diff = A[l, l] - A[k, k]

    # if a position is very small, we can approximate it
    if abs(A[k, l]) < abs(diff) * 1.0e-10:
        t = A[k, l] / diff
    else:
        theta = diff / (2.0 * A[k, l])
        t = 1.0 / (abs(theta) + sqrt(theta ** 2 + 1.0))

        if theta < 0.0:
            t = -t

    c = 1.0 / sqrt(t ** 2 + 1.0)
    s = t * c
    tau = s / (1.0 + c)

    a_kl = A[k, l]
    A[k, l] = 0.0
    A[k, k] -= t * a_kl
    A[l, l] += t * a_kl

    # instead of writing all for loops manually, I will inplement a function rotate_pair that makes the code cleaner and shorter

def rotate_pair(a, row1, col1, row2, col2, s, tau):
    pass
