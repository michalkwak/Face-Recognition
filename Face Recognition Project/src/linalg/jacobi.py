import numpy as np
from math import sqrt

def jacobi_eigenvalue(A, tol=1e-14, max_iter=1000):
    '''
    Diagonalizes a symmetric matrix using the Jacobian method, that is, st each step, 
    finds the largest off-diagonal element and applies
    a rotation that zeroes it out, repeating until all off-diagonal
    elements are below "tol".

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
    '''Returns the largest off-diagonal element of A[p, q] where p < q'''
    max_val = 0.0
    k, l = 0, 1

    for i in range(n-1):
        for j in range(i+1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                k, l = i, j
    return max_val, k, l

def rotate(a, v, n, k, l):
    '''Rotate A in place to eliminate A[p, q]'''
    diff = a[l, l] - a[k, k]

    if abs(a[k, l]) < abs(diff) * 1.0e-10:
        t = a[k, l] / diff
    else:
        phi = diff / (2.0 * a[k, l])
        t = 1.0 / (abs(phi) + sqrt(phi ** 2 + 1.0))
        if phi < 0.0:
            t = -t

    c = 1.0 / sqrt(t ** 2 + 1.0)
    s = t * c
    tau = s / (1.0 + c)

    a_kl = a[k, l]
    a[k, l] = 0.0
    a[k, k] -= t * a_kl
    a[l, l] += t * a_kl

def rotate_pair(a, row1, col1, row2, col2, s, tau):
    pass
