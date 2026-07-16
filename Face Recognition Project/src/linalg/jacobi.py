import numpy as np

def jacobi_eigenvalue(A, tol=1e-14, max_iter=1000):
    '''
    Diagonalizes a symmetric matrix using the Jacobian method

    At each step, finds the largest off-diagonal element and applies
    a rotation that zeroes it out, repeating until all off-diagonal
    elements are below "tol".

    Inputs: 
        symmetric matrix A
        the largest allowed off-diagonal element threshold
        maximum iterations

    Returns:
        eigenvalues: 1D array of eigenvalues
        eigenvectors: 2D array where each column eigenvectors[:, i] is the eigenvector for eigenvalues[i]
        number of rotations actually performed
    '''
    pass

def max_off_diagonal(A, n):
    '''Returns the largest off-diagonal element of A[p, q] where p < q'''
    max_val = 0.0
    p, q = 0, 1

    for i in range(n-1):
        for j in range(i+1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j
    return max_val, p, q

def rotate(a, v, n, q, p):
    '''Rotate A in place to eliminate A[p, q]'''
    pass

def rotate_pair(a, row1, col1, row2, col2, s, tau):
    pass
