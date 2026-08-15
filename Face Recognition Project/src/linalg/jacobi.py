'''Jacobi eigendecomposition algorithm'''

from math import sqrt
import numpy as np

def jacobi_eigenvalue(a, tol=1e-10, max_sweeps=100):
    """
    Diagonalizes a symmetric matrix using cyclic Jacobi sweeps
    Every off-diagonal pair (p, q) is rotated once per sweep 
    (before I used the classic method, where it would target 
    the single largest element each iteration, making it run for way too long)
    """
    a = a.copy().astype(float)
    n = a.shape[0]
    v = np.identity(n)

    for sweep in range(max_sweeps):
        off_diagonal_sum = sum(a[i, j] ** 2 for i in range(n - 1) for j in range(i + 1, n))

        if off_diagonal_sum < tol:
            return np.diag(a), v, sweep

        for k in range(n - 1):
            for l in range(k + 1, n):
                if abs(a[k, l]) > 1e-14:  # skip pairs close to zero
                    rotate(a, v, n, k, l)

    print("The matrix did not converge within max_sweeps")
    return np.diag(a), v, max_sweeps

def max_off_diagonal(a, n):
    '''Returns the largest off-diagonal element of a[p, q] where p < q'''
    max_val = 0.0
    k, l = 0, 1

    for i in range(n-1):
        for j in range(i+1, n):
            if abs(a[i, j]) > max_val:
                max_val = abs(a[i, j])
                k, l = i, j
    return max_val, k, l

def rotate(a, v, n, k, l):
    '''Rotate 'a' in place to eliminate a[p, q] (only operates the upper triangle, since matrix is symmetric). 
    Accumulate all the rotations into v, which becomes the eigenvectors'''
    diff = a[l, l] - a[k, k]

    # if a[k, l] is very small, we cam approximate it
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

    # update the rest of row/column k and l
    for i in range(k):
        rotate_pair(a, i, k, i, l, s, tau)
    for i in range(k + 1, l):
        rotate_pair(a, k, i, i, l, s, tau)
    for i in range(l + 1, n):
        rotate_pair(a, k, i, l, i, s, tau)

    # accumulate the rotation into v
    for i in range(n):
        v_ik, v_il = v[i, k], v[i, l]
        v[i, k] = v_ik - s * (v_il + tau * v_ik)
        v[i, l] = v_il + s * (v_ik - tau * v_il)

def rotate_pair(a, row1, col1, row2, col2, s, tau):
    """Apply one (i, j) update during a rotation"""
    a1, a2 = a[row1, col1], a[row2, col2]
    a[row1, col1] = a1 - s * (a2 + tau * a1)
    a[row2, col2] = a2 + s * (a1 - tau * a2)
