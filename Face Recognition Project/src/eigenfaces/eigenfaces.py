"""
Extracts eigenfaces from a set of training images

1. Mean-subtract the training images (mean_face.py)
2. Build the reduced m x m matrix L = A_mp * A_mp^T (covariance.py)
3. Eigendecompose L (Jacobi algorithm) to get eigenvectors v_i
4. Recover the p-pixel eigenfaces: u_i = A_mp^T * v_i
5. Keep the top eigenfaces by eigenvalue
"""

import numpy as np

from src.linalg.jacobi import jacobi_eigenvalue


def compute_eigenfaces(mean_subtracted, L, num_components=None):
    """
    Compute eigenfaces from mean-subtracted training images and their reduced covariance matrix L

    Inputs:
        mean_subtracted: 2D matrix of shape (m, p), mean-subtracted training images, one image per row
        L: 2D matrix of shape (m, m) (the output of covariance.compute_covariance_matrix)
        num_components: how many eigenfaces to keep, ordered by eigenvalue, descending

    Returns:
        eigenfaces: 2D matrix of shape (p, k), one eigenface per column, each normalized to unit length
        eigenvalues: 1D vector of shape k, the corresponding eigenvalues, descending
    """

    m = mean_subtracted.shape[0]
    if num_components is None:
        num_components = m
    if not (1 <= num_components <= m):
        raise ValueError(f"num_components must be between 1 and {m}")

    eigenvalues, v, _ = jacobi_eigenvalue(L)

    # sort by eigenvalue, descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    v = v[:, order]

    # recover the p-pixel eigenfaces u_i = A_mp^T * v_i
    eigenfaces = mean_subtracted.T @ v

    # normalize each eigenface to unit length
    norms = np.linalg.norm(eigenfaces, axis=0)
    norms[norms == 0] = 1.0  # if eigenvector is 0
    eigenfaces = eigenfaces / norms

    return eigenfaces[:, :num_components], eigenvalues[:num_components]


def project_to_eigenspace(image, mean_face, eigenfaces):
    """
    Project a single image into eigenface space, producing its weight vector

    Inputs:
        image: 1D vector of shape p (not mean-subtracted) image
        mean_face: 1D vector of shape p
        eigenfaces: 2D matrix of shape (p, k)

    Returns:
        1D vector of shape k (the weight vector)
    """
    return eigenfaces.T @ (image - mean_face)