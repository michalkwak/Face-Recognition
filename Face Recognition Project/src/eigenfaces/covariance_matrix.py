"""
Builds the reduced covariance matrix used to extract eigenfaces

Instead of computing the full p*p covariance matrix C = A_mp^T * A_mp 
(where m is the number of training images and p is x * y), 
compute the much smaller m * m matrix L = A_mp * A_mp^T
"""

def compute_covariance_matrix(mean_subtracted):
    """
    compute the reduced m * m matrix L = A_mp * A_mp^T

    Inputs:
        mean_subtracted: 2D matrix of shape (m, p)

    Returns:
        2D matrix of shape (m, m): the reduced covariance matrix L
    """
    if mean_subtracted.ndim != 2:
        raise ValueError("mean_subtracted must be a 2D array of shape (m, p)")
    if mean_subtracted.shape[0] == 0:
        raise ValueError("mean_subtracted must contain at least one image")

    return mean_subtracted @ mean_subtracted.T


def compute_full_covariance_matrix(mean_subtracted):
    """
    Compute the full p x p pixel-space covariance matrix C = A_mp^T * A_mp 
    (for testing purposes)

    Inputs:
        mean_subtracted: 2D matrix of shape (m, p)

    Returns:
        2D matrix of shape (p, p)
    """
    if mean_subtracted.ndim != 2:
        raise ValueError("mean_subtracted must be a 2D array of shape (m, p)")

    return mean_subtracted.T @ mean_subtracted