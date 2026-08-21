"""
Computes the mean face and mean-subtracted training set
"""

import numpy as np


def compute_mean_face(images):
    """
    Computes the mean face of a set of training images

    Input:
        2D array of shape (m, p), where m is the number of
        training images and p is the number of pixels per
        image (so each row is one flattened image)

    Return:
        1D array of shape (p,): the mean face
    """
    if images.ndim != 2:
        raise ValueError("images must be a 2D array of shape (m, p)")
    if images.shape[0] == 0:
        raise ValueError("images must contain at least one training image")

    return np.mean(images, axis=0)


def mean_subtract(images, mean_face):
    """
    Subtracts the mean face from every image in the training set

    Inputs:
        images: 2D array of shape (m, p), the training images
        mean_face: 1D array of shape (p,) (output of compute_mean_face(images))

    Return:
        2D array of shape (m, p): the mean-subtracted images
        stacked as rows (one mean-subtracted face per row)
    """
    if images.shape[1] != mean_face.shape[0]:
        raise ValueError(
            f"image dimension {images.shape[1]} does not match "
            f"mean face dimension {mean_face.shape[0]}"
        )

    return images - mean_face