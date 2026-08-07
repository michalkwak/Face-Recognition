"""
Nearest-neighbor classification in eigenface space
"""

import numpy as np

from src.eigenfaces.eigenfaces import project_to_eigenspace


def project_all(images, mean_face, eigenfaces):
    """Project every image in a training set into eigenface space"""
    weights = []
    for image in images:
        weights.append(project_to_eigenspace(image, mean_face, eigenfaces))
    return np.array(weights)


def predict(image, mean_face, eigenfaces, training_weights, training_labels):
    """
    Classify a single image by finding its nearest neighbor among the projected training faces

    Returns the predicted label and the distance to the closest matching image
    """
    query_weights = project_to_eigenspace(image, mean_face, eigenfaces)

    best_label = None
    best_distance = None

    for weights, label in zip(training_weights, training_labels):
        distance = np.linalg.norm(weights - query_weights)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = label

    return best_label, best_distance