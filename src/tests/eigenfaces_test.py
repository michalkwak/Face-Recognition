import numpy as np

from eigenfaces.mean_face import compute_mean_face, mean_subtract
from eigenfaces.covariance_matrix import compute_covariance_matrix
from eigenfaces.eigenfaces import compute_eigenfaces, project_to_eigenspace


def test_eigenfaces_have_correct_shape():
    images = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 5.0, 5.0]
    ])
    mean_face = compute_mean_face(images)
    mean_subtracted = mean_subtract(images, mean_face)
    l_matrix = compute_covariance_matrix(mean_subtracted)

    eigenfaces, eigenvalues = compute_eigenfaces(mean_subtracted, l_matrix, num_components=2)

    assert eigenfaces.shape == (4, 2)
    assert eigenvalues.shape == (2,)


def test_eigenfaces_are_unit_length():
    images = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 5.0, 5.0]
    ])
    mean_face = compute_mean_face(images)
    mean_subtracted = mean_subtract(images, mean_face)
    l_matrix = compute_covariance_matrix(mean_subtracted)

    eigenfaces, _ = compute_eigenfaces(mean_subtracted, l_matrix)

    norms = np.linalg.norm(eigenfaces, axis=0)
    np.testing.assert_allclose(norms, np.ones(norms.shape), atol=1e-8)


def test_projecting_mean_face_gives_zero_weights():
    mean_face = np.array([1.0, 2.0, 3.0])
    eigenfaces = np.identity(3)

    weights = project_to_eigenspace(mean_face, mean_face, eigenfaces)

    np.testing.assert_allclose(weights, np.zeros(3), atol=1e-10)