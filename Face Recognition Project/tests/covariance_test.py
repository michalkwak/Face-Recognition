import unittest
import numpy as np

from src.eigenfaces.covariance_matrix import (
    compute_covariance_matrix,
    compute_full_covariance_matrix,
)


class TestComputeCovarianceMatrix(unittest.TestCase):

    def test_matches_example(self):
        # two mean-subtracted images
        mean_subtracted = np.array([
            [1.0, 0.0, -1.0],
            [0.0, 2.0, -2.0],
        ])

        l_matrix = compute_covariance_matrix(mean_subtracted)

        # L[i,j] = dot product of image i and image j
        expected = np.array([
            [2.0, 2.0],
            [2.0, 8.0],
        ])
        np.testing.assert_allclose(l_matrix, expected)

    def test_result_has_correct_shape(self):
        mean_subtracted = np.random.rand(5, 100)    # 5 images, 100 pixels each

        l_matrix = compute_covariance_matrix(mean_subtracted)

        self.assertEqual(l_matrix.shape, (5, 5))

    def test_result_is_symmetric(self):
        mean_subtracted = np.random.rand(4, 50) # 4 images, 50 pixels each

        l_matrix = compute_covariance_matrix(mean_subtracted)

        np.testing.assert_allclose(l_matrix, l_matrix.T)

    def test_raises_on_empty_input(self):
        mean_subtracted = np.empty((0, 10))

        with self.assertRaises(ValueError):
            compute_covariance_matrix(mean_subtracted)

    def test_raises_on_wrong_dimensions(self):
        mean_subtracted = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            compute_covariance_matrix(mean_subtracted)


class TestComputeFullCovarianceMatrix(unittest.TestCase):

    def test_result_has_shape_p_by_p(self):
        mean_subtracted = np.random.rand(4, 6)  # 4 images, 6 pixels each

        c_matrix = compute_full_covariance_matrix(mean_subtracted)

        self.assertEqual(c_matrix.shape, (6, 6))

    def test_result_is_symmetric(self):
        mean_subtracted = np.random.rand(4, 6)

        c_matrix = compute_full_covariance_matrix(mean_subtracted)

        np.testing.assert_allclose(c_matrix, c_matrix.T)

    def test_l_and_c_share_nonzero_eigenvalues(self):
        # Check that the smaller matrix L (m x m)
        # and C (p x p) share the same nonzero eigenvalues
        mean_subtracted = np.array([
            [1.0, 0.0, -1.0, 2.0],
            [0.0, 2.0, -2.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
        ])

        l_matrix = compute_covariance_matrix(mean_subtracted)
        c_matrix = compute_full_covariance_matrix(mean_subtracted)

        l_eigenvalues = np.linalg.eigvalsh(l_matrix)
        c_eigenvalues = np.linalg.eigvalsh(c_matrix)

        # C is 4x4 but rank at most 3, so it has one near-zero eigenvalue
        # L doesn't have it so we compare only the top len(l_eigenvalues)
        c_nonzero = np.sort(c_eigenvalues)[-len(l_eigenvalues):]

        np.testing.assert_allclose(np.sort(l_eigenvalues), c_nonzero, atol=1e-8)
