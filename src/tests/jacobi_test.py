import unittest
import numpy as np
from linalg.jacobi import rotate, jacobi_eigenvalue

class TestApplyRotation(unittest.TestCase):
    """Tests for a single rotation"""

    def setUp(self):
        self.a = np.array([
            [2.0, 1.0],
            [1.0, 3.0]
        ])
        self.v = np.identity(2)

    def test_zeroes_target_element(self):
        rotate(self.a, self.v, 2, 0, 1)

        self.assertAlmostEqual(self.a[0, 1], 0.0)

    def test_preserves_trace(self):
        trace = np.trace(self.a)

        rotate(self.a, self.v, 2, 0, 1)

        self.assertAlmostEqual(np.trace(self.a), trace)

    def test_almost_equal_diagonal(self):
        a = np.array([
            [2.0, 1.0],
            [1.0, 2.0 + 1e-12]
        ])
        v = np.identity(2)

        rotate(a, v, 2, 0, 1)

        self.assertAlmostEqual(a[0, 1], 0.0)

class TestJacobiEigenvalue(unittest.TestCase):
    """End to end tests for the final method against Numpy"""
    def setUp(self):
        self.a = np.array([
            [6.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 2.0]
        ])

    def test_diagonal_matrix_returns_immediately(self):
        diagonal = np.diag([1.0, 2.0, 3.0])

        eigenvalues, eigenvectors, iterations = jacobi_eigenvalue(diagonal)

        self.assertEqual(iterations, 0)
        np.testing.assert_allclose(sorted(eigenvalues), [1.0, 2.0, 3.0])

    def test_ev_matches_numpy_2x2(self):
        a = np.array([
            [2.0, 1.0],
            [1.0, 3.0]
        ])

        eigenvalues, eigenvectors, iterations = jacobi_eigenvalue(a)
        expected_ev, eigenvectors = np.linalg.eigh(a)

        np.testing.assert_allclose(sorted(eigenvalues), sorted(expected_ev), atol=1e-8)

    def test_ev_matches_numpy_3x3(self):

        eigenvalues, eigenvectors, iterations = jacobi_eigenvalue(self.a)
        expected_ev, eigenvectors = np.linalg.eigh(self.a)

        np.testing.assert_allclose(sorted(eigenvalues), sorted(expected_ev), atol=1e-8)

    def test_eigenvectors_are_orthonormal(self):

        eigenvalues, eigenvectors, iterations = jacobi_eigenvalue(self.a)

        identity_reconstructed = eigenvectors.T @ eigenvectors
        np.testing.assert_allclose(identity_reconstructed, np.identity(3), atol=1e-8)

    def test_satisfies_eigenvalue_equation(self):
        # A v = lambda v for every (eigenvalue, eigenvector) pair
        eigenvalues, eigenvectors, iterations = jacobi_eigenvalue(self.a)

        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lhs = self.a @ v
            rhs = eigenvalues[i] * v
            np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    def test_reconstructs_original_matrix(self):
        # A = V diag(eigenvalues) V^T should give the original matrix
        eigenvalues, eigenvectors, iterations = jacobi_eigenvalue(self.a)

        original_reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(original_reconstructed, self.a, atol=1e-8)

    def test_converges_within_iterations(self):
        eigenvalues, eigenvectors, iterations = jacobi_eigenvalue(self.a)
        self.assertLess(iterations, 100)