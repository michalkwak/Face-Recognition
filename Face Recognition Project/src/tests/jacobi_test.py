import unittest
import numpy as np
from linalg.jacobi import max_off_diagonal

class TestJacobi(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([
            [6.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [0.0, 2.0, 2.0],
        ])
    
    def test_max_off_diagonal_on_diagonal_matrix_returns_zero(self):
        diagonal = np.diag([1.0, 2.0, 3.0])

        max_val, p, q = max_off_diagonal(diagonal, 3)

        self.assertEqual(max_val, 0.0)

    def test_max_off_diagonal_finds_correct_element(self):
        max_val, p, q = max_off_diagonal(self.matrix, 3)

        self.assertEqual(max_val, 2.0)
        self.assertEqual((p, q), (0, 2))
