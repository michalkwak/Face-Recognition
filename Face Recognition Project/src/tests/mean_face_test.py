import unittest
import numpy as np

from eigenfaces.mean_face import compute_mean_face, mean_subtract


class TestComputeMeanFace(unittest.TestCase):

    def test_mean_of_images(self):
        images = np.array([
            [0.0, 2.0, 4.0],
            [2.0, 4.0, 6.0]
        ])

        mean_face = compute_mean_face(images)

        np.testing.assert_allclose(mean_face, [1.0, 3.0, 5.0])

    def test_mean_of_single_image_equals_the_image(self):
        images = np.array([[1.0, 2.0, 3.0]])

        mean_face = compute_mean_face(images)

        np.testing.assert_allclose(mean_face, [1.0, 2.0, 3.0])

    def test_mean_face_has_correct_shape(self):
        images = np.random.rand(10, 400)  # 10 flattened 20x20 images

        mean_face = compute_mean_face(images)

        self.assertEqual(mean_face.shape, (400,))

    def test_raises_on_wrong_dimensionality(self):
        images = np.array([1.0, 2.0, 3.0])  # m and not (m, p)

        with self.assertRaises(ValueError):
            compute_mean_face(images)

    def test_raises_on_empty_input(self):
        images = np.empty((0, 10))

        with self.assertRaises(ValueError):
            compute_mean_face(images)

class TestMeanSubtract(unittest.TestCase):

    def setUp(self):
        self.images = np.array([
            [0.0, 2.0, 4.0],
            [2.0, 4.0, 6.0]
        ])
        self.mean_face = compute_mean_face(self.images)

    def test_subtracts_mean_correctly(self):
        result = mean_subtract(self.images, self.mean_face)

        expected = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0]
        ])
        np.testing.assert_allclose(result, expected)

    def test_correct_shape(self):
        result = mean_subtract(self.images, self.mean_face)

        self.assertEqual(result.shape, self.images.shape)

    def test_raises_on_wrong_dimension(self):
        wrong_mean = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            mean_subtract(self.images, wrong_mean)

    def test_result_has_zero_mean(self):
        # the mean-subtracted set should average to zero
        result = mean_subtract(self.images, self.mean_face)

        np.testing.assert_allclose(np.mean(result, axis=0), np.zeros(3), atol=1e-12)
