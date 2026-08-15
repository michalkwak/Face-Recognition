import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from eigenfaces.mean_face import compute_mean_face, mean_subtract
from eigenfaces.covariance_matrix import compute_covariance_matrix
from eigenfaces.eigenfaces import compute_eigenfaces
from eigenfaces.classifier import project_all, predict
from linalg.jacobi import jacobi_eigenvalue


def load_test_data(num_people=10, images_per_person=5):
    """Loads a small subset from the Olivetti dataset"""
    data = fetch_olivetti_faces()
    #images = data.images.reshape(len(data.images), -1)
    images = np.reshape(data.images, (len(data.images), -1))
    labels = data.target

    subset_images, subset_labels = [], []
    for person in np.unique(labels)[:num_people]:
        person_images = images[labels == person][:images_per_person]
        subset_images.extend(person_images)
        subset_labels.extend([person] * len(person_images))

    return np.array(subset_images), subset_labels


def test_jacobi_converges_on_large_covariance_matrix():
    """Tests if the Jacobi eigendecomposition converges on a representative input size"""
    images, _ = load_test_data(num_people=10, images_per_person=5)  # 50 images, 4096 pixels each

    mean_face = compute_mean_face(images)
    mean_subtracted = mean_subtract(images, mean_face)
    l_matrix = compute_covariance_matrix(mean_subtracted)

    _, _, sweeps = jacobi_eigenvalue(l_matrix, max_sweeps=100)

    assert sweeps < 100, "Jacobi did not converge on a real (subset) covariance matrix"


def test_model_accuracy():
    """End-to-end test for the whole pipeline measuring the accuracy of the model"""
    images, labels = load_test_data(num_people=10, images_per_person=5)

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for person in np.unique(labels):
        person_images = images[np.array(labels) == person]
        train_images.extend(person_images[:-1])
        train_labels.extend([person] * (len(person_images) - 1))
        test_images.append(person_images[-1])
        test_labels.append(person)

    train_images = np.array(train_images)
    test_images = np.array(test_images)

    mean_face = compute_mean_face(train_images)
    mean_subtracted = mean_subtract(train_images, mean_face)
    l_matrix = compute_covariance_matrix(mean_subtracted)
    eigenfaces, _ = compute_eigenfaces(mean_subtracted, l_matrix, num_components=10)

    train_weights = project_all(train_images, mean_face, eigenfaces)

    correct = 0
    for image, true_label in zip(test_images, test_labels):
        predicted_label, _ = predict(image, mean_face, eigenfaces, train_weights, train_labels)
        correct += predicted_label == true_label

    accuracy = correct / len(test_labels)
    assert accuracy > 0.3, f"accuracy is {accuracy:.2%} on test data"