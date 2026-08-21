"""
Command-line interface for the eigenfaces

1. Loads the ORL/Olivetti dataset
2. Trains on most images per person
3. Holds one out per person for testing
4. Reports accuracy
"""
import argparse
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from eigenfaces.mean_face import compute_mean_face, mean_subtract
from eigenfaces.covariance_matrix import compute_covariance_matrix
from eigenfaces.eigenfaces import compute_eigenfaces
from eigenfaces.classifier import project_all, predict
from data.visualization import show_eigenfaces, show_mean_face

def load_train_test_split():
    """
    Load the Olivetti faces dataset and split it so that each person's 
    last image is used for testing and the rest for training
    """
    data = fetch_olivetti_faces()
    images = data.images.reshape(len(data.images), -1)  # flatten each image
    labels = data.target

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    people = np.unique(labels)
    for person in people:
        person_images = images[labels == person]

        for image in person_images[:-1]:
            train_images.append(image)
            train_labels.append(person)

        test_images.append(person_images[-1])
        test_labels.append(person)

    return (np.array(train_images), train_labels, np.array(test_images), test_labels)


def train_eigenfaces(train_images, num_components):
    """Compute the mean face, eigenfaces and projected training weights"""

    mean_face = compute_mean_face(train_images)
    mean_subtracted = mean_subtract(train_images, mean_face)
    L = compute_covariance_matrix(mean_subtracted)

    eigenfaces, eigenvalues = compute_eigenfaces(mean_subtracted, L, num_components=num_components)

    return mean_face, eigenfaces, eigenvalues


def compare(mean_face, eigenfaces, train_images, train_labels, test_images, test_labels):
    """Classify every test image and print how many were correct"""
    train_weights = project_all(train_images, mean_face, eigenfaces)

    correct = 0
    for image, true_label in zip(test_images, test_labels):
        predicted_label, distance = predict(
            image, mean_face, eigenfaces, train_weights, train_labels
        )
        is_correct = predicted_label == true_label
        correct += is_correct

        print(f"true={true_label}  predicted={predicted_label}  distance={distance:.2f}")

    accuracy = correct / len(test_labels)
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(test_labels)})")


def main():
    '''Run the whole model'''
    parser = argparse.ArgumentParser(description="Eigenfaces face recognition")
    parser.add_argument("--num-components", type=int, default=20) # how many top eigenfaces to keep
    args = parser.parse_args()

    print("Loading dataset")
    train_images, train_labels, test_images, test_labels = load_train_test_split()

    print("Computing mean face and eigenfaces")
    mean_face, eigenfaces, eigenvalues = train_eigenfaces(train_images, args.num_components)

    print("Comparing images")
    compare(mean_face, eigenfaces, train_images, train_labels, test_images, test_labels)

    show_mean_face(mean_face, image_shape=(64, 64))
    show_eigenfaces(eigenfaces, eigenvalues, image_shape=(64, 64), n=16)


if __name__ == "__main__":
    main()
