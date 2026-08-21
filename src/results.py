"""
Runs the program comparing accuracy vs number of eigenfaces kept and accuracy
vs amount of training data per person. Saves plots as PNG files
"""

import numpy as np
import matplotlib.pyplot as plt

from cli import load_train_test_split, train_eigenfaces
from eigenfaces.mean_face import compute_mean_face, mean_subtract
from eigenfaces.covariance_matrix import compute_covariance_matrix
from eigenfaces.eigenfaces import compute_eigenfaces
from eigenfaces.classifier import project_all, predict


def accuracy_for(eigenfaces, mean_face, train_images, train_labels, test_images, test_labels):
    train_weights = project_all(train_images, mean_face, eigenfaces)

    correct = 0
    for image, true_label in zip(test_images, test_labels):
        predicted_label, _ = predict(image, mean_face, eigenfaces, train_weights, train_labels)
        correct += predicted_label == true_label

    return correct / len(test_labels)


def sweep_num_components():
    print("Loading dataset")
    train_images, train_labels, test_images, test_labels = load_train_test_split()

    print("Decomposing (takes a few minutes)...")
    mean_face, eigenfaces_all, _ = train_eigenfaces(train_images, num_components=None)

    component_counts = [1, 2, 5, 10, 20, 50, 100, 150, 200]
    accuracies = []

    for k in component_counts:
        eigenfaces_k = eigenfaces_all[:, :k]
        acc = accuracy_for(eigenfaces_k, mean_face, train_images, train_labels, test_images, test_labels)
        accuracies.append(acc)
        print(f"  num_components={k:>3}  accuracy={acc:.2%}")

    plt.figure(figsize=(8, 5))
    plt.plot(component_counts, accuracies, marker="o")
    plt.xlabel("Number of eigenfaces kept (num_components)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs number of eigenfaces")
    plt.grid(True)
    plt.savefig("accuracy_vs_num_components.png")
    print("Saved accuracy_vs_num_components.png")

    return component_counts, accuracies


def sweep_training_set_size():
    """Accuracy vs how many images per person are used for training"""
    from sklearn.datasets import fetch_olivetti_faces

    data = fetch_olivetti_faces()
    all_images = data.images.reshape(len(data.images), -1)
    all_labels = data.target

    images_per_person_options = [1, 2, 3, 5, 7, 9]
    accuracies = []

    for n_train in images_per_person_options:
        train_images, train_labels = [], []
        test_images, test_labels = [], []

        for person in np.unique(all_labels):
            person_images = all_images[all_labels == person]
            train_images.extend(person_images[:n_train])
            train_labels.extend([person] * n_train)
            test_images.append(person_images[-1])  # always hold out the last image
            test_labels.append(person)

        train_images = np.array(train_images)
        test_images = np.array(test_images)

        mean_face = compute_mean_face(train_images)
        mean_subtracted = mean_subtract(train_images, mean_face)
        l_matrix = compute_covariance_matrix(mean_subtracted)
        eigenfaces, _ = compute_eigenfaces(mean_subtracted, l_matrix, num_components=20)

        acc = accuracy_for(eigenfaces, mean_face, train_images, train_labels, test_images, test_labels)
        accuracies.append(acc)
        print(f"  images_per_person={n_train}  accuracy={acc:.2%}")

    plt.figure(figsize=(8, 5))
    plt.plot(images_per_person_options, accuracies, marker="o")
    plt.xlabel("Training images per person")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs amount of training data")
    plt.grid(True)
    plt.savefig("accuracy_vs_training_size.png")
    print("Saved accuracy_vs_training_size.png")

    return images_per_person_options, accuracies


if __name__ == "__main__":
    print("Sweep 1: accuracy vs num_components")
    sweep_num_components()

    print("\nSweep 2: accuracy vs training set size")
    sweep_training_set_size()