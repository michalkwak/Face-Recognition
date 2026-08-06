import data.face_dataset_loader
import matplotlib.pyplot as plt
from math import ceil, sqrt

def showImgs(imgs, n_imgs, i_imgs):
    '''Plot the Olivetti faces dataset'''
    rows = ceil(sqrt(n_imgs))
    cols = ceil(n_imgs / rows)

    fig = plt.figure(figsize=(10, 10))

    for p, i in enumerate(i_imgs, start=1):
        fig.add_subplot(rows, cols, p)
        plt.imshow(imgs[i], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_eigenfaces(eigenfaces, eigenvalues, image_shape, n=16):
    """
    Display a grid of eigenfaces

    Inputs:
        eigenfaces: 2D array of shape (p, k), one eigenface per column
        eigenvalues: 1D array of shape k, matching eigenfaces columns
        image_shape: tuple (height, width) the original images had before being flattened, (64, 64) for Olivetti
        n: how many eigenfaces to display
    """
    n = min(n, eigenfaces.shape[1])
    rows = ceil(sqrt(n))
    cols = ceil(n / rows)

    fig = plt.figure(figsize=(10, 10))

    for i in range(n):
        eigenface_image = eigenfaces[:, i].reshape(image_shape)

        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(eigenface_image, cmap="gray")
        plt.title(f"#{i + 1}\nλ={eigenvalues[i]:.1f}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_mean_face(mean_face, image_shape):
    """Display the mean face as a single image."""
    mean_face_image = mean_face.reshape(image_shape)

    plt.figure(figsize=(4, 4))
    plt.imshow(mean_face_image, cmap="gray")
    plt.title("Mean face")
    plt.axis("off")
    plt.show()



