import face_dataset_loader
import matplotlib.pyplot as plt
from math import ceil, sqrt

def showImgs(imgs, n_imgs, i_imgs):
    '''Plot the Olivetti faces dataset.'''
    rows = ceil(sqrt(n_imgs))
    cols = ceil(n_imgs / rows)

    fig = plt.figure(figsize=(10, 10))

    for p, i in enumerate(i_imgs, start=1):
        fig.add_subplot(rows, cols, p)
        plt.imshow(imgs[i], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images = face_dataset_loader.load_olivetti_faces()
    showImgs(images, 100, range(100))