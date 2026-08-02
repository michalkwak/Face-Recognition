# Face-Recognition

The project implements eigenfaces using the Jacobi eigenvalue algorithm implemented from scratch. Given a set of labeled training face images, the
program learns a low-dimensional "face space" that captures the directions of greatest
variation across the training faces, then classifies new, unseen face images by
projecting them into this space and finding the nearest training face by distance.

To run the program run:

<code>poetry run python src/cli.py</code>


