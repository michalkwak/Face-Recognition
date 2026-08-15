# Face-Recognition

The project implements eigenfaces using the Jacobi eigenvalue algorithm implemented from scratch. Given a set of labeled training face images, the
program learns a low-dimensional "face space" that captures the directions of greatest
variation across the training faces, then classifies new, unseen face images by
projecting them into this space and finding the nearest training face by distance.

The core pipeline (mean face, covariance matrix, eigenvalue decomposition and NN classification) is fully implemented and (almost) tested.
Using the Olivetti dataset, the program produces around 85% accuracy with 20 components.

## Structure

- `src/linalg/jacobi.py` - the Jacobi eigenvalue algorithm implemented from scratch. Takes a symmetric matrix and returns its eigenvalues and eigenvectors
- `src/eigenfaces/mean_face.py` - computes the mean face of the training images and subtracts it from every image
- `src/eigenfaces/covariance_matrix.py` - builds the reduced covariance matrix (the small `m x m` matrix instead of the large `p x p` pixel space, where `m` = number of images, `p` = number of pixels)
- `src/eigenfaces/eigenfaces.py` - runs Jacobi on the covariance matrix, then recovers the eigenfaces and keeps the top `k` of them
- `src/eigenfaces/classifier.py` - projects a face into eigenface space and finds the closest training face using nearest neighbor
- `src/data/loader.py` and `src/eigenfaces/visualization.py` - loading the Olivetti face dataset and displaying images/eigenfaces
- `src/cli.py` - command-line interface for running the program: loads data, trains, tests, prints accuracy
- `src/tests/` - unit and end-to-end tests

## Installation and running the program

Install the dependencies:

```python
poetry install
```

To run the program:

```python
poetry run python src/cli.py --num-components 20
```

`--num-components` controls how many eigenfaces are kept (default is 20).

## Tests

To run tests:

```python
poetry run pytest src
```

## Results

![Accuracy vs number of components](https://github.com/michalkwak/Face-Recognition/blob/main/Documentation/accuracy_vs_num_components.png)

![Accuracy vs training size](https://github.com/michalkwak/Face-Recognition/blob/main/Documentation/accuracy_vs_training_size.png)
