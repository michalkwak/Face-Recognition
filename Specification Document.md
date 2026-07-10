# Specification Document

I am a Computer and Data Science student in the Bachelor's Programme in Science (BSC). The reports will be written in English. I will use Python for my project. I can give peer reviews only in Python.

## Problem

The project implements eigenfaces, which is a face recognition method based on
Principal Component Analysis (PCA). Given a set of labeled training face images, the
program learns an eigensapce ("face space") that shows the directions of greatest
variation across the training faces. It then classifies new, unseen face images by
projecting them into this space and finding the nearest training face.

## Inputs

First the model is trained on a set of grayscale face images, each with a known identity. 
Then the program takes a query image (a face from the training set or not) and uses the training images to compute the mean face, 
covariance structure and eigenfaces. It uses the query image to compute a projection that is compared
against the projected training faces to produce a prediction of the identity.

## Core

The core is the PCA pipeline. I need to compute the mean face and
covariance structure, extract eigenfaces with a Jacobi
eigendecomposition and project faces into it for classification.

## Algorithms used

- Vector/matrix representation of flattened images
- Covariance matrix computation
- Eigendecomposition using the Jacobian method
- Nearest-neighbor search for classification

## Expected time and space complexity

n = number of training images, d = number of pixels in an image

First the covariance matrix construction (reduced n×n form) which takes O(n²·d) time and O(n²) space for the matrix itself and O(n·d) space for training data.
Then the Jacobi algorithm on the n×n covariance matrix where each rotation updates two rows/columns in O(n) time, so a full computation over all off-diagonal elements is O(n³) with O(n²) space.
Recovering eigenfaces from the reduced eigenvectors is O(n·d) time.
Finally the classification of an input image takes O(k) to project into eigenspace, where k = eigenfaces retained, and O(n·k) to compare against all training projections.


## Sources

- Turk, M., & Pentland, A. (1991). "Eigenfaces for Recognition." Journal of Cognitive
  Neuroscience
- Rizon, Mohamed & Hashim, Muhammad & Saad, Puteh & Yaacob, Sazali & Mamat, Mohd & md shakaff, ali yeon & Saad, Abdul & Desa, Hazri & Karthigaya, M.. (2006). Face Recognition using Eigenfaces and Neural Networks. American Journal of Applied Sciences
- Wikipedia: Eigenfaces, Jacobi eigenvalue algorithm, Principal component analysis
