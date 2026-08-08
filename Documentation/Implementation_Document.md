# Implementation Document

## General structure of the program

At the core is `src/linalg/jacobi.py`, which a self implementation of the Jacobi eigenvalue algorithm, which takes a symmetric matrix and returns its eigenvalues and eigenvectors. Then under `src/eigenfaces/`: `mean_face.py` computes the mean face of the training images and subtracts it from every image, `covariance_matrix.py` builds the reduced covariance matrix (the much smaller `m x m` matrix instead of the large `p x p` pixel space one, where `m` is the number of images and `p` the number of pixels), `eigenfaces.py` glues these together by running the Jacobi algorithm on the covariance matrix and recovering the actual eigenfaces (keeping only the top `k` of them), and `classifier.py` projects a face into eigenface space and finds the closest training face using nearest neighbor. Everything is tied together in `src/cli.py`, where it trains the model, runs it on test images, and prints the accuracy.

The general pipeline is: load face images → compute the mean face → subtract it from all images → build the reduced covariance matrix → run the Jacobi algorithm on it to get eigenvectors → turn those into actual eigenfaces → project every training face into eigenface space → for a new face, project it the same way and find the closest training face by distance.

## Achieved time and space complexity

...

## Performance comparison

I actually tried two versions of the Jacobi algorithm:

- **Classical Jacobi** - always rotate the single largest off-diagonal element. On the real dataset (400 images, so a 400x400 matrix), this did not converge even after 40,000 iterations and took around 5 minutes. The problem is that finding the largest element every single time costs O(m^2) so most of the time was spent searching and not actually fixing the matrix.
- **Cyclic Jacobi** - go through every pair once per sweep, in order. This actually converged in under 100 sweeps, but still took a about 3 minutes on the full dataset since each sweep does a lot of work.

## Possible shortcomings and improvements

- The Jacobi implementation is in plain Python with nested loops so it takes a while
- Accuracy (around 87.5% on the full dataset with 20 eigenfaces) could be improved with a different classification method (for example the paper I read used a neural network on top of the eigenface features instead of plain nearest neighbor), but that would be a bigger project on its own

## Use of LLMs

I used Claude to:

- Help me understand the difference between the Jacobi iterative method for solving linear systems and the Jacobi eigenvalue algorithm (these have the same name but are different algorithms and I originally mixed them up)
as well as the difference between classical and cyclic Jacobi
- Help me understand what some properties I missed should be tested
- Help me debug import/module errors while setting up Poetry

## Sources

- Turk, M.A. & Pentland, A.L. (1991). "Face recognition using eigenfaces." Proc. IEEE Computer Society Conf. on Computer Vision and Pattern Recognition, pp. 586-591
- Rizon, M. et al. (2006). "Face Recognition using Eigenfaces and Neural Networks." American Journal of Applied Sciences, 2(6), 1872-1875
- Wikipedia articles on Eigenface, Jacobi eigenvalue algorithm, and Principal component analysis
- Olivetti/ORL Face Database (via `scikit-learn`)
