# Testing documentation

## jacobi_test.py

### Unit tests 1

**`test_finds_correct_element()`** → verifies that `max_off_diagonal` correctly identifies both the magnitude and the (row, column) indices of the largest off-diagonal element in a 3x3 symmetric matrix

**`test_diagonal_matrix_returns_zero()`** → verifies that a diagonal matrix returns a max value of exactly 0, confirming the convergence check will correctly trigger immediately

**`test_finds_largest_absolute_negative()`** → verifies that the function compares elements by absolute value rather than raw value, correctly finding a large negative element over a smaller positive one

**`test_zeroes_target_element()`** → verifies that a single rotation drives the targeted off-diagonal element `a[k, l]` to (approximately) zero

**`test_preserves_trace()`** → verifies that the sum of diagonal elements is unchanged after a rotation, since a similarity transformation must preserve the trace (and therefore the sum of eigenvalues) of the matrix

**`test_almost_equal_diagonal()`** → verifies that the numerically unstable case, where the diagonal difference `a[l,l] - a[k,k]` is small relative to `a[k,l]`, is handled by the approximation branch without error and still correctly zeroes the target element

### More comprehensive tests 1

**`test_diagonal_matrix_returns_immediately()`** → verifies that a matrix already in diagonal form is recognized as converged in 0 iterations and returns its diagonal entries unchanged as the eigenvalues

**`test_ev_matches_numpy_2x2(), test_ev_matches_numpy_3x3`** → verifies correctness of the full algorithm by comparing the computed eigenvalues against `numpy.linalg.eigh` on both a 2x2 and a 3x3 symmetric matrix with distinct eigenvalues within a tolerance

**`test_eigenvectors_are_orthonormal()`** → verifies that the full set of returned eigenvectors satisfies `V^T V = I`, confirming they form a valid orthonormal basis instead of just individually normalized vectors

**`test_satisfies_eigenvalue_equation()`** → verifies correctness independent of any reference implementation by directly checking that `A v = lambda v` holds for every computed (eigenvalue, eigenvector) pair, which remains true regardless of sign or ordering conventions

**`test_reconstructs_original_matrix()`** → verifies that `V · diag(eigenvalues) · V^T` reproduces the original input matrix, confirming the decomposition is a correct factorization and not just a set of individually valid eigenpairs

**`test_converges_within_iterations()`** → verifies that a "reasonable" 3x3 matrix converges well below `max_iterations`

## mean_face_test.py

### Unit tests 2

**`test_mean_of_images()`** → verifies that `compute_mean_face` returns the correct elementwise average for a small set of training images

**`test_mean_of_single_image_equals_the_image()`** → verifies that when the training set contains only one image, the mean face is simply that image

**`test_mean_face_has_correct_shape()`** → verifies that the returned mean face has shape `(p,)` matching the pixel dimension of the input images, independent of how many training images were used

**`test_raises_on_empty_input()`** → verifies that `compute_mean_face` raises a `ValueError` when given zero training images, rather than returning a result

**`test_raises_on_wrong_dimensionality()`** → verifies that a 1D vector, instead of the expected 2D matrix of shape `(m, p)`, is rejected with a `ValueError`

**`test_subtracts_mean_correctly()`** → verifies that `mean_subtract` produces the exact expected mean-subtracted values for a small training set

**`test_preserves_shape()`** → verifies that mean-subtraction does not change the shape of the training set

**`test_raises_on_wrong_dimension()`** → verifies that passing a mean face vector of the wrong length raises a `ValueError`

### More comprehensive tests 2

**`test_result_has_zero_mean()`** → verifies that averaging the mean-subtracted training set across all images results in the zero vector

## covariance_test.py

### Unit tests 3

**`test_matches_example()`** → verifies that `compute_covariance_matrix` produces the exact expected values for a small hand-computed example, that is `L[i,j]` equals the dot product of mean-subtracted images `i` and `j`

**`test_result_has_correct_shape()`** → verifies that the reduced covariance matrix has shape `(m, m)`, where `m` is the number of training images, rather than the full `(p, p)` pixel space

**`test_result_is_symmetric()`** → verifies that the computed matrix satisfies `L = L^T`

**`test_raises_on_empty_input()`** → verifies that `compute_covariance_matrix` raises a `ValueError` when given zero training images

**`test_raises_on_wrong_dimensions()`** → verifies that a 1D input, instead of the expected 2D array of shape `(m, p)` is rejected with a `ValueError`

### More comprehensive tests 3

**`test_l_and_c_share_nonzero_eigenvalues()`** → verifies the property that the reduced `m x m` matrix `L` and the full `p x p` pixel space covariance matrix `C` share the same nonzero eigenvalues.

## real_input_test.py

Tests using representative inputs: real face images from
the Olivetti dataset (4096 pixels each) using a subset of 10 people with 5 images each

### Tests with representative inputs

**`test_jacobi_converges_on_large_covariance_matrix()`** → verifies that the Jacobi eigendecomposition actually converges within `max_sweeps` on a real covariance matrix (50x50 built from real face images), rather than only on the small hand-constructed matrices.

**`test_model_accuracy()`** → runs the full pipeline end-to-end (mean face, covariance matrix, Jacobi eigendecomposition, eigenface extraction, and NN classification) on real face images and checks that accuracy is real (above 0.3, compared to 0.1 for random guessing among 10 people)

## eigenfaces_test.py

### Unit Tests 4

**`test_eigenfaces_have_correct_shape()`** → verifies that `compute_eigenfaces` returns eigenfaces and eigenvalues with the expected shapes, `(p, k)` and `(k,)`

**`test_eigenfaces_are_unit_length()`** → verifies that every returned eigenface has been normalized to unit length

**`test_projecting_mean_face_gives_zero_weights()`** → verifies that projecting the mean face into eigenspace should give a weight vector of all zeros since the mean face has nothing different from itself

## loader_test.py

### Unit Tests 5

**`test_no_person_appears_in_both_train_and_test()`** → verifies that the train/test split includes every person in both sets correctly

**`test_test_set_has_one_image_per_person()`** → verifies that the test set contains exactly one image per person

## Coverage report

| Name                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing              |
|--------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------:              |
| src\\cli.py                            |       55 |       26 |        8 |        1 |     54% |50-56, 61-74, 79-93, 97 |
| src\\data\\\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |                        |
| src\\data\\visualization.py            |       32 |       27 |        4 |        0 |     14% |6-17, 29-44, 49-55      |
| src\\eigenfaces\\classifier.py         |       17 |        0 |        6 |        0 |    100% |                        |
| src\\eigenfaces\\covariance\_matrix.py |       10 |        1 |        6 |        1 |     88% |        39              |
| src\\eigenfaces\\eigenfaces.py         |       19 |        1 |        4 |        1 |     91% |        37              |
| src\\eigenfaces\\mean\_face.py         |       11 |        0 |        6 |        0 |    100% |                        |
| src\\linalg\\\_\_init\_\_.py           |        0 |        0 |        0 |        0 |    100% |                        |
| src\\linalg\\jacobi.py                 |       54 |        2 |       28 |        1 |     96% |     28-29              |
| src\\tests\\\_\_init\_\_.py            |        0 |        0 |        0 |        0 |    100% |                        |
| src\\tests\\covariance\_test.py        |       47 |        8 |        0 |        0 |     83% |67-69, 72, 85-88, 93-95 |
| src\\tests\\eigenfaces\_test.py        |       25 |        0 |        0 |        0 |    100% |                        |
| src\\tests\\jacobi\_test.py            |       69 |        0 |        2 |        0 |    100% |                        |
| src\\tests\\loader\_test.py            |        7 |        0 |        0 |        0 |    100% |                        |
| src\\tests\\mean\_face\_test.py        |       42 |        0 |        0 |        0 |    100% |                        |
| src\\tests\\real\_input\_test.py       |       47 |        0 |        6 |        0 |    100% |                        |
| **TOTAL**                              |  **435** |   **65** |   **70** |    **4** | **85%** |                        |
