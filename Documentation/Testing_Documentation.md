# Testing documentation

## jacobi_test.py

### Unit Tests

**`test_finds_correct_element()`** → verifies that `max_off_diagonal` correctly identifies both the magnitude and the (row, column) indices of the largest off-diagonal element in a 3x3 symmetric matrix
**`test_diagonal_matrix_returns_zero()`** → verifies that a diagonal matrix returns a max value of exactly 0, confirming the convergence check will correctly trigger immediately
**`test_finds_largest_absolute_negative()`** → verifies that the function compares elements by absolute value rather than raw value, correctly finding a large negative element over a smaller positive one

**`test_zeroes_target_element()`** → verifies that a single rotation drives the targeted off-diagonal element `a[k, l]` to (approximately) zero
**`test_preserves_trace()`** → verifies that the sum of diagonal elements is unchanged after a rotation, since a similarity transformation must preserve the trace (and therefore the sum of eigenvalues) of the matrix
**`test_almost_equal_diagonal()`** → verifies that the numerically unstable case, where the diagonal difference `a[l,l] - a[k,k]` is small relative to `a[k,l]`, is handled by the approximation branch without error and still correctly zeroes the target element

### More comprehensive integration and property-based tests

**`test_diagonal_matrix_returns_immediately()`** → verifies that a matrix already in diagonal form is recognized as converged in 0 iterations and returns its diagonal entries unchanged as the eigenvalues
**`test_ev_matches_numpy_2x2(), test_ev_matches_numpy_3x3`** → verifies correctness of the full algorithm by comparing the computed eigenvalues against `numpy.linalg.eigh` on both a 2x2 and a 3x3 symmetric matrix with distinct eigenvalues within a tolerance
**`test_eigenvectors_are_orthonormal()`** → verifies that the full set of returned eigenvectors satisfies `V^T V = I`, confirming they form a valid orthonormal basis rather than just individually-normalized vectors
**`test_satisfies_eigenvalue_equation()`** → verifies correctness independent of any reference implementation by directly checking that `A v = lambda v` holds for every computed (eigenvalue, eigenvector) pair, which remains true regardless of sign or ordering conventions
**`test_reconstructs_original_matrix()`** → verifies that `V · diag(eigenvalues) · V^T` reproduces the original input matrix, confirming the decomposition is a correct factorization and not just a set of individually valid eigenpairs
**`test_converges_within_iterations()`** → verifies that a "reasonable" 3x3 matrix converges well below the `max_iterations` cap
