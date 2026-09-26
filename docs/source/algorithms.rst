Algorithm Reference
===================

.. note::

   This page describes the mathematical algorithms. The current PyTorch
   compatibility backend stores matrices densely and uses dense solver and
   decomposition wrappers. Algorithmic sparse complexity statements below are
   not runtime guarantees for this implementation. See :doc:`performance`.


This document provides mathematical details of the algorithms implemented in Sparse Kappa.

1-Norm Algorithms
-----------------

Hager-Higham Algorithm
~~~~~~~~~~~~~~~~~~~~~~

**Reference**: Higham & Tisseur (2000), SIAM J. Matrix Anal. Appl.

**Problem**: Estimate :math:`\kappa_1(A) = \|A\|_1 \cdot \|A^{-1}\|_1`

**Algorithm**:

1. Compute :math:`\|A\|_1` exactly:

   .. math::
   
      \|A\|_1 = \max_j \sum_{i=1}^n |a_{ij}|

2. Initialize :math:`x = \mathbf{1}/n` where :math:`\mathbf{1} = (1,1,\ldots,1)^T`

3. For :math:`k = 1, 2, \ldots, k_{\max}`:

   a. Solve :math:`Aw = x` for :math:`w`
   
   b. Estimate :math:`\gamma = \|w\|_1`
   
   c. Compute :math:`\text{sign}(w) = (\text{sgn}(w_1), \ldots, \text{sgn}(w_n))^T`
   
   d. Solve :math:`A^T z = \text{sign}(w)` for :math:`z`
   
   e. **Stopping criterion**: If :math:`\max_i |z_i| \leq z^T x`, stop
   
   f. Find :math:`j = \arg\max_i |z_i|`
   
   g. Update :math:`x = e_j` (standard basis vector)
   
   h. If :math:`x` unchanged, stop

4. Return :math:`\kappa_1(A) = \|A\|_1 \cdot \gamma`

**Complexity**: :math:`O(k \cdot \text{nnz}(A))` where :math:`k` is iterations

**Why LU is crucial**: 
- Requires :math:`2k + 2` solves with :math:`A` and :math:`A^T`
- LU factorization: Compute once, solve :math:`O(n^2)` each
- LSMR: Each solve is :math:`O(m \cdot \text{nnz}(A))`

Power Iteration (1-norm)
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Estimate :math:`\|A^{-1}\|_1`

**Algorithm**:

1. Initialize :math:`x` randomly, normalize by :math:`\|x\|_1`

2. For :math:`k = 1, 2, \ldots`:

   a. Solve :math:`Ay = x` for :math:`y`
   
   b. Estimate :math:`\gamma = \|y\|_1`
   
   c. Normalize :math:`x = y / \|y\|_1`
   
   d. If converged, stop

3. Return :math:`\|A^{-1}\|_1 \approx \gamma`

**Convergence**: Linear, depends on eigenvalue gap

Oettli-Prager Method
~~~~~~~~~~~~~~~~~~~~

**Reference**: Oettli & Prager (1964), Numer. Math.

**Idea**: Sample columns to estimate :math:`\|A^{-1}\|_1`

**Variants**:

1. **Adaptive** (original):

   .. code-block:: none
   
      Start with column j* = argmax ||A(:,j)||₁
      Solve A x = e_j* 
      Use dual problem A^T y = sign(x) to select next column
      Repeat until converged

2. **Random**:

   .. code-block:: none
   
      Sample random vectors b₁, b₂, ..., b_m
      Solve A x_i = b_i
      Return max ||x_i||₁

3. **Hybrid**:

   Combines adaptive and random strategies

2-Norm Algorithms
-----------------

Singular Value Decomposition (SVDS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Compute :math:`\kappa_2(A) = \sigma_{\max}(A) / \sigma_{\min}(A)`

**Method**: Partial SVD using Lanczos bidiagonalization

.. math::

   A = U \Sigma V^T

**Algorithm** (simplified):

1. Compute :math:`k` largest singular values via Lanczos on :math:`A^T A`

2. Compute :math:`k` smallest singular values:

   - Build :math:`A^T A` as linear operator
   - Use ARPACK with shift-invert for smallest eigenvalues
   - :math:`\sigma = \sqrt{\lambda(A^T A)}`

3. Return :math:`\kappa_2(A) = \sigma_{\max} / \sigma_{\min}`

**Accuracy**: Exact for computed singular values

**Complexity**: :math:`O(k \cdot \text{nnz}(A))` where :math:`k` is number of singular values

Lanczos Method
~~~~~~~~~~~~~~

**Reference**: Golub & Van Loan (2013), Matrix Computations

**Idea**: Build tridiagonal matrix :math:`T` whose eigenvalues approximate those of :math:`A^T A`

**Algorithm**:

1. Start with random vector :math:`v_1`, :math:`\|v_1\|_2 = 1`

2. For :math:`j = 1, 2, \ldots, k`:

   .. math::
   
      w &= A^T A v_j - \beta_{j-1} v_{j-1} \\
      \alpha_j &= v_j^T w \\
      w &= w - \alpha_j v_j \\
      \beta_j &= \|w\|_2 \\
      v_{j+1} &= w / \beta_j

3. Build tridiagonal matrix:

   .. math::
   
      T = \begin{pmatrix}
      \alpha_1 & \beta_1 & & \\
      \beta_1 & \alpha_2 & \beta_2 & \\
      & \ddots & \ddots & \ddots \\
      & & \beta_{k-1} & \alpha_k
      \end{pmatrix}

4. Compute eigenvalues :math:`\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_k` of :math:`T`

5. Estimate:

   .. math::
   
      \sigma_{\max}(A) \approx \sqrt{\lambda_1}, \quad
      \sigma_{\min}(A) \approx \sqrt{\lambda_k}

**For symmetric** :math:`A`: Apply Lanczos directly to :math:`A`

Golub-Kahan Bidiagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Reference**: Golub & Kahan (1965), SIAM J. Numer. Anal.

**Advantage**: More stable than forming :math:`A^T A` explicitly

**Algorithm**:

1. Start with :math:`u_1, v_1` random unit vectors

2. For :math:`j = 1, 2, \ldots, k`:

   .. math::
   
      v_j^* &= A^T u_j - \beta_{j-1} v_{j-1} \\
      \alpha_j &= \|v_j^*\|_2, \quad v_j = v_j^* / \alpha_j \\
      u_{j+1}^* &= A v_j - \alpha_j u_j \\
      \beta_j &= \|u_{j+1}^*\|_2, \quad u_{j+1} = u_{j+1}^* / \beta_j

3. Build bidiagonal matrix:

   .. math::
   
      B = \begin{pmatrix}
      \alpha_1 & \beta_1 & & \\
      & \alpha_2 & \beta_2 & \\
      & & \ddots & \ddots \\
      & & & \alpha_k
      \end{pmatrix}

4. Singular values of :math:`B` approximate those of :math:`A`

**Stability**: Better conditioned than :math:`A^T A`

Power Method (2-norm)
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Estimate :math:`\sigma_{\max}(A)`

**Algorithm**:

1. Start with random vector :math:`v`, :math:`\|v\|_2 = 1`

2. For :math:`k = 1, 2, \ldots`:

   .. math::
   
      w &= A^T A v \\
      \lambda &= v^T w \\
      v &= w / \|w\|_2

3. Return :math:`\sigma_{\max} \approx \sqrt{\lambda}`

**For** :math:`\sigma_{\min}`: Use inverse iteration or shift-invert

Convergence Analysis
--------------------

Hager-Higham
~~~~~~~~~~~~

**Typical iterations**: 2-5

**Convergence**: Depends on matrix structure, not proven in general

**In practice**: Usually converges quickly

Power Iteration
~~~~~~~~~~~~~~~

**Convergence rate**:

.. math::

   \text{error}_k \approx \left|\frac{\lambda_2}{\lambda_1}\right|^k \text{error}_0

where :math:`\lambda_1, \lambda_2` are two largest eigenvalues.

**Fast if**: :math:`\lambda_1 \gg \lambda_2` (well-separated)

Lanczos
~~~~~~~

**Accuracy**: Extremal eigenvalues converge first

**Iterations needed**: Typically :math:`k = 10-50` for good estimates

Numerical Stability
-------------------

Loss of Orthogonality
~~~~~~~~~~~~~~~~~~~~~

**Problem**: In Lanczos, :math:`v_j` vectors lose orthogonality due to rounding

**Solution**: 

- Reorthogonalization
- Use more iterations
- Check residuals

Ill-Conditioned Systems
~~~~~~~~~~~~~~~~~~~~~~~

**Challenge**: Solving :math:`Ax = b` when :math:`\kappa(A)` is large

**Solutions**:

1. Use iterative refinement
2. Increase solver tolerance
3. Use preconditioners
4. Accept approximate results

References
----------

1. **Hager, W. W.** (1984). "Condition estimates." *SIAM J. Sci. Stat. Comput.*, 5(2), 311-316.

2. **Higham, N. J., & Tisseur, F.** (2000). "A block algorithm for matrix 1-norm estimation." *SIAM J. Matrix Anal. Appl.*, 21(4), 1185-1201.

3. **Golub, G. H., & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.

4. **Golub, G. H., & Kahan, W.** (1965). "Calculating the singular values and pseudo-inverse of a matrix." *SIAM J. Numer. Anal.*, 2(2), 205-224.

5. **Saad, Y.** (2011). *Numerical Methods for Large Eigenvalue Problems* (2nd ed.). SIAM.

6. **Oettli, W., & Prager, W.** (1964). "Compatibility of approximate solution of linear equations." *Numer. Math.*, 6(1), 405-409.
