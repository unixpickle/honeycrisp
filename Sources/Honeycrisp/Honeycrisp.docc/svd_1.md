# ``Tensor/svd(full:function:file:line:)``

Compute the singular value decomposition of the matrix.

If the input matrix is m-by-n, then the returned matrices will have the shapes:

 * `u`: if `full` is true, then m-by-m; otherwise, m-by-min(m,n).
 * `s`: a 1-D array of length min(m,n).
 * `vt`: if `full` is true, then n-by-n; otherwise, n-by-min(m,n).

If `full` is true, then redundant, arbitrary dimensions might be present in `u` and `vt` when the input matrix is rectangular.

If `full` is false, then `u &* Tensor.diagonal(s) &* vt` should approximate the input matrix.
