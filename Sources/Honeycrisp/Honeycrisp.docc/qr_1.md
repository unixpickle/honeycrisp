# ``Tensor/qrDecomposition(full:function:file:line:)``

Compute the QR decomposition of the matrix, such that `q &* r` approximates the original matrix.

If the input matrix is m-by-n, then the returned matrices will have the shapes:

 * `q`: if `full` is true, then m-by-m; otherwise, if m > n, then m-by-n
 * `r`: if `full` is true, then m-by-n; otherwise, if m > n, then n-by-n.

If `full` is true, then redundant, arbitrary directions may be present in `q`, accompanied by zeros in `r`.
