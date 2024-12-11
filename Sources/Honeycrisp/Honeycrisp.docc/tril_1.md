# ``Tensor/tril(offset:function:file:line:)``

Return the lower-triangular part of the ``Tensor``, with all elements above (but not on) the diagonal set to zero.

## Overview

This can be useful for masking out elements of a tensor.

The `offset` parameter can be specified to move the diagonal to the right (positive) or left (negative). Positive offsets result in fewer entries being set to zero.
