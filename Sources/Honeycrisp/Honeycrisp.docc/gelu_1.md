# ``Tensor/gelu(mode:function:file:line:)``

Applies the Gaussian Error Linear Unit (GELU) activation function to each element of the ``Tensor``.

## Overview

Typically, GELU is implemented via the expensive expression

    x * 0.5 * (1 + erf(x/sqrt(2)))

However, when `mode` is `.approx` (the default), then the approximation
  
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

is used instead.