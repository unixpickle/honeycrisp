# ``Tensor/onGrad(_:function:file:line:)``

Create a new tensor with the same data as this one, but with a provided callback to implement the backward pass.

If the current tensor already includes a backward callback, it is not preserved for the returned tensor.