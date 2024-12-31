# ``Tensor/saveForBackward(function:file:line:)``

Create a ``Tensor/BackwardHandle`` to this tensor to be used during the backward pass.

This is typically only necessary when implementing a custom operation's backward pass. Otherwise, it may be sufficient to directly call ``Tensor/backward(_:function:file:line:)``.
