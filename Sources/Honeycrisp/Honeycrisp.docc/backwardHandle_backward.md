# ``Tensor/BackwardHandle/backward(_:_:function:file:line:)``

Enqueue an operation to contribute to the gradient of the underlying ``Tensor`` that this handle was created for.

## Overview

When the gradient implementation is called, it is done so with the provided backend accessible as ``Backend/current``. This is done to avoid scenarios where a backward implementation mistakenly fails to use the correct ``Backend`` in the backward pass, by forcing the implementor to _think_ about what ``Backend`` should be used.

When the gradient is computed, it may trigger further gradients to be computed for inputs of the handle's underlying Tensor. If multiple downstream operations use the underlying `Tensor`, then the results may be accumulated until the last gradient is computed for the underlying `Tensor`.

When this is called, the provided backward implementation may not be immediately run, but may be added to the back of a queue. However, it is guaranteed (under unexceptional circumstances) that this function will be called during this same backward pass.