# ``Tensor/checkpoint(enabled:saveRandomState:waitForData:_:_:function:file:line:)``

Apply a differentiable function without storing intermediate results for the backward pass; during the backward pass, call the function again with gradients enabled to backpropagate through it.

## Overview

This is useful for saving memory when a computation graph is deep.

Set `saveRandomState` to `false` to avoid saving and restoring the state of the current backend's default random number generator for the second call to the function. The `saveRandomState` behavior is not thread-safe, since other threads might mutate or depend on the random state at the same time as the checkpointed function.

If `waitForData` is `false`, then gradients may be in the process of being computed asynchronously on-device while the forward pass is recomputed. This might defeat the purpose of using checkpointing as a means to save memory.
