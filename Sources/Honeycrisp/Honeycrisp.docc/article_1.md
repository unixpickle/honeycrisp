# Automatic Differentiation

An overview of how to compute gradients of functions with respect to inputs using ``Tensor``s.

## Overview

It is essential to be able to compute derivatives of functions when training neural networks.
With a `Tensor`, we can leverage reverse-mode automatic differentiation to compute derivatives automatically.

For example, let's compute the gradient of `sqrt(sum_i x_i^2)` with respect to an array of inputs `x_i`:

```swift
let x = Tensor(data: [1.0, 2.0, 3.0])

// Create a version of `x` that will store its gradients in `grad`
var grad: Tensor?
let differentiableX = x.onGrad { g in grad = g }

// Compute the final value we want to differentiate
let norm = differentiableX.pow(2).sum().sqrt()

// Perform reverse-mode automatic differentiation to compute gradients.
norm.backward()

print(try await grad!.floats())
// Output: [0.26726124, 0.5345225, 0.8017837]
```

When a `Tensor` is created from data, it is typically not differentiable.
Instead, it will be treated as a constant in the computation graph.
To tell if a `Tensor` is a constant, check the ``Tensor/needsGrad`` property.

When we call ``Tensor/onGrad(_:function:file:line:)``, we create a new `Tensor` that requires gradients, and is therefore not a constant.
Furthermore, whenever we perform an operation on a `Tensor` that requires gradients, the resulting `Tensor` will also require gradients. When the backward pass is triggered by a `backward()` call on some downstream result `Tensor`, all non-constant `Tensor`s that were used in the computation will receive callbacks with gradients.

### Pitfall: don't create multiple computation graphs!

The computation graph is tracked by leveraging Swift reference counting.
This can cause errors when a `Tensor` is used in two different computation graphs, and only one
graph is differentiated with a `backward()` call.
For example, this code is incorrect:

```swift
let x = Tensor(data: [1.0, 2.0, 3.0]).onGrad { _ in
  print("this won't end up being called")
}
let y = x + 2
let z = x * 3

// Incorrect: backward through graph that used x, when a different graph is
// still around that also used x.
// As a result, `x` will not receive a gradient since it is still waiting for
// a backward pass through `z`.
y.sum().backward()

print(try await z.floats())

// After the scope exits and `z` is released, we will see an assertion failure:
//
//     Assertion failure: backward pass was incompleted due to an unused reference.
//
//     Traceback of reference creation:
// 
//     *(_:_:) at /Users/alex/code/github.com/unixpickle/honeycrisp/Sources/Honeycrisp/Tensor.swift:685
```
