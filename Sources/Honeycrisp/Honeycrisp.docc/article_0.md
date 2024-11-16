# Introduction to Tensors

An introduction to the ``Tensor`` object.

## Overview

A `Tensor` is an array of primitive values such as numbers or booleans.

We can create an array of floating point values like so:

```swift
let values = Tensor(data: [1.0, 2.0, 3.0, 4.0])
```

Unlike an `Array<Float>` in Swift, we cannot modify the values inside of our `Tensor` in-place.
However, we can create new `Tensor`s using other operations:

```swift
let valuesPlus1 = values + 1 // contains: [2.0, 3.0, 4.0, 5.0]
```

What if we want to print the values inside of our new `Tensor`?
For this, we must use `try await` with a helper method like ``Tensor/floats(function:file:line:)``.
This is because the computations performed on `Tensor`s asynchronously, possibly on devices like GPUs.
This computation might even fail, in which case the result will not be available.

```swift
do {
  print("contents of array:", try await valuesPlus1.floats())
  // Output: [2.0, 3.0, 4.0, 5.0]
} catch {
  // ...
}
```

### Data types

In Swift, we can use generics to define arrays of arbitrary types, such as `[Int64]`, `[Bool]`, `[Float]`, etc.
The data that can be stored in a `Tensor` is restricted to a few primitive types, which are enumerated
in the ``Tensor/DType`` enum.

We can access the type of data stored in a `Tensor` using the ``Tensor/dtype`` attribute.
We can also pass `dtype` arguments to various constructors of `Tensors`.

```swift
let x = Tensor(data: [1, 2], dtype: .float32)
print(x.dtype) // Output: [1.0, 2.0]
```

You can also cast the elements of a `Tensor` using the ``Tensor/cast(_:function:file:line:)`` method.

### Shapes

It is often useful to define multi-dimensional arrays.
One way to think of these is as arrays of arrays (or arrays of arrays of arrays, etc.).
For example, you could create a two-dimensional `Array` in Swift like

```swift
let x: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

When we create multi-dimensional `Tensor`s, we don't explicitly represent them as arrays of arrays.
Instead, we talk about "data" and its "shape".
In the above example, the data might be `[1.0, 2.0, 3.0, 4.0]`, and the shape could be `[2, 3]`.
We can think of the shape as the sizes of the recursively nested arrays; so the outer array is of size `2`, and the inner array is of size `3`.

To create a `Tensor` corresponding to our Swift array above, we can do

```swift
let x = Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3])
```

### Indexing

Suppose we have created a 2-dimensional `Tensor` like so

```swift
let x = Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3])
```

We can think of this as a matrix with two rows and three columns.
Let's say we want to access the second row as its own `Tensor`. We can do

```swift
let secondRow = x[1] // Tensor of shape [3] with data [4.0, 5.0, 6.0]
```

We can also use multiple indices, separated by commas. For example, we can do the following

```swift
let y = x[0, 1] // Tensor of shape [] with data [2.0]
let z = x[1, 1] // Tensor of shape [] with data [5.0]
let w = x[1, 2] // Tensor of shape [] with data [6.0]
```

If the index is an integer, then it will select a specific element along a dimension.
What if we want to select more than one item?
For this, we can use a range:

```swift
let topRightPair = x[0, 1..<3] // Tensor of shape [2] and data [2.0, 3.0]
let rightColumn = x[0...1, 2] // Tensor of shape [2] and data [3.0, 6.0]
```

To select the entire dimension, we can use the ellipsis `...`:

```swift
let rightColumn = x[..., 2] // Tensor of shape [2] and data [3.0, 6.0]
```

We can also use negative indexing, which produces an index beginning from the end rather than the beginning:

```swift
let rightColumn = x[..., -1] // Tensor of shape [2] and data [3.0, 6.0]
```
