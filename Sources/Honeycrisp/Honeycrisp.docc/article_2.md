# Random Number Generation

How to seed, save, and use random number generators.

## Overview

Randomness is often useful in machine learning workloads.

Random ``Tensor`` objects can be constructed with a few convenience initializers. For example:

```swift
// Generate 12 uniformly random floating-point values in the range [0, 1).
let x = Tensor(rand: [12])

// Generate a 3x3 matrix of samples from the Normal distribution.
let y = Tensor(randn: [3, 3])

// Generate a list of 32 integers randomly sampled from [0, 16).
let z = Tensor(randInt: [32], in: 0..<16)
```

All of the above initializers take an optional `generator` argument which is an instance of ``RandomGenerator``. If unspecified, then the generator returned by ``Backend/defaultRandom()`` is used. You can create your own random generator using ``Backend/createRandom()``, and it can be manipulated using ``RandomGenerator/state`` and ``RandomGenerator/seed(_:)``.

```swift
let rng = Backend.current.createRandom()

rng.seed(1337) // Optionally seed the generator
let state = rng.state // Get the current RNG state (as a Tensor)

let sampled = Tensor(rand: [3, 3], generator: rng)

rng.state = state // Restore the state to a previously saved state
let sampled1 = Tensor(rand: [3, 3], generator: rng)

// `sampled1` should equal `sampled`

rng.seed(1337) // Re-seed with the same value as before
let sampled2 = Tensor(rand: [3, 3], generator: rng)
// `sampled2` should equal `sampled` and `sampled1`
```
