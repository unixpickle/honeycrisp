# 🍎 Honeycrisp

[Documentation](https://swiftpackageindex.com/unixpickle/honeycrisp/main/documentation/honeycrisp) | [Examples](https://github.com/unixpickle/honeycrisp-examples) | [Package Index Page](https://swiftpackageindex.com/unixpickle/honeycrisp)

Automatic differentiation and neural networks, all in Swift for Apple Silicon.

# Examples

See [honeycrisp-examples](https://github.com/unixpickle/honeycrisp-examples) for in-depth usage examples.

## Tensors and operations

We can create a tensor with a shape and data:

```swift
// Create a 2x3 matrix:
//   1  2  3
//   4  5  6
let matrix = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
```

You can perform operations on tensors to get new tensors:

```swift
let matrixPlus1 = matrix + 1
let sumOfColumns = matrix.sum(axis: 1)
```

We can get data out of a tensor using `try await`:

```swift
// Print a [Float] from the raw data of the matrix
print("data as floats:", try await matrix.floats())
```

## Swappable backends

We can run different parts of our computation in different backends:

```swift
Backend.defaultBackend = try MPSBackend()  // Use the GPU by default
let cpuBackend = CPUBackend()
let x = Tensor(rand: [128, 128])  // Performed on GPU
let y = cpuBackend.use { x + 3 }  // Performed on CPU
let z = y - 3  // Performed on GPU
```

## Full training example

Here is a full example of training a dummy model on a simple objective.

First, we define a model with trainable parameters and sub-modules:

```swift
class MyModel: Trainable {
  // A parameter which will be tracked automatically
  @Param var someParameter: Tensor

  // We can also give parameters custom names
  @Param(name: "customName") var otherParameter: Tensor

  // A sub-module whose parameters will also be tracked
  @Child var someLayer: Linear

  override init() {
    super.init()
    self.someParameter = Tensor(data: [1.0])
    self.otherParameter = Tensor(zeros: [7])
    self.someLayer = Linear(inCount: 3, outCount: 7)
  }

  func callAsFunction(_ input: Tensor) -> Tensor {
    // We can access properties like normal
    return someParameter * (someLayer(input) + otherParameter)
  }
}
```

The training loop looks like this:

```swift
@main
struct Main {
  static func main() async {
    do {
      let model = MyModel()
      let optimizer = Adam(model.parameters, lr: 0.1)

      // We will use the same input batch for all iterations.
      let batchSize = 8
      let input = Tensor(rand: [batchSize, 3])

      for i in 0..<10 {
        let output = model(input)
        let loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.clearGrads()
        print("step \(i): loss=\(try await loss.item())")
      }
    } catch {
      print("FATAL ERROR: \(error)")
    }
  }
}
```
