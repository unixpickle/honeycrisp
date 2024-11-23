# 🍎 Honeycrisp

Automatic differentiation and neural networks, all in Swift for Apple Silicon.

# Examples

See [honeycrisp-examples](https://github.com/unixpickle/honeycrisp-examples) for in-depth usage examples.

Here is a full example of training a dummy model on a simple objective:

```swift
import Honeycrisp

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

@main
struct Main {
  static func main() async {
    do {

      let batchSize = 8
      let model = MyModel()

      let optimizer = Adam(model.parameters, lr: 0.1)

      let input = Tensor(rand: [batchSize, 3])

      for i in 0..<10 {
        let output = model(input)
        let loss = output.pow(2).mean()
        loss.backward()

        if i == 0 {
          print("----------------")
          for (name, param) in model.parameters {
            let grad = param.grad!
            print(
              "parameter \(name) has shape \(grad.shape) and gradient: \(try await grad.floats())")
          }
          print("----------------")
        }
        print("step \(i): loss=\(try await loss.item())")

        // Automatically updates the data inside of each parameter
        optimizer.step()

        // Unsets the grad of each parameter in preparation for future updates
        optimizer.clearGrads()
      }
      print("finished training model")

    } catch {
      print("FATAL ERROR: \(error)")
    }
  }
}
```
