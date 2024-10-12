import Foundation

extension Tensor {
  public func printing(onForward: String? = nil, onGrad: String? = nil) -> Tensor {
    if onForward == nil && onGrad == nil {
      return self
    }
    let task = createDataTask { t in
      let result = try await t.data
      if let onForward = onForward {
        print(onForward)
      }
      return result
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: task, shape: shape, dtype: dtype)
    } else {
      let handle = saveForBackward()
      return Tensor(dataTask: task, shape: shape, dtype: dtype) { grad in
        handle.backward(Backend.current) { grad.printing(onForward: onGrad) }
      }
    }
  }

  public func checkNaN(onForward: String? = nil, onGrad: String? = nil) -> Tensor {
    alwaysAssert(dtype == .float32)
    if onForward == nil && onGrad == nil {
      return self
    }
    let task = createDataTask { t in
      let result = try await t.data
      if let c = result.completeOnAllDevices {
        let _ = try await c.value
      }
      var floats = [Float](repeating: 0, count: t.shape.product())
      try pointerToArray(result.buffer.contents(), output: &floats, dtype: t.dtype)
      if let onForward = onForward, !floats.allSatisfy({ !$0.isNaN }) {
        print("nan detected: \(onForward)")
      }
      return result
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: task, shape: shape, dtype: dtype)
    } else {
      let handle = saveForBackward()
      return Tensor(dataTask: task, shape: shape, dtype: dtype) { grad in
        handle.backward(Backend.current) { grad.checkNaN(onForward: onGrad) }
      }
    }
  }
}

func alwaysAssert(
  _ condition: Bool, _ message: String? = nil, file: StaticString = #filePath, line: UInt = #line
) {
  if !condition {
    let msg =
      if let message = message {
        "Assertion failure at \(file):\(line) with message: \(message)"
      } else {
        "Assertion failure at \(file):\(line)"
      }
    fatalError(msg)
  }
}
