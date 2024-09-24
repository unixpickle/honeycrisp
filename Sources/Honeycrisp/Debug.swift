extension Tensor {
  public func printing(onForward: String? = nil, onGrad: String? = nil) -> Tensor {
    if onForward == nil && onGrad == nil {
      return self
    }
    let task = Task {
      let result = try await dataTask.value
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
        handle.backward(grad.printing(onForward: onGrad))
      }
    }
  }

  public func checkNaN(onForward: String? = nil, onGrad: String? = nil) -> Tensor {
    assert(dtype == .float32)
    if onForward == nil && onGrad == nil {
      return self
    }
    let task = Task {
      let result = try await dataTask.value
      if let c = result.completeOnAllDevices {
        let _ = try await c.value
      }
      var floats = [Float](repeating: 0, count: shape.product())
      try pointerToArray(result.buffer.contents(), output: &floats, dtype: dtype)
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
        handle.backward(grad.checkNaN(onForward: onGrad))
      }
    }
  }
}
