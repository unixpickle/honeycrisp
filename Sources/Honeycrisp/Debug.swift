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
}
