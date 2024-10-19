import Foundation

extension Tensor {
  public func printing(onForward: String? = nil, onGrad: String? = nil) -> Tensor {
    let forwardFn: ((Tensor) async throws -> Void)? =
      if let onForward = onForward {
        { t in
          let _ = try await t.data
          print(onForward)
        }
      } else {
        nil
      }
    let bwdFn: ((Tensor) async throws -> Void)? =
      if let onGrad = onGrad {
        { t in
          let _ = try await t.data
          print(onGrad)
        }
      } else {
        nil
      }
    return printing(onForward: forwardFn, onGrad: bwdFn)
  }

  public func printing(
    onForward: ((Tensor) async throws -> Void)? = nil,
    onGrad: ((Tensor) async throws -> Void)? = nil
  ) -> Tensor {
    if onForward == nil && onGrad == nil {
      return self
    }
    let task =
      if let onForward = onForward {
        createDataTask { t in
          try await onForward(t)
          return try await t.data
        }
      } else {
        dataTask
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
