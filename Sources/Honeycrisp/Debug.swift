import Foundation
import HCBacktrace

extension Tensor {
  @recordCaller
  private func _printing(onForward: String? = nil, onGrad: String? = nil) -> Tensor {
    let forwardFn: (@Sendable (Tensor) async throws -> Void)? =
      if let onForward = onForward {
        { t in
          let _ = try await t.data
          print(onForward)
        }
      } else {
        nil
      }
    let bwdFn: (@Sendable (Tensor) async throws -> Void)? =
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

  @recordCaller
  private func _printing(
    onForward: (@Sendable (Tensor) async throws -> Void)? = nil,
    onGrad: (@Sendable (Tensor) async throws -> Void)? = nil
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

  @recordCaller
  private func _checkNaN(onForward: String? = nil, onGrad: String? = nil) -> Tensor {
    #alwaysAssert(dtype == .float32)
    if onForward == nil && onGrad == nil {
      return self
    }
    let task = createDataTask { t in
      let floats = try await t.floats()
      if let onForward = onForward, !floats.allSatisfy({ !$0.isNaN }) {
        print("nan detected: \(onForward)")
      }
      return try await t.data
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
