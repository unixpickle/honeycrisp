import Foundation
import HCBacktrace

extension Tensor {
  @recordCaller
  private static func _checkpoint(
    enabled: Bool = true,
    saveRandomState: Bool = true,
    waitForData: Bool = true,
    _ args: [Tensor],
    _ fn: @escaping @Sendable ([Tensor]) -> [Tensor]
  ) -> [Tensor] {
    let backend = Backend.current
    let rng: RandomGenerator? =
      if saveRandomState {
        backend.defaultRandom()
      } else {
        nil
      }
    if !Tensor.isGradEnabled || enabled == false {
      return fn(args)
    }
    let saver = RandomStateGuard(rng)
    let capturedResults = Tensor.withGrad(enabled: false) {
      saver.saveOrRestoreRandom {
        fn(args.map { $0.noGrad() })
      }
    }

    // Avoid a case of prematurely backward'ing.
    if capturedResults.allSatisfy({ !$0.dtype.supportsGrad }) {
      return capturedResults
    }

    let inputHandles = args.map { $0.saveForBackward() }
    let state = CheckpointResultState(count: capturedResults.count, trace: Backtrace.current) {
      grads in

      var handlesAndParams = [(BackwardHandle, Trainable.Param<Tensor>)]()
      let newOutputs = Tensor.withGrad(enabled: true) {
        var newInputs = [Tensor]()
        for (x, handle) in zip(args, inputHandles) {
          if x.needsGrad {
            let p: Trainable.Param<Tensor> = .init()
            p.data = x.noGrad()
            handlesAndParams.append((handle, p))
            newInputs.append(p.data!.onGrad { g in backend.use { p.addGrad(g) } })
          } else {
            newInputs.append(x)
          }
        }
        return Tensor.asDependencies(
          grads.filter({ $0 != nil }).map({ $0! }), waitForData: waitForData
        ) {
          backend.use { saver.saveOrRestoreRandom { fn(newInputs) } }
        }
      }

      // We are already in a backward context here, so all of our backward() blocks
      // will be called after this closure returns, but they will still be called in
      // the order we might expect.

      for (output, upstreamGrad) in zip(newOutputs, grads) {
        if let grad = upstreamGrad {
          output.saveForBackward().backward(backend) { grad }
        }
      }

      // We can reference param.grad safely because these backward calls won't be
      // triggered until the above backward()'s and all of their downstream backwards()
      // have completed.
      for (inputHandle, param) in handlesAndParams {
        inputHandle.backward(backend) {
          param.grad ?? Tensor(zerosLike: param.data!)
        }
      }
    }

    let results = capturedResults.enumerated().map { (i, result) in
      if !result.dtype.supportsGrad {
        state.record(index: i, grad: nil)
        return result
      } else {
        let handle = CheckpointResultTensorTracker(state: state, index: i)
        return result.onGrad { g in handle.backward(g) }
      }
    }
    return results
  }
}

final class RandomStateGuard: Sendable {
  let rng: RandomGenerator?
  let lock = NSLock()
  nonisolated(unsafe) var state: Tensor? = nil

  init(_ rng: RandomGenerator?) {
    self.rng = rng
  }

  func saveOrRestoreRandom<T>(_ fn: () throws -> T) rethrows -> T {
    guard let rng = rng else {
      return try fn()
    }
    lock.lock()
    if let state = state {
      lock.unlock()
      let oldState = rng.state
      rng.state = state
      defer { rng.state = oldState }
      return try fn()
    } else {
      state = rng.state
      lock.unlock()
      return try fn()
    }
  }
}

final class CheckpointResultState: Sendable {
  // It will only be called once, and it is passed with sending.
  nonisolated(unsafe) let callback: ([Tensor?]) -> Void

  let trace: [CodeLocation]
  let lock = NSLock()
  nonisolated(unsafe) var completed: [Bool]
  nonisolated(unsafe) var grads: [Tensor?]

  #if compiler(>=6.0)
    init(count: Int, trace: [CodeLocation], _ callback: sending @escaping ([Tensor?]) -> Void) {
      completed = Array(repeating: false, count: count)
      grads = Array(repeating: nil, count: count)
      self.trace = trace
      self.callback = callback
    }
  #else
    init(count: Int, trace: [CodeLocation], _ callback: @escaping ([Tensor?]) -> Void) {
      completed = Array(repeating: false, count: count)
      grads = Array(repeating: nil, count: count)
      self.trace = trace
      self.callback = callback
    }
  #endif

  func record(index: Int, grad: Tensor?) {
    let result: [Tensor?]? = lock.withLock {
      if completed[index] {
        // Allow extra call from deinit
        #alwaysAssert(grad == nil, "second call must be from deinit")
        return nil
      }

      completed[index] = true
      grads[index] = grad
      if completed.allSatisfy({ $0 }) {
        #alwaysAssert(
          grad != nil,
          "Checkpoint backward pass was delayed until one of the results was deinitialized.\n\n"
            + "The offending tensor was returned at index \(index) from code location \(Backtrace.format(trace))"
        )
        return grads
      } else {
        return nil
      }
    }
    if let result = result {
      callback(result)
    }
  }
}

final class CheckpointResultTensorTracker: Sendable {
  let state: CheckpointResultState
  let index: Int

  init(state: CheckpointResultState, index: Int) {
    self.state = state
    self.index = index
  }

  func backward(_ grad: Tensor?) {
    state.record(index: index, grad: grad)
  }

  deinit {
    state.record(index: index, grad: nil)
  }
}
