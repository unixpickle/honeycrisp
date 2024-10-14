import Foundation

open class Optimizer {
  public typealias Parameter = Trainable.Parameter

  public let parameters: [String: Parameter]

  public init(_ parameters: [(String, Parameter)]) {
    self.parameters = [String: Parameter](uniqueKeysWithValues: parameters)
  }

  public func clearGrads() {
    for var p in self.parameters.values {
      p.grad = nil
    }
  }

  open func step() {
    preconditionFailure("Method not implemented")
  }
}

public class Adam: Optimizer {
  public var lr: Float
  public var beta1: Float
  public var beta2: Float
  public var eps: Float

  public var stepIndex: [String: Int] = [:]
  public var moment1: [String: Tensor] = [:]
  public var moment2: [String: Tensor] = [:]

  public init(
    _ parameters: [(String, Parameter)], lr: Float, beta1: Float = 0.9, beta2: Float = 0.999,
    eps: Float = 1e-8
  ) {
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    super.init(parameters)
  }

  override public func step() {
    for (name, var param) in parameters {
      guard let grad = param.grad else {
        continue
      }
      let t = stepIndex[name] ?? 1
      stepIndex[name] = t + 1

      var mt = moment1[name] ?? Tensor(zerosLike: grad)
      var vt = moment2[name] ?? Tensor(zerosLike: grad)
      mt = beta1 * mt + (1 - beta1) * grad
      vt = beta2 * vt + (1 - beta2) * grad.pow(2)
      moment1[name] = mt
      moment2[name] = vt
      mt = mt / (1 - pow(beta1, Float(t)))
      vt = vt / (1 - pow(beta2, Float(t)))
      param.data! = param.data! - lr * mt / (vt.sqrt() + eps)
    }
  }

  public struct State: Codable {
    public var stepIndex: [String: Int] = [:]
    public var moment1: [String: TensorState] = [:]
    public var moment2: [String: TensorState] = [:]
  }

  public func state() async throws -> State {
    State(
      stepIndex: stepIndex,
      moment1: try await tensorsToStates(moment1),
      moment2: try await tensorsToStates(moment2)
    )
  }

  public func loadState(_ state: State) throws {
    stepIndex = state.stepIndex
    moment1 = statesToTensors(state.moment1)
    moment2 = statesToTensors(state.moment2)
  }
}

private func tensorsToStates(_ d: [String: Tensor]) async throws -> [String: TensorState] {
  var result = [String: TensorState]()
  for (k, v) in d {
    result[k] = try await v.state()
  }
  return result
}

private func statesToTensors(_ d: [String: TensorState]) -> [String: Tensor] {
  var result = [String: Tensor]()
  for (k, v) in d {
    result[k] = Tensor(state: v)
  }
  return result
}
