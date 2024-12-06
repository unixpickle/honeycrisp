import Foundation
import HCBacktrace

extension Tensor {
  @recordCaller
  private static func adamW(
    param: Tensor,
    grad: Tensor,
    moment1: Tensor,
    moment2: Tensor,
    beta1: Float,
    beta2: Float,
    eps: Float,
    weightDecay: Float,
    lr: Float,
    step: Float
  ) -> (param: Tensor, moment1: Tensor, moment2: Tensor) {
    if Tensor.isGradEnabled {
      #alwaysAssert(!param.needsGrad, "adamW does not support gradients")
      #alwaysAssert(!grad.needsGrad, "adamW does not support gradients")
      #alwaysAssert(!moment1.needsGrad, "adamW does not support gradients")
      #alwaysAssert(!moment2.needsGrad, "adamW does not support gradients")
    }
    #alwaysAssert(
      grad.dtype == param.dtype,
      "mismatching dtypes for param \(param.dtype) and grad \(grad.dtype)")
    #alwaysAssert(
      grad.dtype == moment1.dtype,
      "mismatching dtypes for param \(param.dtype) and moment1 \(moment1.dtype)")
    #alwaysAssert(
      grad.dtype == moment2.dtype,
      "mismatching dtypes for param \(param.dtype) and moment2 \(moment2.dtype)")
    #alwaysAssert(
      grad.shape == param.shape,
      "mismatching shapes for param \(param.shape) and grad \(grad.shape)")
    #alwaysAssert(
      grad.shape == moment1.shape,
      "mismatching shapes for param \(param.shape) and moment1 \(moment1.shape)")
    #alwaysAssert(
      grad.shape == moment2.shape,
      "mismatching shapes for param \(param.shape) and moment2 \(moment2.shape)")

    let backend = Backend.current

    let param = param.noGrad()
    let grad = grad.noGrad()
    let moment1 = moment1.noGrad()
    let moment2 = moment2.noGrad()
    let newData = createDataTask {
      try await backend.adamW(
        param: try await param.data,
        grad: try await grad.data,
        moment1: try await moment1.data,
        moment2: try await moment2.data,
        beta1: beta1,
        beta2: beta2,
        eps: eps,
        weightDecay: weightDecay,
        lr: lr,
        step: step,
        count: param.shape.product(),
        dtype: param.dtype
      )
    }
    return (
      param: Tensor(
        dataTask: Task { try await newData.value.param }, shape: param.shape, dtype: param.dtype),
      moment1: Tensor(
        dataTask: Task { try await newData.value.moment1 }, shape: param.shape, dtype: param.dtype),
      moment2: Tensor(
        dataTask: Task { try await newData.value.moment2 }, shape: param.shape, dtype: param.dtype)
    )
  }
}

/// A base class for gradient-based optimizers.
open class Optimizer {
  public typealias Parameter = Trainable.Parameter

  public let parameters: [String: Parameter]

  public init(_ parameters: [(String, Parameter)]) {
    self.parameters = [String: Parameter](uniqueKeysWithValues: parameters)
  }

  /// Reset the gradients of the parameters.
  ///
  /// This should be called between steps to avoid accumulating gradients incorrectly.
  public func clearGrads() {
    for var p in self.parameters.values {
      p.grad = nil
    }
  }

  internal func step() {
    preconditionFailure("Method not implemented")
  }

  @recordCaller
  private func _step() {
    step()
  }
}

/// An ``Optimizer`` implementation for [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
///
/// This implements [AdamW](https://arxiv.org/abs/1711.05101) when using `weightDecay`.
public class Adam: Optimizer {
  public var lr: Float
  public var beta1: Float
  public var beta2: Float
  public var eps: Float
  public var weightDecay: Float

  public var stepIndex: [String: Int] = [:]
  public var moment1: [String: Tensor] = [:]
  public var moment2: [String: Tensor] = [:]

  /// Create an optimizer wrapping the given parameters.
  ///
  /// The default arguments match those from the original paper.
  public init(
    _ parameters: [(String, Parameter)], lr: Float, beta1: Float = 0.9, beta2: Float = 0.999,
    eps: Float = 1e-8, weightDecay: Float = 0.0
  ) {
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.weightDecay = weightDecay
    super.init(parameters)
  }

  override internal func step() {
    for (name, var param) in parameters {
      guard let grad = param.grad else {
        continue
      }
      let t = stepIndex[name] ?? 1
      stepIndex[name] = t + 1

      let mt = moment1[name] ?? Tensor(zerosLike: grad)
      let vt = moment2[name] ?? Tensor(zerosLike: grad)

      (param.data!, moment1[name], moment2[name]) = Tensor.adamW(
        param: param.data!,
        grad: grad,
        moment1: mt,
        moment2: vt,
        beta1: beta1,
        beta2: beta2,
        eps: eps,
        weightDecay: weightDecay,
        lr: lr,
        step: Float(t)
      )
    }
  }

  /// An encodable object that contains all of the values that this optimizer
  /// tracks during optimization trajectories.
  public struct State: Codable {
    public var stepIndex: [String: Int] = [:]
    public var moment1: [String: TensorState] = [:]
    public var moment2: [String: TensorState] = [:]
  }

  @recordCaller
  private func _state() async throws -> State {
    State(
      stepIndex: stepIndex,
      moment1: try await tensorsToStates(moment1),
      moment2: try await tensorsToStates(moment2)
    )
  }

  @recordCaller
  private func _loadState(_ state: State) throws {
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
