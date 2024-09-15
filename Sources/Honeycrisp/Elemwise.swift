import Foundation

public enum ElemwiseOp {
  case sin
  case cos
  case minusSin
  case exp
  case log
  case recip
  case sigmoid
  case sigmoidGrad
  case relu
  case reluGrad

  public func apply<T: NumericTensorElement>(x: T) -> T {
    switch self {
    case .sin:
      T(Foundation.sin(x.toFloat()))
    case .cos:
      T(Foundation.cos(x.toFloat()))
    case .minusSin:
      T(-Foundation.sin(x.toFloat()))
    case .exp:
      T(Foundation.exp(x.toFloat()))
    case .log:
      T(Foundation.log(x.toFloat()))
    case .recip:
      T(1 / x.toFloat())
    case .sigmoid:
      T(safeSigmoid(x.toFloat()))
    case .sigmoidGrad:
      T(safeSigmoid(x.toFloat()) * safeSigmoid(-x.toFloat()))
    case .relu:
      x < T(0.0) ? T(0.0) : x
    case .reluGrad:
      x < T(0.0) ? T(0.0) : T(1.0)
    }
  }
}

private func safeSigmoid(_ x: Float) -> Float {
  if x < -20 {
    0
  } else if x > 20 {
    1
  } else {
    1 / (1 + exp(-x))
  }
}

extension Tensor {
  public func elemwise(op: ElemwiseOp, grad gradOp: ElemwiseOp? = nil) -> Tensor {
    let backend = Backend.current
    let newData = Task {
      try await backend.elemwise(
        try await self.data, op: op, count: shape.product(), dtype: dtype)
    }
    if let gradOp = gradOp, needsGrad && Tensor.isGradEnabled {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        handle.backward(backend.use { grad * self.noGrad().elemwise(op: gradOp) })
      }
    } else {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    }
  }

  public func sin() -> Tensor {
    self.elemwise(op: .sin, grad: .cos)
  }

  public func cos() -> Tensor {
    self.elemwise(op: .cos, grad: .minusSin)
  }

  public func exp() -> Tensor {
    self.elemwise(op: .exp, grad: .exp)
  }

  public func log() -> Tensor {
    self.elemwise(op: .log, grad: .recip)
  }

  public func sigmoid() -> Tensor {
    self.elemwise(op: .sigmoid, grad: .sigmoidGrad)
  }

  public func relu() -> Tensor {
    self.elemwise(op: .relu, grad: .reluGrad)
  }

  public func tanh() -> Tensor {
    2 * (2 * self).sigmoid() - 1
  }

  public func gelu() -> Tensor {
    0.5 * self * (1 + (0.797884561 * (self + 0.044715 * self.pow(3))).tanh())
  }

  public func silu() -> Tensor {
    return self * self.sigmoid()
  }

  public func sqrt() -> Tensor {
    pow(0.5)
  }

  public func rsqrt() -> Tensor {
    pow(-0.5)
  }
}
