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
  case abs
  case absGrad
  case gelu
  case geluGrad

  public func apply<T: NumericTensorElement>(_ x: T) -> T {
    let f = x.toFloat()
    return switch self {
    case .sin:
      T(Foundation.sin(f))
    case .cos:
      T(Foundation.cos(f))
    case .minusSin:
      T(-Foundation.sin(f))
    case .exp:
      T(Foundation.exp(f))
    case .log:
      T(Foundation.log(f))
    case .recip:
      T(1 / f)
    case .sigmoid:
      T(safeSigmoid(f))
    case .sigmoidGrad:
      T(safeSigmoid(f) * safeSigmoid(-f))
    case .relu:
      x < T(0.0) ? T(0.0) : x
    case .reluGrad:
      x < T(0.0) ? T(0.0) : T(1.0)
    case .abs:
      T(f < 0 ? -f : f)
    case .absGrad:
      T(f < 0 ? -1.0 : 1.0)
    case .gelu:
      T(0.5 * f * (1 + safeTanh(0.797884561 * (f + 0.044715 * pow(f, 3)))))
    case .geluGrad:
      T(
        {
          let tanhTerm = tanh(0.035677408145115 * pow(f, 3) + 0.797884561 * f)
          return 0.5 * f * (1 - pow(tanhTerm, 2)) * (0.107032224435345 * pow(f, 2) + 0.797884561)
            + 0.5 * tanhTerm + 0.5
        }())
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

private func safeTanh(_ x: Float) -> Float {
  2 * safeSigmoid(2 * x) - 1
  // if x < -20 {
  //   -1
  // } else if x > 20 {
  //   1
  // } else {
  //   tanh(x)
  // }
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
        handle.backward(backend) { grad * self.noGrad().elemwise(op: gradOp) }
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

  public func abs() -> Tensor {
    self.elemwise(op: .abs, grad: .absGrad)
  }

  public func tanh() -> Tensor {
    2 * (2 * self).sigmoid() - 1
  }

  public func gelu() -> Tensor {
    self.elemwise(op: .gelu, grad: .geluGrad)
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
