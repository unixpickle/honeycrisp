import Foundation
import HCBacktrace

/// A flag which determines the implementation behind ``Tensor/gelu(mode:function:file:line:)``.
///
/// When the mode is approx, a tanh-based approximation is used.
public enum GeLUMode: Sendable {
  case approx
  case exact
}

/// An element-wise operation which can be applied to a numeric ``Tensor``.
public enum ElemwiseOp: Sendable {
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
  case geluApprox
  case geluApproxGrad
  case geluExact
  case geluExactGrad
  case erf
  case erfGrad
  case floor
  case ceil
  case round

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
    case .geluApprox:
      T(geluApproxImpl(f))
    case .geluApproxGrad:
      T(geluApproxGradImpl(f))
    case .geluExact:
      T(geluExactImpl(f))
    case .geluExactGrad:
      T(geluExactGradImpl(f))
    case .erf:
      T(fastErf(f))
    case .erfGrad:
      T(simpleErfGrad(f))
    case .floor:
      T(f.rounded(.down))
    case .ceil:
      T(f.rounded(.up))
    case .round:
      T(f.rounded())
    }
  }
}

private func geluApproxImpl(_ f: Float) -> Float {
  0.5 * f * (1 + safeTanh(0.797884561 * (f + 0.044715 * pow(f, 3))))
}

private func geluApproxGradImpl(_ f: Float) -> Float {
  let tanhTerm = tanh(0.035677408145115 * pow(f, 3) + 0.797884561 * f)
  return 0.5 * f * (1 - pow(tanhTerm, 2)) * (0.107032224435345 * pow(f, 2) + 0.797884561)
    + 0.5 * tanhTerm + 0.5
}

private func geluExactImpl(_ f: Float) -> Float {
  return f * 0.5 * (1 + fastErf(f * 0.7071067811865475))
}

private func geluExactGradImpl(_ f: Float) -> Float {
  let c: Float = 0.7071067811865475
  let term1 = 0.5 * (1 + fastErf(f * c))
  let term2 = 0.5 * f * c * simpleErfGrad(f * c)
  return term1 + term2
}

private func simpleErfGrad(_ f: Float) -> Float {
  abs(f) > 20 ? 0 : 1.1283791670955126 * exp(-f * f)
}

private func fastErf(_ a: Float) -> Float {
  // https://github.com/ml-explore/mlx/blob/0d5e7716ad0adadae215ece6eb70861a6a8b55a3/mlx/backend/common/ops.h#L47
  var r: Float
  var s: Float
  var t: Float
  var u: Float
  t = abs(a)
  s = a * a
  func fma(_ x: Float, _ y: Float, _ z: Float) -> Float {
    return x * y + z
  }
  if t > 0.927734375 {
    // maximum error 0.99527 ulp
    r = fma(
      -1.72853470e-5, t, 3.83197126e-4)  // -0x1.220000p-16,0x1.91cfb2p-12
    u = fma(
      -3.88396438e-3, t, 2.42546219e-2)  // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = fma(r, s, u)
    r = fma(r, t, -1.06777877e-1)  // -0x1.b55cb8p-4
    r = fma(r, t, -6.34846687e-1)  // -0x1.450aa0p-1
    r = fma(r, t, -1.28717512e-1)  // -0x1.079d0cp-3
    r = fma(r, t, -t)
    r = 1.0 - exp(r)
    r = copysign(r, a)
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4  // -0x1.38e000p-11
    r = fma(r, s, 4.99119423e-3)  //  0x1.471a58p-8
    r = fma(r, s, -2.67681349e-2)  // -0x1.b691b2p-6
    r = fma(r, s, 1.12819925e-1)  //  0x1.ce1c44p-4
    r = fma(r, s, -3.76125336e-1)  // -0x1.812700p-2
    r = fma(r, s, 1.28379166e-1)  //  0x1.06eba8p-3
    r = fma(r, a, a)
  }
  return r
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
}

extension Tensor {
  @recordCaller
  private func _elemwise(op: ElemwiseOp, grad gradOp: ElemwiseOp? = nil) -> Tensor {
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.elemwise(
        try await t.data, op: op, scales: nil, count: t.shape.product(), dtype: t.dtype)
    }
    if needsGrad && Tensor.isGradEnabled {
      guard let gradOp = gradOp else {
        tracedFatalError("no gradient operation was specified")
      }
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        handle.backward(backend) { self.noGrad().elemwiseGrad(op: gradOp, grad: grad) }
      }
    } else {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    }
  }

  @recordCaller
  internal func _elemwiseGrad(op: ElemwiseOp, grad: Tensor) -> Tensor {
    #alwaysAssert(!self.needsGrad && !grad.needsGrad, "second derivatives are not supported")
    let backend = Backend.current
    let newData = Tensor.createDataTask(self, grad) { t, grad in
      try await backend.elemwise(
        try await t.data, op: op, scales: try await grad.data, count: t.shape.product(),
        dtype: t.dtype)
    }
    return Tensor(dataTask: newData, shape: shape, dtype: dtype)
  }

  @recordCaller
  private func _sin() -> Tensor {
    self.elemwise(op: .sin, grad: .cos)
  }

  @recordCaller
  private func _cos() -> Tensor {
    self.elemwise(op: .cos, grad: .minusSin)
  }

  @recordCaller
  private func _exp() -> Tensor {
    self.elemwise(op: .exp, grad: .exp)
  }

  @recordCaller
  private func _log() -> Tensor {
    self.elemwise(op: .log, grad: .recip)
  }

  @recordCaller
  private func _sigmoid() -> Tensor {
    self.elemwise(op: .sigmoid, grad: .sigmoidGrad)
  }

  @recordCaller
  private func _relu() -> Tensor {
    self.elemwise(op: .relu, grad: .reluGrad)
  }

  @recordCaller
  private func _abs() -> Tensor {
    self.elemwise(op: .abs, grad: .absGrad)
  }

  @recordCaller
  private func _tanh() -> Tensor {
    2 * (2 * self).sigmoid() - 1
  }

  @recordCaller
  private func _gelu(mode: GeLUMode = .approx) -> Tensor {
    switch mode {
    case .approx:
      self.elemwise(op: .geluApprox, grad: .geluApproxGrad)
    case .exact:
      self.elemwise(op: .geluExact, grad: .geluExactGrad)
    }
  }

  @recordCaller
  private func _erf() -> Tensor {
    self.elemwise(op: .erf, grad: .erfGrad)
  }

  @recordCaller
  private func _silu() -> Tensor {
    return self * self.sigmoid()
  }

  @recordCaller
  private func _sqrt() -> Tensor {
    pow(0.5)
  }

  @recordCaller
  private func _rsqrt() -> Tensor {
    pow(-0.5)
  }

  @recordCaller
  private func _floor() -> Tensor {
    noGrad().elemwise(op: .floor)
  }

  @recordCaller
  private func _ceil() -> Tensor {
    noGrad().elemwise(op: .ceil)
  }

  @recordCaller
  private func _round() -> Tensor {
    noGrad().elemwise(op: .round)
  }
}
