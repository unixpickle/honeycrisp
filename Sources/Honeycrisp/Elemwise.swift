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
