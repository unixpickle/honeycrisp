public enum NumericBinaryOp {
  case add
  case mul
  case sub
  case div
  case mod

  public func apply<T: NumericTensorElement>(_ a: T, _ b: T) -> T {
    switch self {
    case .add:
      a + b
    case .mul:
      a * b
    case .sub:
      a - b
    case .div:
      a / b
    case .mod:
      T.modulus(a, b)
    }
  }
}

public enum ComparisonOp {
  case equal
  case less
  case lessEqual
  case greater
  case greaterEqual

  public func apply<T: TensorElement>(_ a: T, _ b: T) -> Bool {
    switch self {
    case .equal:
      a == b
    case .less:
      a < b
    case .lessEqual:
      a <= b
    case .greater:
      a > b
    case .greaterEqual:
      a >= b
    }
  }
}
