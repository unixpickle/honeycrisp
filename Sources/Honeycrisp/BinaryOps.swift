/// A binary operator which can be performed on numeric ``Tensor``s.
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
