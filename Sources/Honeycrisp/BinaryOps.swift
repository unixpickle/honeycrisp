public enum BinaryOp {
  case add

  func apply<T: TensorElement>(_ a: [T], _ b: [T]) -> [T] {
    switch self {
    case .add:
      return Array(zip(a, b).map { $0 + $1 })
    }
  }
}
