import HCBacktrace

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

extension Tensor {
  @recordCaller
  internal static func _compare(lhs: Tensor, rhs: Tensor, op: ComparisonOp) -> Tensor {
    alwaysAssert(
      lhs.dtype == rhs.dtype, "dtypes for == operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let (newShape, ((lhs, lhsStrides), (rhs, rhsStrides))) = Tensor.lazyBroadcast(lhs, rhs)

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.compare(
        BroadcastData(strides: lhsStrides, data: try await lhs.data),
        BroadcastData(strides: rhsStrides, data: try await rhs.data),
        op: op,
        count: newShape.product(),
        dtype: lhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: newShape, dtype: .bool)
  }

  @recordCaller
  internal static func _compare<T: TensorElement>(lhs: Tensor, rhs: T, op: ComparisonOp) -> Tensor {
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.compare(
        try await lhs.data, rhs, op: op, count: lhs.shape.product(), dtype: lhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: lhs.shape, dtype: .bool)
  }

  @recordCaller
  internal static func _compare<T: TensorElement>(lhs: T, rhs: Tensor, op: ComparisonOp) -> Tensor {
    let backend = Backend.current
    let newData = createDataTask(rhs) { rhs in
      try await backend.compare(
        lhs, try await rhs.data, op: op, count: rhs.shape.product(), dtype: rhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: rhs.shape, dtype: .bool)
  }

  /*
  for op, name in [
    ("==", "equal"),
    ("<", "less"),
    (">", "greater"),
    ("<=", "lessEqual"),
    (">=", "greaterEqual"),
  ]:
    print(
        f"""
  public static func {op} <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {{
    compare(lhs: lhs, rhs: rhs, op: .{name})
  }}

  public static func {op} <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {{
    compare(lhs: lhs, rhs: rhs, op: .{name})
  }}

  public static func {op} (lhs: Tensor, rhs: Tensor) -> Tensor {{
    compare(lhs: lhs, rhs: rhs, op: .{name})
  }}
        """
    )
  */

  public static func == <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .equal)
  }

  public static func == <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .equal)
  }

  public static func == (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .equal)
  }

  public static func < <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .less)
  }

  public static func < <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .less)
  }

  public static func < (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .less)
  }

  public static func > <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greater)
  }

  public static func > <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greater)
  }

  public static func > (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greater)
  }

  public static func <= <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .lessEqual)
  }

  public static func <= <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .lessEqual)
  }

  public static func <= (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .lessEqual)
  }

  public static func >= <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greaterEqual)
  }

  public static func >= <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greaterEqual)
  }

  public static func >= (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greaterEqual)
  }
}
