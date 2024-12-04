import HCBacktrace

/// A value which can be used as a bit pattern for bitwise operators on a ``Tensor``.
public protocol TensorElementBitPattern {
  init(_: Int)
}

extension UInt8: TensorElementBitPattern {
}

extension UInt16: TensorElementBitPattern {
}

extension UInt32: TensorElementBitPattern {
}

extension UInt64: TensorElementBitPattern {
}

extension TensorElementBitPattern {
  public var bitsForBitwiseOp: [UInt8] {
    var x = self
    return withUnsafeBytes(of: &x) { data in (0..<MemoryLayout<Self>.size).map { data[$0] } }
  }

  public init(fromBitwiseOpBits bits: [UInt8]) {
    var x: Self = Self(0)
    withUnsafeMutableBytes(of: &x) { out in
      for (i, x) in bits.enumerated() {
        out[i] = x
      }
    }
    self = x
  }
}

/// A bitwise operator which can be performed on a pair of ``Tensor``s.
public enum BitwiseOp {
  case or
  case and
  case xor

  public func apply<T: TensorElementBitPattern>(_ a: T, _ b: T) -> T {
    T(
      fromBitwiseOpBits: zip(a.bitsForBitwiseOp, b.bitsForBitwiseOp).map { x, y in
        switch self {
        case .or:
          x | y
        case .and:
          x & y
        case .xor:
          x ^ y
        }
      })
  }
}

extension Tensor {

  @recordCaller
  internal static func _bitwise(lhs: Tensor, rhs: Tensor, op: BitwiseOp) -> Tensor {
    alwaysAssert(
      lhs.dtype == rhs.dtype,
      "dtypes for bitwise operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let (newShape, (lhsStrides, rhsStrides)) = Tensor.lazyBroadcast(lhs, rhs)

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.bitwiseOp(
        BroadcastData(strides: lhsStrides, data: try await lhs.data),
        BroadcastData(strides: rhsStrides, data: try await rhs.data),
        op: op,
        dtype: lhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: newShape, dtype: lhs.dtype)
  }

  @recordCaller
  internal static func _bitwise<T: TensorElementBitPattern>(lhs: Tensor, rhs: T, op: BitwiseOp)
    -> Tensor
  {
    alwaysAssert(
      rhs.bitsForBitwiseOp.count == lhs.dtype.byteSize,
      "dtype \(lhs.dtype) cannot be used with scalar type \(T.self) in bitwise operations because they are different sizes"
    )
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.bitwiseOp(
        try await lhs.data, rhs, op: op, count: lhs.shape.product(), dtype: lhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
  }

  @recordCaller
  internal static func _bitwise<T: TensorElementBitPattern>(lhs: T, rhs: Tensor, op: BitwiseOp)
    -> Tensor
  {
    alwaysAssert(
      lhs.bitsForBitwiseOp.count == rhs.dtype.byteSize,
      "dtype \(rhs.dtype) cannot be used with scalar type \(T.self) in bitwise operations because they are different sizes"
    )
    let backend = Backend.current
    let newData = createDataTask(rhs) { rhs in
      try await backend.bitwiseOp(
        try await rhs.data, lhs, op: op, count: rhs.shape.product(), dtype: rhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: rhs.shape, dtype: rhs.dtype)
  }

  /*
  for op, name in [
    ("^", "xor"),
    ("|", "or"),
    ("&", "and"),
  ]:
    print(
        f"""
  public static func {op} <T: TensorElementBitPattern>(lhs: Tensor, rhs: T) -> Tensor {{
    bitwise(lhs: lhs, rhs: rhs, op: .{name})
  }}

  public static func {op} <T: TensorElementBitPattern>(lhs: T, rhs: Tensor) -> Tensor {{
    bitwise(lhs: lhs, rhs: rhs, op: .{name})
  }}

  public static func {op} (lhs: Tensor, rhs: Tensor) -> Tensor {{
    bitwise(lhs: lhs, rhs: rhs, op: .{name})
  }}
        """
    )
  */

  public static func ^ <T: TensorElementBitPattern>(lhs: Tensor, rhs: T) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .xor)
  }

  public static func ^ <T: TensorElementBitPattern>(lhs: T, rhs: Tensor) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .xor)
  }

  public static func ^ (lhs: Tensor, rhs: Tensor) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .xor)
  }

  public static func | <T: TensorElementBitPattern>(lhs: Tensor, rhs: T) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .or)
  }

  public static func | <T: TensorElementBitPattern>(lhs: T, rhs: Tensor) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .or)
  }

  public static func | (lhs: Tensor, rhs: Tensor) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .or)
  }

  public static func & <T: TensorElementBitPattern>(lhs: Tensor, rhs: T) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .and)
  }

  public static func & <T: TensorElementBitPattern>(lhs: T, rhs: Tensor) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .and)
  }

  public static func & (lhs: Tensor, rhs: Tensor) -> Tensor {
    bitwise(lhs: lhs, rhs: rhs, op: .and)
  }

}
