// Borrowed from https://forums.developer.apple.com/forums/thread/93282

import Accelerate
import Foundation

enum ConversionError: Error {
  case vImageError(String)
}

/// A native numeric or boolean value which can represent a single element of a ``Tensor``.
public protocol TensorElement: Comparable {
  /// If true, then ``TensorElement/toFloat()`` may lose information.
  static var isFloatLossy: Bool { get }

  /// If true, then ``TensorElement/toInt64()`` may lose information.
  static var isInt64Lossy: Bool { get }

  /// If false, then ``TensorElement/toInt64()`` will return `0` or `1`.
  static var isBoolLossy: Bool { get }

  /// Get the default tensor datatype for this scalar type.
  static var dtype: Tensor.DType { get }

  init(_ value: Float)
  init(_ value: Int64)

  /// Obtain a floating-point representation of this value.
  func toFloat() -> Float

  /// Obtain an integer representation of this value, possibly rounding.
  func toInt64() -> Int64

  static func == (lhs: Self, rhs: Self) -> Bool
}

/// A ``TensorElement`` which supports numerical operations.
public protocol NumericTensorElement: TensorElement, Strideable {
  func pow<T: TensorElement>(_ exponent: T) -> Self

  static func + (lhs: Self, rhs: Self) -> Self
  static func * (lhs: Self, rhs: Self) -> Self
  prefix static func - (t: Self) -> Self
  static func - (lhs: Self, rhs: Self) -> Self
  static func / (lhs: Self, rhs: Self) -> Self
  static func modulus(_ lhs: Self, _ rhs: Self) -> Self
}

extension Double: NumericTensorElement {
  public static var isInt64Lossy: Bool { true }
  public static var isFloatLossy: Bool { false }
  public static var isBoolLossy: Bool { true }
  public static var dtype: Tensor.DType { .float32 }

  public func toFloat() -> Float {
    return Float(self)
  }

  public func toInt64() -> Int64 {
    return Int64(self)
  }

  public func pow<T: TensorElement>(_ exponent: T) -> Double {
    return Foundation.pow(self, Double(exponent.toFloat()))
  }

  public static func modulus(_ lhs: Self, _ rhs: Self) -> Self {
    if rhs < 0 {
      -modulus(-lhs, -rhs)
    } else if lhs < 0 {
      fmod(rhs - fmod(rhs - lhs, rhs), rhs)
    } else {
      fmod(lhs, rhs)
    }
  }
}

extension Int: NumericTensorElement {
  public static var isInt64Lossy: Bool { false }
  public static var isFloatLossy: Bool { false }
  public static var isBoolLossy: Bool { true }
  public static var dtype: Tensor.DType { .int64 }

  public func toFloat() -> Float {
    return Float(self)
  }

  public func toInt64() -> Int64 {
    return Int64(self)
  }

  public func pow<T: TensorElement>(_ exponent: T) -> Int {
    return Int(Foundation.pow(Double(self), Double(exponent.toFloat())))
  }

  public static func modulus(_ lhs: Self, _ rhs: Self) -> Self {
    if rhs < 0 {
      -modulus(-lhs, -rhs)
    } else if lhs < 0 {
      (rhs - ((rhs - lhs) % rhs)) % rhs
    } else {
      lhs % rhs
    }
  }
}

extension Float: NumericTensorElement {
  public static var isInt64Lossy: Bool { true }
  public static var isFloatLossy: Bool { false }
  public static var isBoolLossy: Bool { true }
  public static var dtype: Tensor.DType { .float32 }

  public func toFloat() -> Float {
    return self
  }

  public func toInt64() -> Int64 {
    return Int64(self)
  }

  public func pow<T: TensorElement>(_ exponent: T) -> Float {
    return Foundation.pow(self, exponent.toFloat())
  }

  public static func modulus(_ lhs: Self, _ rhs: Self) -> Self {
    if rhs < 0 {
      -modulus(-lhs, -rhs)
    } else if lhs < 0 {
      fmod(rhs - fmod(rhs - lhs, rhs), rhs)
    } else {
      fmod(lhs, rhs)
    }
  }
}

extension Bool: TensorElement {
  public static var isInt64Lossy: Bool { false }
  public static var isFloatLossy: Bool { false }
  public static var isBoolLossy: Bool { false }
  public static var dtype: Tensor.DType { .bool }

  public init(_ value: Float) {
    self.init(value == 0 ? false : true)
  }

  public init(_ value: Int64) {
    self.init(value == 0 ? false : true)
  }

  public func toFloat() -> Float {
    return self ? 1 : 0
  }

  public func toInt64() -> Int64 {
    return self ? 1 : 0
  }

  public static func < (lhs: Self, rhs: Self) -> Bool {
    lhs == false && rhs == true
  }

  public static func <= (lhs: Self, rhs: Self) -> Bool {
    lhs == false
  }

  public static func >= (lhs: Self, rhs: Self) -> Bool {
    lhs == true
  }

  public static func > (lhs: Self, rhs: Self) -> Bool {
    lhs == true && rhs == false
  }
}

extension Int64: NumericTensorElement {
  public static var isInt64Lossy: Bool { false }
  public static var isFloatLossy: Bool { true }
  public static var isBoolLossy: Bool { true }
  public static var dtype: Tensor.DType { .int64 }

  public func toFloat() -> Float {
    return Float(self)
  }

  public func toInt64() -> Int64 {
    return self
  }

  public func pow<T: TensorElement>(_ exponent: T) -> Int64 {
    return Int64(Foundation.pow(Double(self), Double(exponent.toFloat())))
  }

  public static func modulus(_ lhs: Self, _ rhs: Self) -> Self {
    if rhs < 0 {
      -modulus(-lhs, -rhs)
    } else if lhs < 0 {
      (rhs - ((rhs - lhs) % rhs)) % rhs
    } else {
      lhs % rhs
    }
  }
}

func arrayToPointer<T: TensorElement>(
  _ input: [T], output: UnsafeMutableRawPointer, dtype: Tensor.DType
) throws {
  switch dtype {
  case .int64:
    arrayToPointer(input, UnsafeMutablePointer<Int64>(OpaquePointer(output)))
  case .bool:
    arrayToPointer(input, UnsafeMutablePointer<Bool>(OpaquePointer(output)))
  case .float32:
    arrayToPointer(input, UnsafeMutablePointer<Float>(OpaquePointer(output)))
  case .float16:
    convertFloatToHalf(input.map { $0.toFloat() }, output)
  }
}

private func arrayToPointer<A: TensorElement, B: TensorElement>(
  _ input: [A], _ output: UnsafeMutablePointer<B>
) {
  if A.self == B.self {
    input.withUnsafeBufferPointer({ srcBuf in
      UnsafeMutableRawPointer(output).copyMemory(
        from: UnsafeRawPointer(srcBuf.baseAddress!), byteCount: input.count * MemoryLayout<A>.stride
      )
    })
  } else {
    if A.isFloatLossy {
      for (i, x) in input.enumerated() {
        output[i] = B(x.toInt64())
      }
    } else {
      for (i, x) in input.enumerated() {
        output[i] = B(x.toFloat())
      }
    }
  }
}

func convertFloatToHalf(
  _ input: [Float], _ output: UnsafeMutableRawPointer
) {
  input.withUnsafeBufferPointer { inputPtr in
    var bufferFloat32 = vImage_Buffer(
      data: UnsafeMutableRawPointer(mutating: inputPtr.baseAddress), height: 1,
      width: UInt(input.count),
      rowBytes: input.count * 4)
    var bufferFloat16 = vImage_Buffer(
      data: output, height: 1, width: UInt(input.count), rowBytes: input.count * 2)

    if vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0) != kvImageNoError {
      print("Error converting float32 to float16")
    }
  }
}

func pointerToArray<T: TensorElement>(
  _ input: UnsafeMutableRawPointer, output: inout [T], dtype: Tensor.DType
) throws {
  switch dtype {
  case .bool:
    return pointerToArray(UnsafeMutablePointer<Bool>(OpaquePointer(input)), &output)
  case .float32:
    return pointerToArray(UnsafeMutablePointer<Float>(OpaquePointer(input)), &output)
  case .int64:
    return pointerToArray(UnsafeMutablePointer<Int64>(OpaquePointer(input)), &output)
  case .float16:
    var resultFloats = [Float](repeating: 0, count: output.count)
    try resultFloats.withUnsafeMutableBufferPointer { buffer in
      try convertHalfToFloat(
        input, UnsafeMutableRawPointer(OpaquePointer(buffer.baseAddress!)), count: output.count)
    }
    for (i, x) in resultFloats.enumerated() {
      output[i] = T(x)
    }
  }
}

private func pointerToArray<A: TensorElement, B: TensorElement>(
  _ input: UnsafeMutablePointer<A>, _ output: inout [B]
) {
  if A.self == B.self {
    let count = output.count
    output.withUnsafeMutableBufferPointer({ dstBuf in
      UnsafeMutableRawPointer(dstBuf.baseAddress!).copyMemory(
        from: UnsafeRawPointer(input), byteCount: count * MemoryLayout<A>.stride
      )
    })
  } else {
    if A.isFloatLossy {
      for i in 0..<output.count {
        output[i] = B(input[i].toInt64())
      }
    } else {
      for i in 0..<output.count {
        output[i] = B(input[i].toFloat())
      }
    }
  }
}

private func convertHalfToFloat(
  _ input: UnsafeMutableRawPointer, _ output: UnsafeMutableRawPointer, count: Int
) throws {
  var bufferFloat16 = vImage_Buffer(
    data: input, height: 1, width: UInt(count), rowBytes: count * 2)
  var bufferFloat32 = vImage_Buffer(
    data: output, height: 1, width: UInt(count),
    rowBytes: count * 4)

  let err = vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0)
  if err != kvImageNoError {
    throw ConversionError.vImageError("\(err)")
  }
}
