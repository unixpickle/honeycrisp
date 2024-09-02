// Borrowed from https://forums.developer.apple.com/forums/thread/93282

import Accelerate
import Foundation

enum ConversionError: Error {
  case vImageError(String)
}

public protocol TensorElement {
  static var isFloatLossy: Bool { get }
  static var dtype: Tensor.DType { get }

  init(_ value: Float)
  init(_ value: Int64)
  func toFloat() -> Float
  func toInt64() -> Int64

  static func == (lhs: Self, rhs: Self) -> Bool
}

public protocol NumericTensorElement: TensorElement {
  func pow<T: TensorElement>(_ exponent: T) -> Self

  static func + (lhs: Self, rhs: Self) -> Self
  static func * (lhs: Self, rhs: Self) -> Self
  prefix static func - (t: Self) -> Self
  static func - (lhs: Self, rhs: Self) -> Self
  static func / (lhs: Self, rhs: Self) -> Self
}

extension Double: NumericTensorElement {
  public static var isFloatLossy: Bool { false }
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
}

extension Int: NumericTensorElement {
  public static var isFloatLossy: Bool { false }
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
}

extension Float: NumericTensorElement {
  public static var isFloatLossy: Bool { false }
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
}

extension Bool: TensorElement {
  public static var isFloatLossy: Bool { false }
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
}

extension Int64: NumericTensorElement {
  public static var isFloatLossy: Bool { true }
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
