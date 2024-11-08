import Foundation
import HCBacktrace

public struct TensorState: Codable {
  public enum TensorData {
    case floats([Float])
    case ints([Int64])
    case bools([Bool])
  }

  public enum DecodeError: Error {
    case invalidDataSize
  }

  enum CodingKeys: String, CodingKey {
    case data
    case shape
    case dtype
  }

  public let data: TensorData
  public let shape: [Int]
  public let dtype: Tensor.DType

  public init(data: TensorData, shape: [Int], dtype: Tensor.DType) {
    self.data = data
    self.shape = shape
    self.dtype = dtype
  }

  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    let data = try values.decode(Data.self, forKey: .data)
    shape = try values.decode([Int].self, forKey: .shape)
    dtype = try values.decode(Tensor.DType.self, forKey: .dtype)
    if data.count % dtype.byteSize != 0 {
      throw DecodeError.invalidDataSize
    }
    self.data =
      switch dtype {
      case .bool:
        .bools(data.map { $0 != 0 })
      case .int64:
        .ints(
          data.withUnsafeBytes { $0.bindMemory(to: Int64.self).map { Int64(littleEndian: $0) } })
      case .float32, .float16:
        .floats(
          data.withUnsafeBytes {
            $0.bindMemory(to: UInt32.self).map { Float(bitPattern: UInt32(littleEndian: $0)) }
          })
      }
  }

  public func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    let data =
      switch data {
      case .bools(let x):
        Data(x.map { $0 ? 1 : 0 })
      case .ints(let x):
        x.map { $0.littleEndian }.withUnsafeBufferPointer { Data(buffer: $0) }
      case .floats(let x):
        x.map { $0.bitPattern.littleEndian }.withUnsafeBufferPointer { Data(buffer: $0) }
      }
    try values.encode(data, forKey: .data)
    try values.encode(shape, forKey: .shape)
    try values.encode(dtype, forKey: .dtype)
  }
}

extension Tensor {
  public convenience init(state: TensorState) {
    switch state.data {
    case .floats(let x):
      self.init(data: x, shape: state.shape, dtype: state.dtype)
    case .ints(let x):
      self.init(data: x, shape: state.shape, dtype: state.dtype)
    case .bools(let x):
      self.init(data: x, shape: state.shape, dtype: state.dtype)
    }
  }

  @recordCaller
  private func _state() async throws -> TensorState {
    switch self.dtype {
    case .float16, .float32:
      TensorState(data: .floats(try await floats()), shape: shape, dtype: dtype)
    case .int64:
      TensorState(data: .ints(try await int64s()), shape: shape, dtype: dtype)
    case .bool:
      TensorState(data: .bools(try await bools()), shape: shape, dtype: dtype)
    }
  }
}
