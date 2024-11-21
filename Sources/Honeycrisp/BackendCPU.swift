import Accelerate
import Foundation
import HCBacktrace

/// A ``Backend`` with CPU-only implementations of every operation.
///
/// This may use accelerated APIs to speed up computation, but all computations will still
/// be done on the CPU (rather than the GPU or the ANE).
open class CPUBackend: Backend {

  public class CPUData: Tensor.Data {
    public let data: UnsafeMutableRawPointer

    public init(byteCount: Int) throws {
      self.byteCount = byteCount
      self.data = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
      // TODO: throw an exception if there's an OOM?

      #if DEBUG
        // Fill the data with garbage to catch methods that assume
        // zero initialization.
        let bound = data.bindMemory(to: UInt8.self, capacity: byteCount)
        let noise = (0..<3).map({ _ in UInt8.random(in: 0...255) })
        for i in 0..<byteCount {
          bound[i] = noise[i % 3]
        }
      #endif
    }

    deinit {
      self.data.deallocate()
    }

    public let byteCount: Int

    public func onCPU<T>(_ fn: (_: UnsafeRawPointer) async throws -> T) async throws -> T {
      try await fn(data)
    }

    public func mutateOnCPU<T>(_ fn: (_: UnsafeMutableRawPointer) async throws -> T) async throws
      -> T
    {
      try await fn(data)
    }
  }

  public class NativeRandomGenerator: RandomGenerator {
    public let cpuBackend: CPUBackend

    public var backend: Backend {
      cpuBackend
    }

    init(cpuBackend: CPUBackend) {
      self.cpuBackend = cpuBackend
    }

    public func save() async throws -> Data {
      throw BackendError.notImplemented("save")
    }

    public func restore(_ x: Data) async throws {
      throw BackendError.notImplemented("restore")
    }

    public func seed(_ x: Int) async throws {
      throw BackendError.notImplemented("seed")
    }

    public func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) async throws
      -> any Tensor.Data
    {
      try await cpuBackend.withBuffers(count * dtype.byteSize) { buffer in
        try await cpuBackend.serialize {
          switch dist {
          case .uniform:
            let arr = (0..<count).map { _ in Float.random(in: 0..<1.0) }
            try arrayToPointer(arr, output: buffer, dtype: dtype)
          case .normal:
            let elCount = count / 2 + (count % 2)
            var results = [Float]()
            for _ in 0..<elCount {
              let u1 = Float.random(in: 1e-5..<1.0)
              let u2 = Float.random(in: 0..<1.0)
              let r = sqrt(-2 * log(u1))
              let phi = 2 * Float.pi * u2
              let z1 = r * cos(phi)
              let z2 = r * sin(phi)
              results.append(z1)
              if results.count < count {
                results.append(z2)
              }
            }
            try arrayToPointer(results, output: buffer, dtype: dtype)
          }
        }
      }
    }

    public func sample(count: Int, in range: Range<Int64>) async throws -> any Tensor.Data {
      try await cpuBackend.withBuffers(count * Tensor.DType.int64.byteSize) { buffer in
        try await cpuBackend.serialize {
          let ints = (0..<count).map { _ in Int64.random(in: range) }
          try arrayToPointer(ints, output: buffer, dtype: .int64)
        }
      }
    }
  }

  private static var _global = CPUBackend()
  public static var global: CPUBackend { CPUBackend._global }

  internal var worker = Backend.WorkerThread()

  internal func serialize<T>(_ work: @escaping () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
      worker.schedule {
        var result: Result<T, Error>?
        do {
          result = Result.success(try work())
        } catch {
          result = Result.failure(error)
        }
        let constResult = result!
        continuation.resume(with: constResult)
      }
    }
  }

  internal func allocate(_ byteCount: Int) async throws -> Tensor.Data {
    let data = try CPUData(byteCount: byteCount)
    return data
  }

  override public func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, a.data, b.data) { buffer, aBuf, bBuf in
      func apply<T: NumericTensorElement>(_: T.Type) async throws {
        try await serialize {
          if dtype == .float32 && (op != .mod) && a.isSimple && b.isSimple {
            let x = UnsafePointer<Float>(
              aBuf.bindMemory(to: Float.self, capacity: a.dataCount))
            let y = UnsafePointer<Float>(
              bBuf.bindMemory(to: Float.self, capacity: b.dataCount))
            let z = buffer.bindMemory(to: Float.self, capacity: count)
            switch op {
            case .add:
              vDSP_vadd(y, 1, x, 1, z, 1, vDSP_Length(count))
            case .mul:
              vDSP_vmul(y, 1, x, 1, z, 1, vDSP_Length(count))
            case .div:
              vDSP_vdiv(y, 1, x, 1, z, 1, vDSP_Length(count))
            case .sub:
              vDSP_vsub(y, 1, x, 1, z, 1, vDSP_Length(count))
            case .mod:
              fatalError()
            }
          } else {
            try readBuffer(T.self, aBuf, count: a.dataCount, dtype: dtype) { aData in
              try readBuffer(T.self, bBuf, count: b.dataCount, dtype: dtype) { bData in
                try writeBuffer(T.self, buffer, count: count, dtype: dtype) { cData in
                  for i in 0..<count {
                    cData[i] = op.apply(
                      aData[a.strides(i)], bData[b.strides(i)])
                  }
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64.self)
      } else {
        try await apply(Float.self)
      }
    }
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, a) { buffer, aBuf in
      func apply<T1: NumericTensorElement>(_ b: T1) async throws {
        try await serialize {
          if dtype == .float32 && (op != .mod) {
            let x = UnsafePointer<Float>(aBuf.bindMemory(to: Float.self, capacity: count))
            var bScalar =
              switch op {
              case .add, .mul: b.toFloat()
              case .div: 1 / b.toFloat()
              case .sub: -b.toFloat()
              case .mod: fatalError()
              }
            let z = buffer.bindMemory(to: Float.self, capacity: count)
            switch op {
            case .add, .sub:
              vDSP_vsadd(x, 1, &bScalar, z, 1, vDSP_Length(count))
            case .mul, .div:
              vDSP_vsmul(x, 1, &bScalar, z, 1, vDSP_Length(count))
            case .mod:
              fatalError()
            }
          } else {
            try readBuffer(T1.self, aBuf, count: count, dtype: dtype) { aData in
              try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { cData in
                for (i, x) in aData.enumerated() {
                  cData[i] = op.apply(x, b)
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(b.toInt64())
      } else {
        try await apply(b.toFloat())
      }
    }
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, b) { buffer, bBuf in
      func apply<T1: NumericTensorElement>(_ a: T1) async throws {
        try await serialize {
          if dtype == .float32 && (op != .mod) {
            let x = UnsafePointer<Float>(bBuf.bindMemory(to: Float.self, capacity: count))
            var aFloat = a.toFloat()
            var neg1 = Float(-1)
            let z = buffer.bindMemory(to: Float.self, capacity: count)
            switch op {
            case .add:
              vDSP_vsadd(x, 1, &aFloat, z, 1, vDSP_Length(count))
            case .mul:
              vDSP_vsmul(x, 1, &aFloat, z, 1, vDSP_Length(count))
            case .div:
              vDSP_svdiv(&aFloat, x, 1, z, 1, vDSP_Length(count))
            case .sub:
              vDSP_vsmsa(x, 1, &neg1, &aFloat, z, 1, vDSP_Length(count))
            case .mod:
              fatalError()
            }
          } else {
            try readBuffer(T1.self, bBuf, count: count, dtype: dtype) { bData in
              try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { cData in
                for (i, x) in bData.enumerated() {
                  cData[i] = op.apply(a, x)
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(a.toInt64())
      } else {
        try await apply(a.toFloat())
      }
    }
  }

  override public func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, count: Int, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, a.data, b.data) { buffer, aBuf, bBuf in
      try await serialize {
        let x = aBuf.bindMemory(to: UInt8.self, capacity: a.dataCount * dtype.byteSize)
        let y = bBuf.bindMemory(to: UInt8.self, capacity: b.dataCount * dtype.byteSize)
        let z = buffer.bindMemory(to: UInt8.self, capacity: count)
        for i in 0..<(count * dtype.byteSize) {
          let aIdx = a.strides(i / dtype.byteSize) * dtype.byteSize + i % dtype.byteSize
          let bIdx = b.strides(i / dtype.byteSize) * dtype.byteSize + i % dtype.byteSize
          z[i] = op.apply(x[aIdx], y[bIdx])
        }
      }
    }
  }

  override public func bitwiseOp<T: TensorElementBitPattern>(
    _ a: Tensor.Data, _ b: T, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, a) { buffer, aBuf in
      let operandBytes = b.bitsForBitwiseOp
      try await serialize {
        let inp = aBuf.bindMemory(to: UInt8.self, capacity: count)
        let out = buffer.bindMemory(to: UInt8.self, capacity: count)
        for i in 0..<(count * dtype.byteSize) {
          out[i] = op.apply(inp[i], operandBytes[i % operandBytes.count])
        }
      }
    }
  }

  override public func bitwiseOp<T: TensorElementBitPattern>(
    _ a: T, _ b: Tensor.Data, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, b) { buffer, bBuf in
      let operandBytes = a.bitsForBitwiseOp
      try await serialize {
        let inp = bBuf.bindMemory(to: UInt8.self, capacity: count)
        let out = buffer.bindMemory(to: UInt8.self, capacity: count)
        for i in 0..<(count * dtype.byteSize) {
          out[i] = op.apply(operandBytes[i % operandBytes.count], inp[i])
        }
      }
    }
  }

  override public func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, input.data, coeff.data, bias.data) {
      buffer, inBuf, coeffBuf, biasBuf in
      func apply<T1: NumericTensorElement>(_: T1.Type) async throws {
        try await serialize {
          if dtype == .float32 && input.isSimple && coeff.isSimple && bias.isSimple {
            let x = UnsafePointer<Float>(
              inBuf.bindMemory(to: Float.self, capacity: count))
            let coeff = UnsafePointer<Float>(
              coeffBuf.bindMemory(to: Float.self, capacity: count))
            let bias = UnsafePointer<Float>(
              biasBuf.bindMemory(to: Float.self, capacity: count))
            let output = buffer.bindMemory(to: Float.self, capacity: count)
            vDSP_vma(x, 1, coeff, 1, bias, 1, output, 1, vDSP_Length(count))
          } else {
            try readBuffer(T1.self, inBuf, count: input.dataCount, dtype: dtype) {
              inData in
              try readBuffer(T1.self, coeffBuf, count: coeff.dataCount, dtype: dtype) {
                coeffData in
                try readBuffer(T1.self, biasBuf, count: bias.dataCount, dtype: dtype) {
                  biasData in
                  try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { outData in
                    for i in 0..<count {
                      outData[i] =
                        inData[input.strides(i)] * coeffData[coeff.strides(i)]
                        + biasData[bias.strides(i)]
                    }
                  }
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64.self)
      } else {
        try await apply(Float.self)
      }
    }
  }

  override public func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, input.data, coeff.data, bias.data) {
      buffer, inBuf, coeffBuf, biasBuf in
      func apply<T1: NumericTensorElement>(_: T1.Type) async throws {
        try await serialize {
          if dtype == .float32 && input.isSimple && coeff.isSimple && bias.isSimple {
            let x = UnsafePointer<Float>(
              inBuf.bindMemory(to: Float.self, capacity: count))
            let coeff = UnsafePointer<Float>(
              coeffBuf.bindMemory(to: Float.self, capacity: count))
            let bias = UnsafePointer<Float>(
              biasBuf.bindMemory(to: Float.self, capacity: count))
            let output = buffer.bindMemory(to: Float.self, capacity: count)
            vDSP_vam(x, 1, bias, 1, coeff, 1, output, 1, vDSP_Length(count))
          } else {
            try readBuffer(T1.self, inBuf, count: input.dataCount, dtype: dtype) {
              inData in
              try readBuffer(T1.self, coeffBuf, count: coeff.dataCount, dtype: dtype) {
                coeffData in
                try readBuffer(T1.self, biasBuf, count: bias.dataCount, dtype: dtype) {
                  biasData in
                  try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { outData in
                    for i in 0..<count {
                      outData[i] =
                        (inData[input.strides(i)] + biasData[bias.strides(i)])
                        * coeffData[coeff.strides(i)]
                    }
                  }
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64.self)
      } else {
        try await apply(Float.self)
      }
    }
  }

  override public func normalize<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, epsilon: T, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    assert(dtype.isFloat)
    return try await withBuffers(count * dtype.byteSize, input.data, mean.data, variance.data) {
      buffer, inBuf, meanBuf, varianceBuf in
      try await serialize {
        try readBuffer(Float.self, inBuf, count: input.dataCount, dtype: dtype) {
          inData in
          try readBuffer(Float.self, meanBuf, count: mean.dataCount, dtype: dtype) {
            meanData in
            try readBuffer(Float.self, varianceBuf, count: variance.dataCount, dtype: dtype) {
              varianceData in
              try writeBuffer(Float.self, buffer, count: count, dtype: dtype) { outData in
                for i in 0..<count {
                  outData[i] =
                    (inData[input.strides(i)] - meanData[mean.strides(i)])
                    / sqrt(varianceData[variance.strides(i)] + epsilon.toFloat())
                }
              }
            }
          }
        }
      }
    }
  }

  override public func normalizeXGrad<T: TensorElement>(
    variance: BroadcastData, outGrad: BroadcastData, epsilon: T, sign: Float, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize, variance.data, outGrad.data) {
      buffer, varianceBuf, outGradBuf in
      try await serialize {
        try readBuffer(Float.self, varianceBuf, count: variance.dataCount, dtype: dtype) {
          varianceData in
          try readBuffer(Float.self, outGradBuf, count: outGrad.dataCount, dtype: dtype) {
            outGradData in
            try writeBuffer(Float.self, buffer, count: count, dtype: dtype) { outData in
              for i in 0..<count {
                outData[i] =
                  sign * outGradData[outGrad.strides(i)]
                  / sqrt(varianceData[variance.strides(i)] + epsilon.toFloat())
              }
            }
          }
        }
      }
    }
  }

  override public func normalizeVarianceGrad<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, outGrad: BroadcastData,
    epsilon: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(
      count * dtype.byteSize, input.data, mean.data, variance.data, outGrad.data
    ) { buffer, inBuf, meanBuf, varianceBuf, outGradBuf in
      try await serialize {
        try readBuffer(Float.self, inBuf, count: input.dataCount, dtype: dtype) {
          inputData in
          try readBuffer(Float.self, meanBuf, count: mean.dataCount, dtype: dtype) {
            meanData in
            try readBuffer(Float.self, varianceBuf, count: variance.dataCount, dtype: dtype) {
              varianceData in
              try readBuffer(Float.self, outGradBuf, count: outGrad.dataCount, dtype: dtype) {
                outGradData in
                try writeBuffer(Float.self, buffer, count: count, dtype: dtype) { outData in
                  for i in 0..<count {
                    outData[i] =
                      -0.5 * outGradData[outGrad.strides(i)]
                      * (inputData[input.strides(i)] - meanData[mean.strides(i)])
                      * Darwin.pow(varianceData[variance.strides(i)] + epsilon.toFloat(), -1.5)
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  override public func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * Tensor.DType.bool.byteSize, a.data, b.data) {
      buffer, aBuf, bBuf in
      let aStrides = a.strides
      let bStrides = b.strides

      func apply<T1: NumericTensorElement>(_: T1.Type) async throws {
        try await serialize {
          try readBuffer(T1.self, aBuf, count: aStrides.dataCount, dtype: dtype) { aData in
            try readBuffer(T1.self, bBuf, count: bStrides.dataCount, dtype: dtype) { bData in
              try writeBuffer(Bool.self, buffer, count: count, dtype: .bool) { cData in
                for i in 0..<count {
                  cData[i] = op.apply(aData[aStrides(i)], bData[bStrides(i)])
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64.self)
      } else {
        try await apply(Float.self)
      }
    }
  }

  override public func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * Tensor.DType.bool.byteSize, a) { buffer, aBuf in
      func apply<T1: NumericTensorElement>(_ b: T1) async throws {
        try await serialize {
          try readBuffer(T1.self, aBuf, count: count, dtype: dtype) { aData in
            try writeBuffer(Bool.self, buffer, count: count, dtype: .bool) { cData in
              for (i, x) in aData.enumerated() {
                cData[i] = op.apply(x, b)
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(b.toInt64())
      } else {
        try await apply(b.toFloat())
      }
    }
  }

  override public func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * Tensor.DType.bool.byteSize, b) { buffer, bBuf in
      func apply<T1: NumericTensorElement>(_ a: T1) async throws {
        try await serialize {
          try readBuffer(T1.self, bBuf, count: count, dtype: dtype) { bData in
            try writeBuffer(Bool.self, buffer, count: count, dtype: .bool) { cData in
              for (i, x) in bData.enumerated() {
                cData[i] = op.apply(a, x)
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(a.toInt64())
      } else {
        try await apply(a.toFloat())
      }
    }
  }

  override public func cast(
    _ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * outType.byteSize, a) { buffer, aBuf in
      func apply<T: TensorElement>(_: T.Type) async throws {
        try await serialize {
          var arr = [T](repeating: T(0.0), count: count)
          try pointerToArray(aBuf, output: &arr, dtype: inType)
          try arrayToPointer(arr, output: buffer, dtype: outType)
        }
      }
      if inType == .int64 {
        try await apply(Int64.self)
      } else {
        try await apply(Float.self)
      }
    }
  }

  override public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, scale: T, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withMaybeBuffer(scales) { scalesBuf in
      try await withBuffers(count * dtype.byteSize, a) { buffer, aBuf in
        func apply<T1: NumericTensorElement>(_ b: T1, _ scale: T1) async throws {
          try await serialize {
            if dtype == .float32 && (b == T1(2.0) || b == T1(-1.0) || b == T1(1.0)) {
              let x = UnsafePointer<Float>(
                aBuf.bindMemory(to: Float.self, capacity: count))
              let z = UnsafeMutablePointer<Float>(
                buffer.bindMemory(to: Float.self, capacity: count))
              var s = scale.toFloat()
              switch b {
              case T1(2.0):
                vDSP_vmul(x, 1, x, 1, z, 1, vDSP_Length(count))
                if s != 1.0 {
                  vDSP_vsmul(z, 1, &s, z, 1, vDSP_Length(count))
                }
              case T1(-1.0):
                vDSP_svdiv(&s, x, 1, z, 1, vDSP_Length(count))
              case T1(1.0):
                vDSP_vsmul(x, 1, &s, z, 1, vDSP_Length(count))
              default:
                fatalError()
              }
              if let scalesBuf = scalesBuf {
                let scalesPtr = UnsafePointer<Float>(
                  scalesBuf.bindMemory(to: Float.self, capacity: count))
                vDSP_vmul(z, 1, scalesPtr, 1, z, 1, vDSP_Length(count))
              }
            } else {
              try readBuffer(T1.self, aBuf, count: count, dtype: dtype) { arr in
                try maybeReadBuffer(T1.self, scalesBuf, count: count, dtype: dtype) { scalesArr in
                  try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { out in
                    for (i, x) in arr.enumerated() {
                      out[i] = scale * (scalesArr?[i] ?? T1(1.0)) * x.pow(b)
                    }
                  }
                }
              }
            }
          }
        }
        if dtype == .int64 {
          try await apply(b.toInt64(), scale.toInt64())
        } else {
          try await apply(b.toFloat(), scale.toFloat())
        }
      }
    }
  }

  override public func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(min != nil || max != nil, "cannot use clamp() without bounds")

    return try await withBuffers(count * dtype.byteSize, a) { buffer, aBuf in
      func apply<T1: NumericTensorElement>(_ min: T1?, _ max: T1?) async throws {
        try await serialize {
          try readBuffer(T1.self, aBuf, count: count, dtype: dtype) { arr in
            try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { out in
              if let max = max, let min = min {
                for (i, x) in arr.enumerated() {
                  out[i] = Swift.max(min, Swift.min(max, x))
                }
              } else if let max = max {
                for (i, x) in arr.enumerated() {
                  out[i] = Swift.min(max, x)
                }
              } else if let min = min {
                for (i, x) in arr.enumerated() {
                  out[i] = Swift.max(min, x)
                }
              } else {
                for (i, x) in arr.enumerated() {
                  out[i] = x
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(min?.toInt64(), max?.toInt64())
      } else {
        try await apply(min?.toFloat(), max?.toFloat())
      }
    }
  }

  override public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let outType = (op == .sum ? dtype : Tensor.DType.int64)
    return try await withBuffers(dims.outCount * outType.byteSize, a) { buffer, aBuf in
      if dtype == .float32 && op == .sum {
        for i in 0..<dims.outerCount {
          for j in 0..<dims.innerCount {
            let inPtr = UnsafePointer<Float>(
              aBuf.advanced(by: 4 * (j + i * dims.reduceCount * dims.innerCount))
                .bindMemory(
                  to: Float.self, capacity: dims.reduceCount))
            let y = UnsafeMutablePointer<Float>(
              buffer.advanced(by: 4 * (i * dims.innerCount + j)).bindMemory(
                to: Float.self, capacity: dims.reduceCount))
            vDSP_sve(inPtr, vDSP_Stride(dims.innerCount), y, vDSP_Length(dims.reduceCount))
          }
        }
      }

      func apply<T: NumericTensorElement>(_: T.Type) async throws {
        switch op {
        case .sum:
          try await serialize {
            try readBuffer(T.self, aBuf, count: dims.inCount, dtype: dtype) { arr in
              try writeBuffer(T.self, buffer, count: dims.outCount, dtype: dtype) { arrOut in
                var index: Int = 0
                for i in 0..<dims.outerCount {
                  for j in 0..<dims.innerCount {
                    var sum = T(0.0)
                    for k in 0..<dims.reduceCount {
                      let item = arr[j + (k + i * dims.reduceCount) * dims.innerCount]
                      sum = sum + item
                    }
                    arrOut[index] = sum
                    index += 1
                  }
                }
              }
            }
          }
        case .argmin, .argmax:
          try await serialize {
            try readBuffer(T.self, aBuf, count: dims.inCount, dtype: dtype) { arr in
              try writeBuffer(Int64.self, buffer, count: dims.outCount, dtype: .int64) { arrOut in
                var outIndex: Int = 0
                for i in 0..<dims.outerCount {
                  for j in 0..<dims.innerCount {
                    var extremum = arr[j + i * dims.reduceCount * dims.innerCount]
                    var index = Int64(0)
                    for k in 0..<dims.reduceCount {
                      let item = arr[j + (k + i * dims.reduceCount) * dims.innerCount]
                      if op == .argmin {
                        if item < extremum {
                          extremum = item
                          index = Int64(k)
                        }
                      } else if op == .argmax {
                        if item > extremum {
                          extremum = item
                          index = Int64(k)
                        }
                      }
                    }
                    arrOut[outIndex] = index
                    outIndex += 1
                  }
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64.self)
      } else {
        try await apply(Float.self)
      }
    }
  }

  override public func logSoftmax(
    _ a: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let totalCount = dims.inCount
    return try await withBuffers(totalCount * dtype.byteSize, a) { buffer, aBuf in
      try await serialize {
        try readBuffer(Float.self, aBuf, count: totalCount, dtype: dtype) { arr in
          try writeBuffer(Float.self, buffer, count: totalCount, dtype: dtype) { arrOut in
            for i in 0..<dims.outerCount {
              let outerOffset = i * dims.innerCount * dims.reduceCount
              for j in 0..<dims.innerCount {
                var max: Float = 0
                for k in 0..<dims.reduceCount {
                  let item = arr[j + k * dims.innerCount + outerOffset]
                  if k == 0 || item > max {
                    max = item
                  }
                }
                var expSum: Float = 0
                for k in 0..<dims.reduceCount {
                  let item = arr[j + k * dims.innerCount + outerOffset]
                  expSum += exp(item - max)
                }
                let logSum = log(expSum) + max
                for k in 0..<dims.reduceCount {
                  let idx = j + k * dims.innerCount + outerOffset
                  let item = arr[idx]
                  arrOut[idx] = item - logSum
                }
              }
            }
          }
        }
      }
    }
  }

  override public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let totalCount = dims.inCount
    return try await withBuffers(totalCount * dtype.byteSize, a, outGrad) {
      buffer, aBuf, outGradBuf in
      try await serialize {
        try readBuffer(Float.self, aBuf, count: totalCount, dtype: dtype) { arr in
          try readBuffer(Float.self, outGradBuf, count: totalCount, dtype: dtype) { arrGrad in
            try writeBuffer(Float.self, buffer, count: totalCount, dtype: dtype) { arrOut in
              for i in 0..<dims.outerCount {
                let outerOffset = i * dims.innerCount * dims.reduceCount
                for j in 0..<dims.innerCount {
                  var max: Float = 0
                  var gradSum: Float = 0
                  for k in 0..<dims.reduceCount {
                    let idx = j + k * dims.innerCount + outerOffset
                    let item = arr[idx]
                    gradSum += arrGrad[idx]
                    if k == 0 || item > max {
                      max = item
                    }
                  }
                  var expSum: Float = 0
                  for k in 0..<dims.reduceCount {
                    let item = arr[j + k * dims.innerCount + outerOffset]
                    expSum += exp(item - max)
                  }
                  let logSum = log(expSum) + max
                  for k in 0..<dims.reduceCount {
                    let idx = j + k * dims.innerCount + outerOffset
                    let item = arr[idx]
                    let itemGrad = arrGrad[idx]
                    arrOut[idx] = itemGrad - gradSum * exp(item - logSum)
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  override public func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(dims.outCount * dtype.byteSize, a) { buffer, inData in
      try await serialize {
        let innerBytes = dtype.byteSize * dims.innerCount
        for i in 0..<dims.outerCount {
          for j in 0..<dims.repeatCount {
            let outBytes = buffer.advanced(
              by: (i * dims.repeatCount + j) * innerBytes)
            let inBytes = inData.advanced(by: i * innerBytes)
            outBytes.copyMemory(from: inBytes, byteCount: innerBytes)
          }
        }
      }
    }
  }

  override public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(s.gatherOutCount * dtype.byteSize, a, s.indices) {
      outData, inData, idxBuf in
      if s.broadcasted {
        let innerSize = s.innerCount * dtype.byteSize
        try await serialize {
          try readBuffer(Int64.self, idxBuf, count: s.indicesCount, dtype: .int64) {
            flatIndices in
            for i in 0..<s.outerCount {
              for (j, idx) in flatIndices.enumerated() {
                let source = inData.advanced(
                  by: i * s.middleCount * innerSize + Int(idx) * innerSize)
                let dst = outData.advanced(
                  by: i * flatIndices.count * innerSize + j * innerSize)
                dst.copyMemory(from: source, byteCount: innerSize)
              }
            }
          }
        }
        return
      }

      // Unbroadcasted case below

      func apply<T: TensorElement>(_ zero: T) async throws {
        try await serialize {
          try readBuffer(Int64.self, idxBuf, count: s.indicesCount, dtype: .int64) {
            flatIndices in
            try readBuffer(T.self, inData, count: s.gatherInCount, dtype: dtype) { inArr in
              try writeBuffer(T.self, outData, count: s.gatherOutCount, dtype: dtype) { outArr in
                for i in 0..<s.outerCount {
                  for j in 0..<s.outCount {
                    for k in 0..<s.innerCount {
                      let outIdx = i * s.outCount * s.innerCount + j * s.innerCount + k
                      let inIdx = Int(flatIndices[outIdx])
                      let source = inArr[
                        i * s.middleCount * s.innerCount + inIdx * s.innerCount + k]
                      outArr[outIdx] = source
                    }
                  }
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64(0))
      } else {
        try await apply(Float(0))
      }
    }
  }

  override public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(s.gatherInCount * dtype.byteSize, a, s.indices) { buffer, aBuf, idxBuf in
      func apply<T: NumericTensorElement>(_ zero: T) async throws {
        try await serialize {
          try readBuffer(Int64.self, idxBuf, count: s.indicesCount, dtype: .int64) {
            flatIndices in
            try readBuffer(T.self, aBuf, count: s.gatherOutCount, dtype: dtype) { inArr in
              var outArr = [T](repeating: zero, count: s.gatherInCount)
              for i in 0..<s.outerCount {
                for j in 0..<s.outCount {
                  for k in 0..<s.innerCount {
                    let inIdx = i * s.outCount * s.innerCount + j * s.innerCount + k
                    let indexIdx = s.broadcasted ? j : inIdx
                    let jOut = Int(flatIndices[indexIdx])
                    let outIdx = i * s.middleCount * s.innerCount + jOut * s.innerCount + k
                    outArr[outIdx] = outArr[outIdx] + inArr[inIdx]
                  }
                }
              }
              try arrayToPointer(outArr, output: buffer, dtype: dtype)
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64(0))
      } else {
        try await apply(Float(0))
      }
    }
  }

  override public func when<T>(
    _ mask: BroadcastData, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    func argData(_ x: TensorOrScalar<T>) async throws -> Tensor.Data {
      switch x {
      case .tensor(let t):
        return t.data
      case .scalar(let s, _):
        let data = try await allocate(dtype.byteSize)
        try await data.mutateOnCPU { out in
          try arrayToPointer([s], output: out, dtype: dtype)
        }
        return data
      }
    }

    return try await withBuffers(count * dtype.byteSize, mask.data, argData(a), argData(b)) {
      buffer, maskBuf, aData, bData in
      let maskStrides = mask.strides
      let aStrides = a.strides
      let bStrides = b.strides

      try await serialize {
        let bools = maskBuf.bindMemory(to: UInt8.self, capacity: count)
        for i in 0..<count {
          let outOff = i * dtype.byteSize
          if bools[maskStrides(i)] != 0 {
            buffer.advanced(by: outOff).copyMemory(
              from: aData.advanced(by: aStrides(i) * dtype.byteSize), byteCount: dtype.byteSize)
          } else {
            buffer.advanced(by: outOff).copyMemory(
              from: bData.advanced(by: bStrides(i) * dtype.byteSize), byteCount: dtype.byteSize)
          }
        }
      }
    }
  }

  override public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    return try await batchedMatmul(
      matrixCount: 1, a: a, transA: transA, b: b, transB: transB, transOut: transOut, rows: rows,
      inner: inner, cols: cols, dtype: dtype)
  }

  override public func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(matrixCount * rows * cols * dtype.byteSize, a, b) { buffer, aBuf, bBuf in
      let aCount = rows * inner
      let bCount = inner * cols
      func apply<T: NumericTensorElement>(_ zero: T) async throws {
        if !transA && !transB && !transOut && dtype == .float32 {
          try await serialize {
            for i in 0..<matrixCount {
              let x = UnsafePointer<Float>(
                aBuf.advanced(by: i * aCount * dtype.byteSize).bindMemory(
                  to: Float.self, capacity: aCount))
              let y = UnsafePointer<Float>(
                bBuf.advanced(by: i * bCount * dtype.byteSize).bindMemory(
                  to: Float.self, capacity: aCount))
              let z = buffer.advanced(by: i * rows * cols * dtype.byteSize).bindMemory(
                to: Float.self, capacity: rows * cols)
              vDSP_mmul(x, 1, y, 1, z, 1, vDSP_Length(rows), vDSP_Length(cols), vDSP_Length(inner))
            }
          }
        } else {
          try await serialize {
            try readBuffer(T.self, aBuf, count: matrixCount * aCount, dtype: dtype) { arrA in
              try readBuffer(T.self, bBuf, count: matrixCount * bCount, dtype: dtype) { arrB in
                try writeBuffer(T.self, buffer, count: matrixCount * rows * cols, dtype: dtype) {
                  arrC in

                  func getA(_ matIdx: Int, _ i: Int, _ j: Int) -> T {
                    if transA {
                      arrA[matIdx * rows * inner + i + j * rows]
                    } else {
                      arrA[matIdx * rows * inner + i * inner + j]
                    }
                  }

                  func getB(_ matIdx: Int, _ i: Int, _ j: Int) -> T {
                    if transB {
                      arrB[matIdx * cols * inner + i + j * inner]
                    } else {
                      arrB[matIdx * cols * inner + i * cols + j]
                    }
                  }

                  func setC(_ matIdx: Int, _ i: Int, _ j: Int, _ x: T) {
                    if transOut {
                      arrC[matIdx * rows * cols + i + j * rows] = x
                    } else {
                      arrC[matIdx * rows * cols + i * cols + j] = x
                    }
                  }

                  for matIdx in 0..<matrixCount {
                    for i in 0..<rows {
                      for j in 0..<cols {
                        var acc = T(0.0)
                        for k in 0..<inner {
                          acc = acc + getA(matIdx, i, k) * getB(matIdx, k, j)
                        }
                        setC(matIdx, i, j, acc)
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64(0))
      } else {
        try await apply(Float(0))
      }
    }
  }

  override public func tril(_ a: Tensor.Data, batch: Int, rows: Int, cols: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    let rowSize = cols * dtype.byteSize
    return try await withBuffers(batch * rows * rowSize, a) { outPtr, inPtr in
      try await serialize {
        for i in 0..<batch {
          for j in 0..<rows {
            let copyBytes = min(rowSize, (j + 1) * dtype.byteSize)
            let offset = (j + i * rows) * cols * dtype.byteSize
            outPtr.advanced(by: offset).copyMemory(
              from: inPtr.advanced(by: offset), byteCount: copyBytes)
            if copyBytes < rowSize {
              let zeroCount = rowSize - copyBytes
              let zeroStart = outPtr.advanced(by: offset + (cols * dtype.byteSize - zeroCount))
              zeroStart.initializeMemory(as: UInt8.self, repeating: 0, count: zeroCount)
            }
          }
        }
      }
    }
  }

  internal func convNd<Dim: SpatialDim>(
    _ config: ConvConfig<Dim>, batch: Int, image: Tensor.Data, kernel: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let imageShape = config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape = config.outputTensorShape(batch: batch)

    assert(kernel.byteCount >= dtype.byteSize * kernelShape.product())
    assert(image.byteCount >= dtype.byteSize * imageShape.product())

    return try await withBuffers(outShape.product() * dtype.byteSize, kernel, image) {
      outBuf, kernelBuf, imageBuf in
      func apply<T: NumericTensorElement>(_ zero: T) async throws {
        try await serialize {
          try readBuffer(T.self, kernelBuf, count: kernelShape.product(), dtype: dtype) {
            arrKernel in
            try readBuffer(T.self, imageBuf, count: imageShape.product(), dtype: dtype) {
              arrImage in
              try writeBuffer(T.self, outBuf, count: outShape.product(), dtype: dtype) {
                arrOut in
                let getKernel = ConvConfig<Dim>.LazyTensor(
                  from: arrKernel, shape: kernelShape, channelsLast: false)
                let getImage = config.lazy(from: arrImage, shape: imageShape)
                let outputFn = config.lazyForward(image: getImage, kernel: getKernel)
                config.unlazify(from: outputFn, to: &arrOut)
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64(0))
      } else {
        try await apply(Float(0))
      }
    }
  }

  internal func convNdTranspose<Dim: SpatialDim>(
    _ config: ConvConfig<Dim>, batch: Int, image: Tensor.Data, kernel: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let imageShape = config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape = config.outputTensorShape(batch: batch)

    assert(kernel.byteCount >= dtype.byteSize * kernelShape.product())
    assert(image.byteCount >= dtype.byteSize * outShape.product())

    return try await withBuffers(imageShape.product() * dtype.byteSize, kernel, image) {
      outBuf, kernelBuf, imageBuf in
      func apply<T: NumericTensorElement>(_ zero: T) async throws {
        try await serialize {
          try readBuffer(T.self, kernelBuf, count: kernelShape.product(), dtype: dtype) {
            arrKernel in
            try readBuffer(T.self, imageBuf, count: outShape.product(), dtype: dtype) {
              arrImage in
              try writeBuffer(T.self, outBuf, count: imageShape.product(), dtype: dtype) { arrOut in
                let getKernel = ConvConfig<Dim>.LazyTensor(
                  from: arrKernel, shape: kernelShape, channelsLast: false)
                let getImage = config.lazy(from: arrImage, shape: outShape)
                let outputFn = config.lazyTranspose(image: getImage, kernel: getKernel)
                config.unlazify(from: outputFn, to: &arrOut)
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64(0))
      } else {
        try await apply(Float(0))
      }
    }
  }

  internal func convNdKernelGrad<Dim: SpatialDim>(
    _ config: ConvConfig<Dim>, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let imageShape = config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape = config.outputTensorShape(batch: batch)

    return try await withBuffers(kernelShape.product() * dtype.byteSize, image, outGrad) {
      outBuf, imageBuf, outGradBuf in
      func apply<T: NumericTensorElement>(_ zero: T) async throws {
        try await serialize {
          try readBuffer(T.self, imageBuf, count: imageShape.product(), dtype: dtype) {
            arrImage in
            try readBuffer(T.self, outGradBuf, count: outShape.product(), dtype: dtype) {
              arrOutGrad in
              try writeBuffer(T.self, outBuf, count: kernelShape.product(), dtype: dtype) {
                arrOut in
                let getImage = config.lazy(from: arrImage, shape: imageShape)
                let getOutGrad = config.lazy(from: arrOutGrad, shape: outShape)
                let outputFn = config.lazyKernelGrad(image: getImage, outGrad: getOutGrad)
                outputFn.unlazify(to: &arrOut, channelsLast: false)
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(Int64(0))
      } else {
        try await apply(Float(0))
      }
    }
  }

  override public func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNd(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdTranspose(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdKernelGrad(config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override public func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNd(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdTranspose(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdKernelGrad(config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override public func elemwise(
    _ a: Tensor.Data, op: ElemwiseOp, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    return try await withMaybeBuffer(scales) { scalesBuf in
      try await withBuffers(count * dtype.byteSize, a) { buffer, aBuf in
        try await serialize {
          try readBuffer(Float.self, aBuf, count: count, dtype: dtype) { arr in
            try maybeReadBuffer(Float.self, scalesBuf, count: count, dtype: dtype) { scalesArr in
              try writeBuffer(Float.self, buffer, count: count, dtype: dtype) { out in
                for (i, x) in arr.enumerated() {
                  out[i] = op.apply(x) * (scalesArr?[i] ?? Float(1.0))
                }
              }
            }
          }
        }
      }
    }
  }

  override public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(inputs.count == innerCounts.count)
    let totalInner = innerCounts.sum()
    return try await withBuffers(outerCount * totalInner * dtype.byteSize, inputs) {
      buffer, inBufs in
      try await serialize {
        var outOffset = 0
        for i in 0..<outerCount {
          for (inBuf, innerCount) in zip(inBufs, innerCounts) {
            let chunkSize = innerCount * dtype.byteSize
            let outPtr = buffer.advanced(by: outOffset)
            let inPtr = inBuf.advanced(by: i * chunkSize)
            outPtr.copyMemory(from: inPtr, byteCount: chunkSize)
            outOffset += chunkSize
          }
        }
      }
    }
  }

  override public func constant<T: TensorElement>(_ value: T, count: Int, dtype: Tensor.DType)
    async throws -> Tensor.Data
  {
    try await withBuffers(count * dtype.byteSize) { buffer in
      func apply<T1: TensorElement>(_ value: T1) async throws {
        try await serialize {
          try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { out in
            for i in 0..<count {
              out[i] = value
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(value.toInt64())
      } else {
        try await apply(value.toFloat())
      }
    }
  }

  override public func collection<T: TensorElement>(
    _ collection: some Collection<T>, reverse: Bool, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    let count = collection.count
    return try await withBuffers(count * dtype.byteSize) { buffer in
      // This is a ~50x speed-up compared to the generic path, even in release mode.
      if let arr = (collection as? [T]), !reverse {
        try await serialize {
          try arrayToPointer(arr, output: buffer, dtype: dtype)
        }
        return
      }

      func apply<T1: TensorElement>(_ collection: some Sequence<T1>) async throws {
        try await serialize {
          try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { out in
            for (i, x) in collection.enumerated() {
              if reverse {
                out[count - (i + 1)] = x
              } else {
                out[i] = x
              }
            }
          }
        }
      }
      if dtype == .int64 {
        try await apply(collection.map { $0.toInt64() })
      } else {
        try await apply(collection.map { $0.toFloat() })
      }
    }
  }

  override public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data
  {
    try await withBuffers(shape.product() * Tensor.DType.int64.byteSize) { buffer in
      try await serialize {
        let oldStrides = stridesForShape(shape)
        let permutedStrides = permutation.map { oldStrides[$0] }
        let newShape = permutation.map { shape[$0] }
        let newStrides = stridesForShape(newShape)
        var newIndices = [Int](repeating: 0, count: shape.product())
        for i in 0..<newIndices.count {
          var flatIndex = 0
          for j in 0..<newStrides.count {
            flatIndex += permutedStrides[j] * ((i / newStrides[j]) % newShape[j])
          }
          newIndices[i] = flatIndex
        }
        try arrayToPointer(newIndices, output: buffer, dtype: .int64)
      }
    }
  }

  override public func defaultRandom() async throws -> RandomGenerator {
    NativeRandomGenerator(cpuBackend: self)
  }

  override public func createRandom() async throws -> RandomGenerator {
    NativeRandomGenerator(cpuBackend: self)
  }

  func withMaybeBuffer<T>(
    _ buf: Tensor.Data?, _ fn: (UnsafeRawPointer?) async throws -> T
  ) async throws -> T {
    if let buf = buf {
      try await buf.onCPU { ptr in
        try await fn(ptr)
      }
    } else {
      try await fn(nil)
    }
  }

  func withBuffers(
    _ outputSize: Int, _ fn: (UnsafeMutableRawPointer) async throws -> Void
  ) async throws -> Tensor.Data {
    let result = try await allocate(outputSize)
    try await result.mutateOnCPU { out in
      try await fn(out)
    }
    return result
  }

  func withBuffers(
    _ outputSize: Int, _ in1: Tensor.Data,
    _ fn: (UnsafeMutableRawPointer, UnsafeRawPointer) async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      let result = try await allocate(outputSize)
      try await result.mutateOnCPU { out in
        try await fn(out, buf1)
      }
      return result
    }
  }

  func withBuffers(
    _ outputSize: Int, _ in1: Tensor.Data, _ in2: Tensor.Data,
    _ fn: (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer) async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      try await in2.onCPU { buf2 in
        let result = try await allocate(outputSize)
        try await result.mutateOnCPU { out in
          try await fn(out, buf1, buf2)
        }
        return result
      }
    }
  }

  func withBuffers(
    _ outputSize: Int, _ in1: Tensor.Data, _ in2: Tensor.Data, _ in3: Tensor.Data,
    _ fn: (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer, UnsafeRawPointer)
      async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      try await in2.onCPU { buf2 in
        try await in3.onCPU { buf3 in
          let result = try await allocate(outputSize)
          try await result.mutateOnCPU { out in
            try await fn(out, buf1, buf2, buf3)
          }
          return result
        }
      }
    }
  }

  func withBuffers(
    _ outputSize: Int, _ in1: Tensor.Data, _ in2: Tensor.Data, _ in3: Tensor.Data,
    _ in4: Tensor.Data,
    _ fn: (
      UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeRawPointer, UnsafeRawPointer,
      UnsafeRawPointer
    ) async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      try await in2.onCPU { buf2 in
        try await in3.onCPU { buf3 in
          try await in4.onCPU { buf4 in
            let result = try await allocate(outputSize)
            try await result.mutateOnCPU { out in
              try await fn(out, buf1, buf2, buf3, buf4)
            }
            return result
          }
        }
      }
    }
  }

  func withBuffers(
    _ outputSize: Int, _ ins: [Tensor.Data],
    _ fn: (UnsafeMutableRawPointer, [UnsafeRawPointer]) async throws -> Void
  ) async throws -> Tensor.Data {
    func inner(_ ins: [Tensor.Data], _ fn: ([UnsafeRawPointer]) async throws -> Tensor.Data)
      async throws -> Tensor.Data
    {
      if ins.isEmpty {
        try await fn([])
      } else {
        try await ins[0].onCPU { ptr in
          try await inner(Array(ins[1...])) { otherPtrs in
            try await fn([ptr] + otherPtrs)
          }
        }
      }
    }
    return try await inner(ins) { allPtrs in
      let result = try await allocate(outputSize)
      try await result.mutateOnCPU { outPtr in
        try await fn(outPtr, allPtrs)
      }
      return result
    }
  }

}

func stridesForShape(_ shape: [Int]) -> [Int] {
  var strides = [Int](repeating: 0, count: shape.count)
  for i in 0..<shape.count {
    strides[i] = shape[(i + 1)...].product()
  }
  return strides
}

func readBuffer<T, T1: TensorElement>(
  _: T1.Type, _ buf: UnsafeRawPointer, count: Int, dtype: Tensor.DType,
  _ fn: (UnsafeBufferPointer<T1>) throws -> T
) throws -> T {
  if dtype == T1.dtype {
    return try fn(
      UnsafeBufferPointer(
        start: buf.bindMemory(to: T1.self, capacity: count), count: count))
  }
  var arr = [T1](repeating: T1(0.0), count: count)
  try pointerToArray(buf, output: &arr, dtype: dtype)
  return try arr.withUnsafeBufferPointer(fn)
}

func maybeReadBuffer<T, T1: TensorElement>(
  _: T1.Type, _ buf: UnsafeRawPointer?, count: Int, dtype: Tensor.DType,
  _ fn: (UnsafeBufferPointer<T1>?) throws -> T
) throws -> T {
  if let buf = buf {
    try readBuffer(T1.self, buf, count: count, dtype: dtype) {
      try fn($0)
    }
  } else {
    try fn(nil)
  }
}

func writeBuffer<T, T1: TensorElement>(
  _: T1.Type, _ buf: UnsafeMutableRawPointer, count: Int, dtype: Tensor.DType,
  _ fn: (inout UnsafeMutableBufferPointer<T1>) throws -> T
) throws -> T {
  if dtype == T1.dtype {
    var buf = UnsafeMutableBufferPointer(
      start: buf.bindMemory(to: T1.self, capacity: count), count: count)
    return try fn(&buf)
  }
  var arr = [T1](repeating: T1(0.0), count: count)
  let result = try arr.withUnsafeMutableBufferPointer(fn)
  try arrayToPointer(arr, output: buf, dtype: dtype)
  return result
}
