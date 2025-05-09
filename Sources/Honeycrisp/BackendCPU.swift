import Accelerate
import Foundation
import HCBacktrace

/// A ``Backend`` with CPU-only implementations of every operation.
///
/// This may use accelerated APIs to speed up computation, but all computations will still
/// be done on the CPU (rather than the GPU or the ANE).
open class CPUBackend: Backend, DataAllocator, @unchecked Sendable {

  public class CPUData: Tensor.Data, @unchecked Sendable {
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

    public func wait() async throws {
    }

    public func onCPU<T>(_ fn: (_: UnsafeRawPointer) async throws -> T) async throws -> T {
      try await fn(data)
    }

    public func mutateOnCPU<T>(_ fn: (_: UnsafeMutableRawPointer) async throws -> T) async throws
      -> T
    {
      try await fn(data)
    }
  }

  public class CPURandomGenerator: RandomGenerator, @unchecked Sendable {

    private let PHILOX_ROUND_A: UInt32 = 0xD251_1F53
    private let PHILOX_ROUND_B: UInt32 = 0xCD9E_8D57
    private let PHILOX_KEY_A: UInt32 = 0x9E37_79B9
    private let PHILOX_KEY_B: UInt32 = 0xBB67_AE85

    public let cpuBackend: CPUBackend

    override open var stateCount: Int {
      2
    }

    override open var stateDType: Tensor.DType {
      .int64
    }

    init(cpuBackend: CPUBackend, seed: Int) {
      self.cpuBackend = cpuBackend
      super.init(
        backend: cpuBackend,
        state: cpuBackend.use { Tensor(data: [seed, 0], dtype: .int64) })
    }

    override open func _seed(_ x: Int) async throws -> Tensor.Data {
      try await cpuBackend.collection([x, 0], reverse: false, dtype: .int64)
    }

    override open func _sample(
      state: Tensor.Data, count: Int, dist: RandomDist, dtype: Tensor.DType
    )
      async throws -> (
        sample: Tensor.Data, state: Tensor.Data
      )
    {
      let cpuBackend = cpuBackend
      return try await philox(state) { philoxFn in
        try await cpuBackend.withBuffers(count * dtype.byteSize) { outBuf in
          try await cpuBackend.serialize {
            try writeBuffer(Float.self, outBuf, count: count, dtype: dtype) { outArr in
              switch dist {
              case .uniform:
                for i in stride(from: 0, through: count, by: 4) {
                  let c = philoxFn()
                  for (j, x) in c.enumerated() {
                    if i + j < count {
                      outArr[i + j] = Float(Double(x) / Double(0xffff_ffff))
                    }
                  }
                }
              case .normal:
                for i in stride(from: 0, through: count, by: 2) {
                  let c = philoxFn()
                  for (j, (c1, c2)) in [(c[0], c[1]), (c[2], c[3])].enumerated() {
                    let u1 = max(1e-5, Float(c1) / Float(0xffff_ffff))
                    let u2 = Float(c2) / Float(0xffff_ffff)
                    let r = sqrt(-2 * log(u1))
                    let phi = 2 * Float.pi * u2
                    let z1 = r * cos(phi)
                    let z2 = r * sin(phi)
                    if i + j * 2 < count {
                      outArr[i + j * 2] = z1
                    }
                    if i + j * 2 + 1 < count {
                      outArr[i + j * 2 + 1] = z2
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    override open func _sample(state: Tensor.Data, count: Int, in range: Range<Int64>)
      async throws -> (
        sample: Tensor.Data, state: Tensor.Data
      )
    {
      func nextPowOf2(_ x: UInt64) -> UInt64 {
        var x = x

        if x == 0 {
          return 1
        }

        if x >= (UInt64.max >> 1) {
          return UInt64.max
        }

        x -= 1
        x |= x >> 1
        x |= x >> 2
        x |= x >> 4
        x |= x >> 8
        x |= x >> 16
        x |= x >> 32
        return x + 1
      }

      let bound = (UInt64(bitPattern: range.upperBound) &- UInt64(bitPattern: range.lowerBound))
      let pow2 = nextPowOf2(bound)
      let cpuBackend = cpuBackend
      return try await philox(state) { philoxFn in
        try await cpuBackend.withBuffers(count * 8) { outBuf in
          try await cpuBackend.serialize {
            try writeBuffer(Int64.self, outBuf, count: count, dtype: .int64) { outArr in
              for i in 0..<count {
                var result: UInt64 = 0
                for _ in 0..<32 {
                  let c = philoxFn()
                  for i in [0, 2] {
                    let x = (UInt64(c[i]) | ((UInt64(c[i + 1]) << 32))) % pow2
                    if x < bound {
                      result = x
                    }
                  }
                }
                outArr[i] = range.lowerBound &+ Int64(bitPattern: result)
              }
            }
          }
        }
      }
    }

    internal func philox(
      _ state: Tensor.Data,
      f: @escaping (@escaping @Sendable () -> [UInt32]) async throws -> Tensor.Data
    )
      async throws -> (sample: Tensor.Data, state: Tensor.Data)
    {
      @Sendable func philoxInner(seed: UInt64, offset: inout UInt64) -> [UInt32] {
        var c = [UInt32(offset & 0xffff_ffff), UInt32(offset >> 32), 0, 0]
        var k0 = UInt32(seed & 0xffff_ffff)
        var k1 = UInt32(seed >> 32)
        for _ in 0..<10 {
          let prevC0 = c[0]
          let prevC2 = c[2]
          c[0] = UInt32((UInt64(PHILOX_ROUND_B) * UInt64(c[2])) >> 32) ^ c[1] ^ k0
          c[2] = UInt32((UInt64(PHILOX_ROUND_A) * UInt64(prevC0)) >> 32) ^ c[3] ^ k1
          c[1] = PHILOX_ROUND_B &* prevC2
          c[3] = PHILOX_ROUND_A &* prevC0
          k0 = (k0 &+ PHILOX_KEY_A)
          k1 = (k1 &+ PHILOX_KEY_B)
        }
        offset += 1
        return c
      }
      return try await state.onCPU { stateIn in
        let stateIn = SendableRawPointer(stateIn)

        let resultState = try await cpuBackend.allocate(16)
        let sample: Tensor.Data = try await resultState.mutateOnCPU { stateOut in
          let stateOut = SendableMutableRawPointer(stateOut)

          let states = stateIn.bindMemory(to: UInt64.self, capacity: 2)
          let outArr = stateOut.bindMemory(to: UInt64.self, capacity: 2)
          outArr[0] = states[0]
          outArr[1] = states[1]

          return try await f {
            // We must re-bind due to Sendable constraints
            let outArr = stateOut.bindMemory(to: UInt64.self, capacity: 2)

            return philoxInner(seed: outArr[0], offset: &outArr[1])
          }
        }
        return (sample: sample, state: resultState)
      }
    }
  }

  public static let global = CPUBackend()

  internal var worker = Backend.WorkerThread()
  internal var _defaultRandom: CPURandomGenerator? = nil

  override public init() {
    super.init()
    self._defaultRandom = CPURandomGenerator(
      cpuBackend: self, seed: Int.random(in: 0..<1_000_000_000))
  }

  open func serialize<T>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
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

  open func allocate(_ byteCount: Int) async throws -> Tensor.Data {
    let data = try CPUData(byteCount: byteCount)
    return data
  }

  override open func broadcast(_ a: BroadcastData, dtype: Tensor.DType) async throws -> Tensor.Data
  {
    let outCount = a.strides.shape.product()
    let elSize = dtype.byteSize
    return try await withBuffers(outCount * elSize, a.data) { buffer, aBuf in
      for i in 0..<outCount {
        buffer.advanced(by: elSize * i).copyMemory(
          from: aBuf.advanced(by: elSize * a.strides(i)), byteCount: elSize)
      }
    }
  }

  override open func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    let count = a.strides.shape.product()
    return try await withBuffers(count * dtype.byteSize, a.data, b.data) { buffer, aBuf, bBuf in
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
              tracedFatalError()
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

  override open func binaryOp<T: NumericTensorElement>(
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
              case .mod: tracedFatalError()
              }
            let z = buffer.bindMemory(to: Float.self, capacity: count)
            switch op {
            case .add, .sub:
              vDSP_vsadd(x, 1, &bScalar, z, 1, vDSP_Length(count))
            case .mul, .div:
              vDSP_vsmul(x, 1, &bScalar, z, 1, vDSP_Length(count))
            case .mod:
              tracedFatalError()
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

  override open func binaryOp<T: NumericTensorElement>(
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
              tracedFatalError()
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

  override open func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    let count = a.strides.shape.product()
    return try await withBuffers(count * dtype.byteSize, a.data, b.data) { buffer, aBuf, bBuf in
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

  override open func bitwiseOp<T: TensorElementBitPattern>(
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

  override open func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let count = input.strides.shape.product()
    return try await withBuffers(count * dtype.byteSize, input.data, coeff.data, bias.data) {
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

  override open func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let count = input.strides.shape.product()
    return try await withBuffers(count * dtype.byteSize, input.data, coeff.data, bias.data) {
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

  override open func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let count = a.strides.shape.product()
    return try await withBuffers(count * Tensor.DType.bool.byteSize, a.data, b.data) {
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

  override open func compare<T: TensorElement>(
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

  override open func compare<T: TensorElement>(
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

  override open func cast(
    _ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(count * outType.byteSize, a) { buffer, aBuf in
      func apply<T: TensorElement>(_: T.Type) async throws {
        try await serialize {
          var arr = [T](repeating: T(0.0), count: count)
          try pointerToArray(aBuf.ptr, output: &arr, dtype: inType)
          try arrayToPointer(arr, output: buffer.ptr, dtype: outType)
        }
      }
      if inType == .int64 {
        try await apply(Int64.self)
      } else {
        try await apply(Float.self)
      }
    }
  }

  override open func pow<T: NumericTensorElement>(
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
                tracedFatalError()
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

  override open func clamp<T: NumericTensorElement>(
    _ a: BroadcastData, _: T.Type, min: TensorOrScalar<T>?, max: TensorOrScalar<T>?,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    #alwaysAssert(min != nil || max != nil, "cannot use clamp() without bounds")

    let count = a.strides.shape.product()

    let minData: Tensor.Data? =
      if let min = min { try await tensorOrScalarData(min, dtype) } else { nil }
    let maxData: Tensor.Data? =
      if let max = max { try await tensorOrScalarData(max, dtype) } else { nil }

    return try await withBuffers(
      outputSizes: [count * dtype.byteSize],
      inputs: Array(
        [a.data] + (minData == nil ? [] : [minData!]) + (maxData == nil ? [] : [maxData!]))
    ) { outputs, inputs in
      var inputs = inputs
      let buffer = outputs[0]
      let aBuf = inputs.remove(at: 0)
      let minBuf: SendableRawPointer? =
        if minData != nil {
          inputs.remove(at: 0)
        } else {
          nil
        }
      let maxBuf: SendableRawPointer? =
        if maxData != nil {
          inputs.remove(at: 0)
        } else {
          nil
        }
      func apply<T1: NumericTensorElement>(_: T1.Type) async throws {
        try await serialize {
          try readBuffer(T1.self, aBuf, count: a.dataCount, dtype: dtype) { arr in
            try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { out in
              try maybeReadBuffer(T1.self, minBuf, count: min?.strides.dataCount ?? 0, dtype: dtype)
              { minArr in
                try maybeReadBuffer(
                  T1.self, maxBuf, count: max?.strides.dataCount ?? 0, dtype: dtype
                ) { maxArr in
                  if let max = max, let min = min, let maxArr = maxArr, let minArr = minArr {
                    let minStrides = min.strides
                    let maxStrides = max.strides
                    let aStrides = a.strides
                    for i in 0..<count {
                      out[i] = Swift.min(
                        maxArr[maxStrides(i)], Swift.max(minArr[minStrides(i)], arr[aStrides(i)]))
                    }
                  } else if let max = max, let maxArr = maxArr {
                    let maxStrides = max.strides
                    let aStrides = a.strides
                    for i in 0..<count {
                      out[i] = Swift.min(maxArr[maxStrides(i)], arr[aStrides(i)])
                    }
                  } else if let min = min, let minArr = minArr {
                    let minStrides = min.strides
                    let aStrides = a.strides
                    for i in 0..<count {
                      out[i] = Swift.max(minArr[minStrides(i)], arr[aStrides(i)])
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
    }[0]
  }

  override open func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let outType: Tensor.DType = op.isIntOut ? .int64 : dtype
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
        return
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
        case .prod:
          try await serialize {
            try readBuffer(T.self, aBuf, count: dims.inCount, dtype: dtype) { arr in
              try writeBuffer(T.self, buffer, count: dims.outCount, dtype: dtype) { arrOut in
                var index: Int = 0
                for i in 0..<dims.outerCount {
                  for j in 0..<dims.innerCount {
                    var prod = T(1.0)
                    for k in 0..<dims.reduceCount {
                      let item = arr[j + (k + i * dims.reduceCount) * dims.innerCount]
                      prod = prod * item
                    }
                    arrOut[index] = prod
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

  override open func argsort(
    _ a: Tensor.Data, dims: ReduceDims, descending: Bool, stable: Bool, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    return try await withBuffers(dims.inCount * Tensor.DType.int64.byteSize, a) { buffer, aBuf in
      func apply<T: NumericTensorElement>(_: T.Type) async throws {
        try await serialize {
          try readBuffer(T.self, aBuf, count: dims.inCount, dtype: dtype) { arr in
            try writeBuffer(Int64.self, buffer, count: dims.inCount, dtype: .int64) { arrOut in
              for i in 0..<dims.outerCount {
                for j in 0..<dims.innerCount {
                  var values = (0..<dims.reduceCount).map { k in
                    (arr[j + (k + i * dims.reduceCount) * dims.innerCount], Int64(k))
                  }
                  values.sort { x, y in descending ? x.0 > y.0 : x.0 < y.0 }
                  for (k, (_, sourceIdx)) in values.enumerated() {
                    arrOut[j + (k + i * dims.reduceCount) * dims.innerCount] = sourceIdx
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

  override open func cumulativeSum(
    _ a: Tensor.Data, dims: ReduceDims, exclusive: Bool, reverse: Bool, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    return try await withBuffers(dims.inCount * dtype.byteSize, a) { buffer, aBuf in
      func apply<T: NumericTensorElement>(_: T.Type) async throws {
        try readBuffer(T.self, aBuf, count: dims.inCount, dtype: dtype) { arr in
          try writeBuffer(T.self, buffer, count: dims.inCount, dtype: dtype) { arrOut in
            let innerIndices =
              if reverse {
                stride(from: dims.reduceCount - 1, through: 0, by: -1)
              } else {
                stride(from: 0, through: dims.reduceCount - 1, by: 1)
              }
            for i in 0..<dims.outerCount {
              for j in 0..<dims.innerCount {
                var sum = T(0.0)
                for k in innerIndices {
                  let idx = j + (k + i * dims.reduceCount) * dims.innerCount
                  let item = arr[idx]
                  let next = sum + item
                  arrOut[idx] = (exclusive ? sum : next)
                  sum = next
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

  override open func cumulativeProd(
    _ a: Tensor.Data, dims: ReduceDims, exclusive: Bool, reverse: Bool, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    return try await withBuffers(dims.inCount * dtype.byteSize, a) { buffer, aBuf in
      func apply<T: NumericTensorElement>(_: T.Type) async throws {
        try readBuffer(T.self, aBuf, count: dims.inCount, dtype: dtype) { arr in
          try writeBuffer(T.self, buffer, count: dims.inCount, dtype: dtype) { arrOut in
            let innerIndices =
              if reverse {
                stride(from: dims.reduceCount - 1, through: 0, by: -1)
              } else {
                stride(from: 0, through: dims.reduceCount - 1, by: 1)
              }
            for i in 0..<dims.outerCount {
              for j in 0..<dims.innerCount {
                var prod = T(1.0)
                for k in innerIndices {
                  let idx = j + (k + i * dims.reduceCount) * dims.innerCount
                  let item = arr[idx]
                  let next = prod * item
                  arrOut[idx] = (exclusive ? prod : next)
                  prod = next
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

  override open func logSoftmax(
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

  override open func logSoftmaxGrad(
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

  override open func repeated(
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

  override open func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(s.gatherOutCount * dtype.byteSize, a, s.indices.data) {
      outData, inData, idxBuf in
      func apply<T: TensorElement>(_ zero: T) async throws {
        try await serialize {
          try readBuffer(Int64.self, idxBuf, count: s.indices.dataCount, dtype: .int64) {
            flatIndices in
            try readBuffer(T.self, inData, count: s.gatherInCount, dtype: dtype) { inArr in
              try writeBuffer(T.self, outData, count: s.gatherOutCount, dtype: dtype) { outArr in
                let outerCount = s.outerCount
                let oldMidCount = s.middleCount
                let newMidCount = s.outCount
                let innerCount = s.innerCount
                for i in 0..<outerCount {
                  for j in 0..<newMidCount {
                    for k in 0..<innerCount {
                      let outIdx = (i * newMidCount + j) * innerCount + k
                      let inIdx = Int(flatIndices[s.indices.strides(outIdx)])
                      let source = inArr[(i * oldMidCount + inIdx) * innerCount + k]
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

  override open func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await withBuffers(s.gatherInCount * dtype.byteSize, a, s.indices.data) {
      buffer, aBuf, idxBuf in
      func apply<T: NumericTensorElement>(_ zero: T) async throws {
        try await serialize {
          try readBuffer(Int64.self, idxBuf, count: s.indices.dataCount, dtype: .int64) {
            flatIndices in
            try readBuffer(T.self, aBuf, count: s.gatherOutCount, dtype: dtype) { inArr in
              let outerCount = s.outerCount
              let oldMidCount = s.middleCount
              let newMidCount = s.outCount
              let innerCount = s.innerCount
              var outArr = [T](repeating: zero, count: s.gatherInCount)
              for i in 0..<outerCount {
                for j in 0..<newMidCount {
                  for k in 0..<innerCount {
                    let inIdx = (i * newMidCount + j) * innerCount + k
                    let jOut = Int(flatIndices[s.indices.strides(inIdx)])
                    let outIdx = (i * oldMidCount + jOut) * innerCount + k
                    outArr[outIdx] = outArr[outIdx] + inArr[inIdx]
                  }
                }
              }
              try arrayToPointer(outArr, output: buffer.ptr, dtype: dtype)
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

  override open func when<T>(
    _ mask: BroadcastData, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let count = mask.strides.shape.product()
    let aData = try await tensorOrScalarData(a, dtype)
    let bData = try await tensorOrScalarData(b, dtype)
    return try await withBuffers(count * dtype.byteSize, mask.data, aData, bData) {
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

  override open func matmul(
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

  override open func batchedMatmul(
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

  override open func triangular(
    _ a: Tensor.Data, batch: Int, rows: Int, cols: Int, upper: Bool, offset: Int,
    dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    let elSize = dtype.byteSize
    let rowSize = elSize * cols
    return try await withBuffers(batch * rows * rowSize, a) { outPtr, inPtr in
      try await serialize {
        outPtr.ptr.initializeMemory(
          as: UInt8.self, repeating: 0, count: elSize * rows * cols * batch)
        for i in 0..<batch {
          for j in 0..<rows {
            let rowIn = inPtr.advanced(by: (i * rows + j) * cols * elSize)
            let rowOut = outPtr.advanced(by: (i * rows + j) * cols * elSize)
            if upper {
              let elCount = min(cols, cols - j - offset)
              if elCount > 0 {
                let skip = (cols - elCount)
                rowOut.advanced(by: skip * elSize).copyMemory(
                  from: rowIn.advanced(by: skip * elSize), byteCount: elCount * elSize)
              }
            } else {
              let elCount = min(cols, (j + 1) + offset)
              if elCount > 0 {
                rowOut.copyMemory(from: rowIn, byteCount: elCount * elSize)
              }
            }
          }
        }
      }
    }
  }

  override open func qrDecomposition(
    _ a: Tensor.Data,
    batch: Int,
    rows m: Int,
    cols n: Int,
    full: Bool,
    dtype: Tensor.DType
  ) async throws -> (q: Tensor.Data, r: Tensor.Data) {
    if #available(macOS 13.3, *) {
      let full = (m < n) || full

      // Determine output shapes.
      // For tall (or square) matrices (m >= n):
      //   Reduced: Q is m×n, R is n×n.
      //   Full:    Q is m×m, R is m×n.
      // For wide matrices (m < n): Q is m×m, R is m×n.
      let qCols = (m >= n) ? (full ? m : n) : m
      let qSize = batch * m * qCols
      let qBytes = qSize * dtype.byteSize

      let rRows = (m >= n) ? (full ? m : n) : m
      let rSize = batch * rRows * n
      let rBytes = rSize * dtype.byteSize

      let qAndR = try await withBuffers(outputSizes: [qBytes, rBytes], inputs: [a]) {
        outBufs, inBufs in
        let qBuf = outBufs[0]
        let rBuf = outBufs[1]
        let inBuf = inBufs[0]
        return try readBuffer(Float.self, inBuf, count: batch * m * n, dtype: dtype) { arrIn in
          try writeBuffer(Float.self, qBuf, count: qSize, dtype: dtype) { arrQ in
            try writeBuffer(Float.self, rBuf, count: rSize, dtype: dtype) { arrR in
              typealias IntType = __LAPACK_int

              let k = min(m, n)
              var localA = Array(repeating: Float(0.0), count: m * n)
              var reflectScalars = [Float](repeating: 0.0, count: k)

              for b in 0..<batch {
                transposeMatrix(
                  input: arrIn.baseAddress!, output: &localA, outRows: n, outCols: m,
                  inOffset: b * m * n)

                var illegalArgIdx: IntType = 0
                var mm = IntType(m)
                var nn = IntType(n)
                var leadingDim = mm

                // Query workspace size for SGEQRF.
                var lwork: IntType = -1
                var workQuery: Float = 0.0
                sgeqrf_(
                  &mm, &nn, &localA, &leadingDim, &reflectScalars, &workQuery, &lwork,
                  &illegalArgIdx)
                lwork = IntType(workQuery)
                var work = [Float](repeating: 0.0, count: Int(lwork))

                // Compute the QR factorization.
                sgeqrf_(
                  &mm, &nn, &localA, &leadingDim, &reflectScalars, &work, &lwork, &illegalArgIdx)
                if illegalArgIdx != 0 {
                  throw BackendError.lapackError(
                    "failed in SGEQRF with illegalArgIdx=\(illegalArgIdx)")
                }

                // Extract R.
                // SGEQRF overwrites localA with the R factor in its upper-triangular part.
                let rBase = b * rRows * n
                for j in 0..<n {
                  for i in 0..<rRows {
                    // For valid entries, only rows i <= j (and i < k) are defined.
                    if i <= j && i < k {
                      arrR[rBase + i * n + j] = localA[i + j * m]
                    } else {
                      arrR[rBase + i * n + j] = 0.0
                    }
                  }
                }

                // Generate Q.
                var transQ: [Float]
                if m > n && full {
                  transQ = Array(repeating: 0.0, count: m * qCols)
                  // Copy the first n columns from localA.
                  for j in 0..<n {
                    for i in 0..<m {
                      transQ[i + j * m] = localA[i + j * m]
                    }
                  }
                } else if m > n {
                  transQ = localA
                } else {
                  transQ = Array(repeating: 0.0, count: m * qCols)
                  for j in 0..<m {
                    for i in 0..<m {
                      transQ[i + j * m] = localA[i + j * m]
                    }
                  }
                }
                var qm = IntType(m)
                var qn = IntType(qCols)
                var qk = IntType(k)
                lwork = -1
                sorgqr_(
                  &qm, &qn, &qk, &transQ, &leadingDim, &reflectScalars, &workQuery, &lwork,
                  &illegalArgIdx)
                lwork = IntType(workQuery)
                work = [Float](repeating: 0.0, count: Int(lwork))
                sorgqr_(
                  &qm, &qn, &qk, &transQ, &leadingDim, &reflectScalars, &work, &lwork,
                  &illegalArgIdx)
                if illegalArgIdx != 0 {
                  throw BackendError.lapackError(
                    "failed in SORGQR (full) with illegalArgIdx=\(illegalArgIdx)")
                }

                transposeMatrix(
                  input: &transQ, output: arrQ.baseAddress!, outRows: m, outCols: qCols,
                  outOffset: b * m * qCols)
              }
            }
          }
        }
      }
      return (q: qAndR[0], r: qAndR[1])
    } else {
      throw BackendError.notImplemented("qrDecomposition (need macOS 13.3 or newer)")
    }
  }

  override open func svd(
    _ a: Tensor.Data,
    batch: Int,
    rows m: Int,
    cols n: Int,
    full: Bool,
    dtype: Tensor.DType
  ) async throws -> (u: Tensor.Data, s: Tensor.Data, vt: Tensor.Data) {
    if #available(macOS 13.3, *) {
      let k = min(m, n)

      let uRows = m
      let uCols = full ? m : k
      let uSize = batch * uRows * uCols
      let uBytes = uSize * dtype.byteSize

      let vtRows = full ? n : k
      let vtCols = n
      let vtSize = batch * vtRows * vtCols
      let vtBytes = vtSize * dtype.byteSize

      let sBytes = batch * k * dtype.byteSize

      let results = try await withBuffers(outputSizes: [uBytes, sBytes, vtBytes], inputs: [a]) {
        outBufs, inBufs in
        let uBuf = outBufs[0]
        let sBuf = outBufs[1]
        let vtBuf = outBufs[2]
        let inBuf = inBufs[0]
        return try readBuffer(Float.self, inBuf, count: batch * m * n, dtype: dtype) { arrIn in
          try writeBuffer(Float.self, uBuf, count: uSize, dtype: dtype) { arrU in
            try writeBuffer(Float.self, sBuf, count: k * batch, dtype: dtype) { arrS in
              try writeBuffer(Float.self, vtBuf, count: vtSize, dtype: dtype) { arrVt in
                typealias IntType = __LAPACK_int

                var localA = Array(repeating: Float(0.0), count: m * n)
                var localU = [Float](repeating: 0, count: uSize)
                var localVt = [Float](repeating: 0, count: vtSize)

                for b in 0..<batch {
                  transposeMatrix(
                    input: arrIn.baseAddress!, output: &localA, outRows: n, outCols: m,
                    inOffset: b * m * n)

                  let job = (full ? "A" : "S").utf8CString[0]
                  var jobU = job
                  var jobVt = job
                  var mm = IntType(m)
                  var mn = IntType(n)
                  var leadingDim = mm

                  var leadingDimU = IntType(uRows)
                  var leadingDimVt = IntType(vtRows)

                  var lwork: IntType = -1
                  var workQuery: Float = 0.0
                  var info: IntType = 0
                  sgesvd_(
                    &jobU, &jobVt, &mm, &mn, &localA, &leadingDim, arrS.baseAddress! + b * k,
                    &localU, &leadingDimU,
                    &localVt, &leadingDimVt, &workQuery, &lwork, &info)
                  lwork = IntType(workQuery)
                  var work = [Float](repeating: 0.0, count: Int(lwork))
                  sgesvd_(
                    &jobU, &jobVt, &mm, &mn, &localA, &leadingDim, arrS.baseAddress! + b * k,
                    &localU, &leadingDimU,
                    &localVt, &leadingDimVt, &work, &lwork, &info)
                  if info < 0 {
                    throw BackendError.lapackError("argument \(-info) is invalid")
                  } else if info > 0 {
                    throw BackendError.lapackError("\(info) superdiagonals did not converge")
                  }

                  transposeMatrix(
                    input: &localU, output: arrU.baseAddress!, outRows: uRows, outCols: uCols,
                    outOffset: b * uRows * uCols)
                  transposeMatrix(
                    input: &localVt, output: arrVt.baseAddress!, outRows: vtRows, outCols: vtCols,
                    outOffset: b * vtRows * vtCols)
                }
              }
            }
          }
        }
      }
      return (u: results[0], s: results[1], vt: results[2])
    } else {
      throw BackendError.notImplemented("svd (need macOS 13.3 or newer)")
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

  override open func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNd(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override open func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdTranspose(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override open func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdKernelGrad(config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override open func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNd(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override open func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdTranspose(config, batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override open func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await convNdKernelGrad(config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override open func elemwise(
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

  override open func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    #alwaysAssert(inputs.count == innerCounts.count)
    let totalInner = innerCounts.sum()
    return try await withBuffers(
      outputSizes: [outerCount * totalInner * dtype.byteSize], inputs: inputs
    ) {
      outBufs, inBufs in
      let buffer = outBufs[0]
      return try await serialize {
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
    }[0]
  }

  override open func constant<T: TensorElement>(_ value: T, count: Int, dtype: Tensor.DType)
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

  override open func collection<T: TensorElement>(
    _ collection: some Collection<T>, reverse: Bool, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    let arr = (reverse ? collection.reversed() : Array(collection))
    let count = arr.count
    return try await withBuffers(count * dtype.byteSize) { buffer in
      try await serialize {
        try arrayToPointer(arr, output: buffer.ptr, dtype: dtype)
      }
    }
  }

  override open func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data {
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
        try arrayToPointer(newIndices, output: buffer.ptr, dtype: .int64)
      }
    }
  }

  override open func defaultRandom() -> RandomGenerator {
    _defaultRandom!
  }

  override open func createRandom() -> RandomGenerator {
    CPURandomGenerator(cpuBackend: self, seed: Int.random(in: 0..<1_000_000_000))
  }

  func tensorOrScalarData<T>(_ x: TensorOrScalar<T>, _ dtype: Tensor.DType) async throws
    -> Tensor.Data
  {
    switch x {
    case .tensor(let t):
      t.data
    case .scalar(let s, _):
      try await constant(s, count: 1, dtype: dtype)
    }
  }

  func withMaybeBuffer<T>(
    _ buf: Tensor.Data?, _ fn: (SendableRawPointer?) async throws -> T
  ) async throws -> T {
    if let buf = buf {
      try await buf.onCPU { ptr in
        try await fn(SendableRawPointer(ptr))
      }
    } else {
      try await fn(nil)
    }
  }

  func withBuffers(
    _ outputSize: Int, _ fn: (SendableMutableRawPointer) async throws -> Void
  ) async throws -> Tensor.Data {
    let result = try await allocate(outputSize)
    try await result.mutateOnCPU { out in
      try await fn(SendableMutableRawPointer(out))
    }
    return result
  }

  func withBuffers(
    _ outputSize: Int, _ in1: Tensor.Data,
    _ fn: (SendableMutableRawPointer, SendableRawPointer) async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      let buf1 = SendableRawPointer(buf1)
      let result = try await allocate(outputSize)
      try await result.mutateOnCPU { out in
        try await fn(SendableMutableRawPointer(out), buf1)
      }
      return result
    }
  }

  func withBuffers(
    _ outputSize: Int, _ in1: Tensor.Data, _ in2: Tensor.Data,
    _ fn: (SendableMutableRawPointer, SendableRawPointer, SendableRawPointer) async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      let buf1 = SendableRawPointer(buf1)
      return try await in2.onCPU { buf2 in
        let buf2 = SendableRawPointer(buf2)
        let result = try await allocate(outputSize)
        try await result.mutateOnCPU { out in
          try await fn(SendableMutableRawPointer(out), buf1, buf2)
        }
        return result
      }
    }
  }

  func withBuffers(
    _ outputSize: Int, _ in1: Tensor.Data, _ in2: Tensor.Data, _ in3: Tensor.Data,
    _ fn: (SendableMutableRawPointer, SendableRawPointer, SendableRawPointer, SendableRawPointer)
      async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      let buf1 = SendableRawPointer(buf1)
      return try await in2.onCPU { buf2 in
        let buf2 = SendableRawPointer(buf2)
        return try await in3.onCPU { buf3 in
          let buf3 = SendableRawPointer(buf3)
          let result = try await allocate(outputSize)
          try await result.mutateOnCPU { out in
            try await fn(SendableMutableRawPointer(out), buf1, buf2, buf3)
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
      SendableMutableRawPointer, SendableRawPointer, SendableRawPointer, SendableRawPointer,
      SendableRawPointer
    ) async throws -> Void
  ) async throws -> Tensor.Data {
    try await in1.onCPU { buf1 in
      let buf1 = SendableRawPointer(buf1)
      return try await in2.onCPU { buf2 in
        let buf2 = SendableRawPointer(buf2)
        return try await in3.onCPU { buf3 in
          let buf3 = SendableRawPointer(buf3)
          return try await in4.onCPU { buf4 in
            let buf4 = SendableRawPointer(buf4)
            let result = try await allocate(outputSize)
            try await result.mutateOnCPU { out in
              try await fn(SendableMutableRawPointer(out), buf1, buf2, buf3, buf4)
            }
            return result
          }
        }
      }
    }
  }

  func withBuffers(
    outputSizes: [Int], inputs ins: [Tensor.Data],
    _ fn: ([SendableMutableRawPointer], [SendableRawPointer]) async throws -> Void
  ) async throws -> [Tensor.Data] {
    func readFunc(_ ins: [Tensor.Data], _ fn: ([SendableRawPointer]) async throws -> [Tensor.Data])
      async throws -> [Tensor.Data]
    {
      if ins.isEmpty {
        try await fn([])
      } else {
        try await ins[0].onCPU { ptr in
          let ptr = SendableRawPointer(ptr)
          return try await readFunc(Array(ins[1...])) { otherPtrs in
            try await fn([ptr] + otherPtrs)
          }
        }
      }
    }

    func writeFunc(
      _ sizes: [Int], _ fn: ([SendableMutableRawPointer]) async throws -> Void
    )
      async throws -> [Tensor.Data]
    {
      if sizes.isEmpty {
        try await fn([])
        return []
      } else {
        let result = try await allocate(sizes[0])
        return [result]
          + (try await result.mutateOnCPU { ptr in
            let ptr = SendableMutableRawPointer(ptr)
            return try await writeFunc(Array(sizes[1...])) { otherPtrs in
              try await fn([ptr] + otherPtrs)
            }
          })
      }
    }

    return try await readFunc(ins) { readPtrs in
      try await writeFunc(outputSizes) { writePtrs in
        try await fn(writePtrs, readPtrs)
      }
    }
  }
}

func readBuffer<T, T1: TensorElement>(
  _: T1.Type, _ buf: SendableRawPointer, count: Int, dtype: Tensor.DType,
  _ fn: (UnsafeBufferPointer<T1>) throws -> T
) throws -> T {
  if dtype == T1.dtype {
    return try fn(
      UnsafeBufferPointer(
        start: buf.bindMemory(to: T1.self, capacity: count), count: count))
  }
  var arr = [T1](repeating: T1(0.0), count: count)
  try pointerToArray(buf.ptr, output: &arr, dtype: dtype)
  return try arr.withUnsafeBufferPointer(fn)
}

func maybeReadBuffer<T, T1: TensorElement>(
  _: T1.Type, _ buf: SendableRawPointer?, count: Int, dtype: Tensor.DType,
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
  _: T1.Type, _ buf: SendableMutableRawPointer, count: Int, dtype: Tensor.DType,
  _ fn: (inout UnsafeMutableBufferPointer<T1>) throws -> T
) throws -> T {
  if dtype == T1.dtype {
    var buf = UnsafeMutableBufferPointer(
      start: buf.bindMemory(to: T1.self, capacity: count), count: count)
    return try fn(&buf)
  }
  var arr = [T1](repeating: T1(0.0), count: count)
  let result = try arr.withUnsafeMutableBufferPointer(fn)
  try arrayToPointer(arr, output: buf.ptr, dtype: dtype)
  return result
}

func transposeMatrix(
  input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, outRows: Int, outCols: Int,
  inOffset: Int = 0, outOffset: Int = 0
) {
  let a = input + inOffset
  let b = output + outOffset
  vDSP_mtrans(a, 1, b, 1, vDSP_Length(outRows), vDSP_Length(outCols))
}

struct SendableRawPointer: @unchecked Sendable {
  let ptr: UnsafeRawPointer

  init(_ ptr: UnsafeRawPointer) {
    self.ptr = ptr
  }

  #if compiler(>=6)
    func bindMemory<T>(
      to type: T.Type,
      capacity count: Int
    ) -> UnsafePointer<T> where T: ~Copyable {
      ptr.bindMemory(to: type, capacity: count)
    }
  #else
    func bindMemory<T>(
      to type: T.Type,
      capacity count: Int
    ) -> UnsafePointer<T> {
      ptr.bindMemory(to: type, capacity: count)
    }
  #endif

  func advanced(by n: Int) -> UnsafeRawPointer {
    ptr.advanced(by: n)
  }
}

struct SendableMutableRawPointer: @unchecked Sendable {
  public let ptr: UnsafeMutableRawPointer

  init(_ ptr: UnsafeMutableRawPointer) {
    self.ptr = ptr
  }

  #if compiler(>=6)
    func bindMemory<T>(
      to type: T.Type,
      capacity count: Int
    ) -> UnsafeMutablePointer<T> where T: ~Copyable {
      ptr.bindMemory(to: type, capacity: count)
    }
  #else
    func bindMemory<T>(
      to type: T.Type,
      capacity count: Int
    ) -> UnsafeMutablePointer<T> {
      ptr.bindMemory(to: type, capacity: count)
    }
  #endif

  func advanced(by n: Int) -> UnsafeMutableRawPointer {
    ptr.advanced(by: n)
  }
}
