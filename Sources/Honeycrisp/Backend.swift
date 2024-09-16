import Accelerate
import Foundation
import Metal

public enum BackendError: Error {
  case notImplemented(String)
  case failedToCreateMTLDevice
  case failedToCreateCommandQueue
  case allocationFailed(Int)
}

open class Backend {

  private static let ThreadKey = "HONEYCRISP_CURRENT_BACKEND"

  public static var defaultBackend: Backend = CPUBackend()

  public static var current: Backend {
    if let backend = Thread.current.threadDictionary[ThreadKey] {
      (backend as? Backend)!
    } else {
      defaultBackend
    }
  }

  public func use<T>(_ fn: () throws -> T) rethrows -> T {
    if let backend = Thread.current.threadDictionary[Backend.ThreadKey] {
      let old = (backend as? Backend)!
      defer {
        Thread.current.threadDictionary[Backend.ThreadKey] = old
      }
      Thread.current.threadDictionary[Backend.ThreadKey] = self
      return try fn()
    } else {
      defer {
        Thread.current.threadDictionary.removeObject(forKey: Backend.ThreadKey)
      }
      Thread.current.threadDictionary[Backend.ThreadKey] = self
      return try fn()
    }
  }

  private var queue = DispatchQueue(label: "backend-worker")

  internal func serialize<T>(_ work: () throws -> T) async throws -> T {
    return try await withCheckedThrowingContinuation { continuation in
      queue.sync {
        do {
          continuation.resume(returning: try work())
        } catch {
          continuation.resume(throwing: error)
        }
      }
    }
  }

  public func allocate(length: Int) async throws -> MTLBuffer {
    throw BackendError.notImplemented("allocate")
  }

  public func binaryOp(
    _ a: Tensor.Data, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("binaryOp")
  }

  public func compare(
    _ a: Tensor.Data, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  public func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  public func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("compare")
  }

  public func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("cast")
  }

  public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("pow")
  }

  public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("reduce")
  }

  public func repeated(
    _ a: Tensor.Data, outerCount: Int, innerCount: Int, repeats: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("repeated")
  }

  public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("gather")
  }

  public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("scatter")
  }

  public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("matmul")
  }

  public func elemwise(_ a: Tensor.Data, op: ElemwiseOp, count: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("elemwise")
  }

  public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("concat")
  }

  public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data {
    throw BackendError.notImplemented("axisPermutation")
  }

  public func defaultRandom() async throws -> RandomGenerator {
    throw BackendError.notImplemented("defaultRandom")
  }

  public func createRandom() async throws -> RandomGenerator {
    throw BackendError.notImplemented("createRandom")
  }

}

open class CPUBackend: Backend {

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
      -> Tensor.Data
    {
      let buffer = try await backend.allocate(length: count * dtype.byteSize)
      try await backend.serialize {
        switch dist {
        case .uniform:
          let arr = (0..<count).map { _ in Float.random(in: 0..<1.0) }
          try arrayToPointer(arr, output: buffer.contents(), dtype: dtype)
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
          try arrayToPointer(results, output: buffer.contents(), dtype: dtype)
        }
      }
      return Tensor.Data(backend: backend, buffer: buffer)
    }

    public func sample(count: Int, in range: Range<Int64>) async throws -> Tensor.Data {
      let buffer = try await backend.allocate(length: count * Tensor.DType.int64.byteSize)
      try await backend.serialize {
        let ints = (0..<count).map { _ in Int64.random(in: range) }
        try arrayToPointer(ints, output: buffer.contents(), dtype: .int64)
      }
      return Tensor.Data(backend: backend, buffer: buffer)
    }
  }

  private var queue = DispatchQueue(label: "cpu-backend-worker")
  private static var _global = CPUBackend()
  public static var global: CPUBackend { CPUBackend._global }

  internal var _device: MTLDevice?
  internal var device: MTLDevice {
    get throws {
      if let d = _device {
        return d
      }
      if let d = MTLCreateSystemDefaultDevice() {
        _device = d
        return d
      }
      throw BackendError.failedToCreateMTLDevice
    }
  }

  internal func waitForData(_ xs: Tensor.Data...) async throws {
    for x in xs {
      if let waiter = x.completeOnAllDevices {
        try await waiter.value
      }
    }
  }

  override public func allocate(length: Int) async throws -> MTLBuffer {
    return try await serialize {
      guard
        let result = (try device).makeBuffer(length: max(1, length), options: [.storageModeShared])
      else {
        throw BackendError.allocationFailed(length)
      }
      return result
    }
  }

  override public func binaryOp(
    _ a: Tensor.Data, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    try await waitForData(a, b)

    func apply<T: NumericTensorElement>(_: T.Type) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        if dtype == .float32 && (op == .add || op == .mul) {
          let x = UnsafePointer<Float>(
            a.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let y = UnsafePointer<Float>(
            b.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let z = buffer.contents().bindMemory(to: Float.self, capacity: count)
          switch op {
          case .add:
            vDSP_vadd(x, 1, y, 1, z, 1, vDSP_Length(count))
          case .mul:
            vDSP_vmul(x, 1, y, 1, z, 1, vDSP_Length(count))
          }
        } else {
          var aData = [T](repeating: T(0.0), count: count)
          var bData = [T](repeating: T(0.0), count: count)
          try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
          try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
          let cData = zip(aData, bData).map { op.apply($0, $1) }
          try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        }
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64.self)
    } else {
      return try await apply(Float.self)
    }
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)

    func apply<T1: NumericTensorElement>(_ b: T1) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        if dtype == .float32 && (op == .add || op == .mul) {
          let x = UnsafePointer<Float>(
            a.buffer.contents().bindMemory(to: Float.self, capacity: count))
          [b.toFloat()].withUnsafeBufferPointer({ y in
            let z = buffer.contents().bindMemory(to: Float.self, capacity: count)
            switch op {
            case .add:
              vDSP_vsadd(x, 1, y.baseAddress!, z, 1, vDSP_Length(count))
            case .mul:
              vDSP_vsmul(x, 1, y.baseAddress!, z, 1, vDSP_Length(count))
            }
          })
        } else {
          var aData = [T1](repeating: T1(0.0), count: count)
          try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
          let cData = aData.map { op.apply($0, b) }
          try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        }
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(b.toInt64())
    } else {
      return try await apply(b.toFloat())
    }
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(b)
    func apply<T1: NumericTensorElement>(_ a: T1) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        if dtype == .float32 && (op == .add || op == .mul) {
          let x = UnsafePointer<Float>(
            b.buffer.contents().bindMemory(to: Float.self, capacity: count))
          [a.toFloat()].withUnsafeBufferPointer({ y in
            let z = buffer.contents().bindMemory(to: Float.self, capacity: count)
            switch op {
            case .add:
              vDSP_vsadd(x, 1, y.baseAddress!, z, 1, vDSP_Length(count))
            case .mul:
              vDSP_vsmul(x, 1, y.baseAddress!, z, 1, vDSP_Length(count))
            }
          })
        } else {
          var bData = [T1](repeating: T1(0.0), count: count)
          try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
          let cData = bData.map { op.apply(a, $0) }
          try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        }
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(a.toInt64())
    } else {
      return try await apply(a.toFloat())
    }
  }

  override public func compare(
    _ a: Tensor.Data, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a, b)

    func apply<T1: NumericTensorElement>(_: T1.Type) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * Tensor.DType.bool.byteSize)
      try await serialize {
        var aData = [T1](repeating: T1(0.0), count: count)
        var bData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
        let cData = zip(aData, bData).map { op.apply($0, $1) }
        try arrayToPointer(cData, output: buffer.contents(), dtype: .bool)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64.self)
    } else {
      return try await apply(Float.self)
    }
  }

  override public func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)
    func apply<T1: NumericTensorElement>(_ b: T1) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * Tensor.DType.bool.byteSize)
      try await serialize {
        var aData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        let cData = aData.map { op.apply($0, b) }
        try arrayToPointer(cData, output: buffer.contents(), dtype: .bool)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(b.toInt64())
    } else {
      return try await apply(b.toFloat())
    }
  }

  override public func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(b)
    func apply<T1: NumericTensorElement>(_ a: T1) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * Tensor.DType.bool.byteSize)
      try await serialize {
        var bData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
        let cData = bData.map { op.apply(a, $0) }
        try arrayToPointer(cData, output: buffer.contents(), dtype: .bool)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(a.toInt64())
    } else {
      return try await apply(a.toFloat())
    }
  }

  override public func cast(
    _ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)

    func apply<T: TensorElement>(_: T.Type) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * outType.byteSize)
      try await serialize {
        var arr = [T](repeating: T(0.0), count: count)
        try pointerToArray(a.buffer.contents(), output: &arr, dtype: inType)
        try arrayToPointer(arr, output: buffer.contents(), dtype: outType)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if inType == .int64 {
      return try await apply(Int64.self)
    } else {
      return try await apply(Float.self)
    }
  }

  override public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)

    func apply<T1: NumericTensorElement>(_ b: T1) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        if dtype == .float32 && (b == T1(2.0) || b == T1(-1.0)) {
          let x = UnsafePointer<Float>(
            a.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let z = UnsafeMutablePointer<Float>(
            buffer.contents().bindMemory(to: Float.self, capacity: count))
          if b == T1(2.0) {
            vDSP_vmul(x, 1, x, 1, z, 1, vDSP_Length(count))
          } else {
            [Float(1)].withUnsafeBufferPointer { one in
              vDSP_svdiv(one.baseAddress!, x, 1, z, 1, vDSP_Length(count))
            }
          }
        } else {
          var arr = [T1](repeating: T1(0.0), count: count)
          try pointerToArray(a.buffer.contents(), output: &arr, dtype: dtype)
          let cData = arr.map { $0.pow(b) }
          try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        }
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(b.toInt64())
    } else {
      return try await apply(b.toFloat())
    }
  }

  override public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)

    func apply<T: NumericTensorElement>(_ x: T) async throws -> Tensor.Data {
      let arr = try await serialize {
        var arr = [T](repeating: x, count: dims.inCount)
        try pointerToArray(a.buffer.contents(), output: &arr, dtype: dtype)
        return arr
      }

      switch op {
      case .sum:
        var arrOut = [T]()
        let buffer = try await allocate(length: dims.outCount * dtype.byteSize)
        try await serialize {
          for i in 0..<dims.outerCount {
            for j in 0..<dims.innerCount {
              var sum = T(0.0)
              for k in 0..<dims.reduceCount {
                let item = arr[j + (k + i * dims.reduceCount) * dims.innerCount]
                sum = sum + item
              }
              arrOut.append(sum)
            }
          }
          assert(arrOut.count == dims.outCount)
          try arrayToPointer(arrOut, output: buffer.contents(), dtype: dtype)
        }
        return Tensor.Data(backend: self, buffer: buffer)
      case .argmin, .argmax:
        assert(dims.outCount > 0, "cannot apply op \(self) to empty dimension")
        var arrOut = [Int64]()
        let buffer = try await allocate(length: dims.outCount * Tensor.DType.int64.byteSize)
        try await serialize {
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
              arrOut.append(index)
            }
          }
          assert(arrOut.count == dims.outCount)
          try arrayToPointer(arrOut, output: buffer.contents(), dtype: .int64)
        }
        return Tensor.Data(backend: self, buffer: buffer)
      }
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
    }
  }

  override public func repeated(
    _ a: Tensor.Data, outerCount: Int, innerCount: Int, repeats: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)
    let outData = try await allocate(length: outerCount * innerCount * repeats * dtype.byteSize)
    try await serialize {
      let inData = a.buffer.contents()
      let innerBytes = dtype.byteSize * innerCount
      for i in 0..<outerCount {
        for j in 0..<repeats {
          let outBytes = outData.contents().advanced(
            by: (i * repeats + j) * innerBytes)
          let inBytes = inData.advanced(by: i * innerBytes)
          outBytes.copyMemory(from: inBytes, byteCount: innerBytes)
        }
      }
    }
    return Tensor.Data(backend: self, buffer: outData)
  }

  override public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a, s.indices)

    let flatIndices = try await serialize {
      var flatIndices = [Int64](repeating: 0, count: s.indicesCount)
      try pointerToArray(s.indices.buffer.contents(), output: &flatIndices, dtype: .int64)
      return flatIndices
    }

    if s.broadcasted {
      let innerSize = s.innerCount * dtype.byteSize
      let outBuffer = try await allocate(
        length: s.innerCount * flatIndices.count * innerSize)
      try await serialize {
        let inData = a.buffer.contents()
        let outData = outBuffer.contents()
        for i in 0..<s.outerCount {
          for (j, idx) in flatIndices.enumerated() {
            let source = inData.advanced(by: i * s.middleCount * innerSize + Int(idx) * innerSize)
            let dst = outData.advanced(
              by: i * flatIndices.count * innerSize + j * innerSize)
            dst.copyMemory(from: source, byteCount: innerSize)
          }
        }
      }
      return Tensor.Data(backend: self, buffer: outBuffer)
    }

    func apply<T: TensorElement>(_ zero: T) async throws -> Tensor.Data {
      let buffer = try await allocate(length: s.scatterOutCount * dtype.byteSize)
      try await serialize {
        var inArr = [T](repeating: zero, count: s.scatterInCount)
        try pointerToArray(a.buffer.contents(), output: &inArr, dtype: dtype)
        var outArr = [T](repeating: zero, count: s.scatterOutCount)
        for i in 0..<s.outerCount {
          for j in 0..<s.outCount {
            for k in 0..<s.innerCount {
              let outIdx = i * s.outCount * s.innerCount + j * s.innerCount + k
              let inIdx = Int(flatIndices[outIdx])
              let source = inArr[i * s.middleCount * s.innerCount + inIdx * s.innerCount + k]
              outArr[outIdx] = source
            }
          }
        }
        try arrayToPointer(outArr, output: buffer.contents(), dtype: dtype)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
    }
  }

  override public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a, s.indices)
    let flatIndices = try await serialize {
      var flatIndices = [Int64](repeating: 0, count: s.indicesCount)
      try pointerToArray(s.indices.buffer.contents(), output: &flatIndices, dtype: .int64)
      return flatIndices
    }
    func apply<T: NumericTensorElement>(_ zero: T) async throws -> Tensor.Data {
      let buffer = try await allocate(length: s.scatterInCount * dtype.byteSize)
      try await serialize {
        var inArr = [T](repeating: zero, count: s.scatterOutCount)
        try pointerToArray(a.buffer.contents(), output: &inArr, dtype: dtype)
        var outArr = [T](repeating: zero, count: s.scatterInCount)
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
        try arrayToPointer(outArr, output: buffer.contents(), dtype: dtype)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
    }
  }

  override public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a, b)

    let aCount = rows * inner
    let bCount = inner * cols
    func apply<T: NumericTensorElement>(_ zero: T) async throws -> Tensor.Data {
      let buffer = try await allocate(length: rows * cols * dtype.byteSize)
      if !transA && !transB && !transOut && dtype == .float32 {
        try await serialize {
          let x = UnsafePointer<Float>(
            a.buffer.contents().bindMemory(to: Float.self, capacity: aCount))
          let y = UnsafePointer<Float>(
            b.buffer.contents().bindMemory(to: Float.self, capacity: aCount))
          let z = buffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
          vDSP_mmul(x, 1, y, 1, z, 1, vDSP_Length(rows), vDSP_Length(cols), vDSP_Length(inner))
        }
      } else {
        try await serialize {
          var arrA = [T](repeating: zero, count: aCount)
          var arrB = [T](repeating: zero, count: bCount)
          try pointerToArray(a.buffer.contents(), output: &arrA, dtype: dtype)
          try pointerToArray(b.buffer.contents(), output: &arrB, dtype: dtype)
          var arrC = [T](repeating: zero, count: rows * cols)

          func getA(_ i: Int, _ j: Int) -> T {
            if transA {
              arrA[i + j * rows]
            } else {
              arrA[i * inner + j]
            }
          }

          func getB(_ i: Int, _ j: Int) -> T {
            if transB {
              arrB[i + j * inner]
            } else {
              arrB[i * cols + j]
            }
          }

          func setC(_ i: Int, _ j: Int, _ x: T) {
            if transOut {
              arrC[i + j * rows] = x
            } else {
              arrC[i * cols + j] = x
            }
          }

          for i in 0..<rows {
            for j in 0..<cols {
              var acc = T(0.0)
              for k in 0..<inner {
                acc = acc + getA(i, k) * getB(k, j)
              }
              setC(i, j, acc)
            }
          }

          try arrayToPointer(arrC, output: buffer.contents(), dtype: dtype)
        }
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
    }
  }

  override public func elemwise(_ a: Tensor.Data, op: ElemwiseOp, count: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)
    let buffer = try await allocate(length: count * dtype.byteSize)
    try await serialize {
      var arr = [Float](repeating: 0, count: count)
      try pointerToArray(a.buffer.contents(), output: &arr, dtype: dtype)
      let cData = arr.map(op.apply)
      try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
    }
    return Tensor.Data(backend: self, buffer: buffer)
  }

  override public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    assert(inputs.count == innerCounts.count)
    for input in inputs {
      try await waitForData(input)
    }
    let totalInner = innerCounts.sum()
    let buffer = try await allocate(length: outerCount * totalInner * dtype.byteSize)
    try await serialize {
      var outOffset = 0
      for i in 0..<outerCount {
        for (input, innerCount) in zip(inputs, innerCounts) {
          let chunkSize = innerCount * dtype.byteSize
          let outPtr = buffer.contents().advanced(by: outOffset)
          let inPtr = input.buffer.contents().advanced(by: i * chunkSize)
          outPtr.copyMemory(from: inPtr, byteCount: chunkSize)
          outOffset += chunkSize
        }
      }
    }
    return Tensor.Data(backend: self, buffer: buffer)
  }

  override public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data
  {
    let buffer = try await allocate(length: shape.product() * Tensor.DType.int64.byteSize)
    try await serialize {
      let oldStrides = stridesForShape(shape)

      let permutedStrides = permutation.map { oldStrides[$0] }
      let newShape = permutation.map { shape[$0] }
      let newStrides = stridesForShape(newShape)
      var newIndices = [Int](repeating: 0, count: shape.product())
      for i in 0..<newIndices.count {
        let newIndex = zip(newStrides, newShape).map { stride, shape in (i / stride) % shape }
        let flatIndex = zip(newIndex, permutedStrides).map { $0 * $1 }.sum()
        assert(flatIndex >= 0 && flatIndex < newIndices.count, "bad flat index for \(newIndex)")
        newIndices[i] = flatIndex
      }
      try arrayToPointer(newIndices, output: buffer.contents(), dtype: .int64)
    }
    return Tensor.Data(backend: self, buffer: buffer)
  }

  override public func defaultRandom() async throws -> RandomGenerator {
    NativeRandomGenerator(cpuBackend: self)
  }

  override public func createRandom() async throws -> RandomGenerator {
    NativeRandomGenerator(cpuBackend: self)
  }

}

func stridesForShape(_ shape: [Int]) -> [Int] {
  var strides = [Int](repeating: 0, count: shape.count)
  for i in 0..<shape.count {
    strides[i] = shape[(i + 1)...].product()
  }
  return strides
}
