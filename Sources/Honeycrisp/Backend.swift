import Foundation
import Metal

public enum BackendError: Error {
  case notImplemented
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
    throw BackendError.notImplemented
  }

  public func binaryOp(
    _ a: Tensor.Data, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func equals(
    _ a: Tensor.Data, _ b: Tensor.Data, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func equals<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func repeated(
    _ a: Tensor.Data, outerCount: Int, innerCount: Int, repeats: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

  public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented
  }

}

open class CPUBackend: Backend {

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

    func apply<T: NumericTensorElement>(_ x: T) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        var aData = [T](repeating: x, count: count)
        var bData = [T](repeating: x, count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
        let cData = op.apply(aData, bData)
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
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
        var aData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        let cData = op.apply(aData, b)
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
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
        var bData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
        let cData = op.apply(a, bData)
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(a.toInt64())
    } else {
      return try await apply(a.toFloat())
    }
  }

  override public func equals(
    _ a: Tensor.Data, _ b: Tensor.Data, count: Int, dtype: Tensor.DType
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
        let cData = zip(aData, bData).map { $0 == $1 }
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

  override public func equals<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
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
        let cData = aData.map { $0 == b }
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

    func apply<T1: NumericTensorElement>(_ x: T1) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        var arr = [T1](repeating: x, count: count)
        try pointerToArray(a.buffer.contents(), output: &arr, dtype: dtype)
        let cData = arr.map { $0.pow(x) }
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
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
      let buffer = try await allocate(length: aCount * bCount * dtype.byteSize)
      try await serialize {
        var arrA = [T](repeating: zero, count: aCount)
        var arrB = [T](repeating: zero, count: bCount)
        try pointerToArray(a.buffer.contents(), output: &arrA, dtype: dtype)
        try pointerToArray(b.buffer.contents(), output: &arrB, dtype: dtype)
        var arrC = [T](repeating: zero, count: aCount * bCount)

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
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
    }
  }

}

open class MPSBackend: CPUBackend {

  private var queue = DispatchQueue(label: "mps-backend-worker")
  private var commandQueue: MTLCommandQueue? = nil

  public init(device: MTLDevice? = nil, commandQueue: MTLCommandQueue? = nil) throws {
    super.init()
    if let device = device {
      self._device = device
      if let commandQueue = commandQueue {
        self.commandQueue = commandQueue
      } else {
        if let q = device.makeCommandQueue() {
          self.commandQueue = q
        } else {
          throw BackendError.failedToCreateCommandQueue
        }
      }
    } else {
      guard let d = MTLCreateSystemDefaultDevice() else {
        throw BackendError.failedToCreateMTLDevice
      }
      self._device = d
      guard let q = d.makeCommandQueue() else {
        throw BackendError.failedToCreateCommandQueue
      }
      self.commandQueue = q
    }
  }

}
