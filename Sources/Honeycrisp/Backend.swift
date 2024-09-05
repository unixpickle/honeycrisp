import Metal

public enum BackendError: Error {
  case notImplemented
  case failedToCreateMTLDevice
  case failedToCreateCommandQueue
  case allocationFailed(Int)
}

public protocol BackendHandle {

  var commandQueue: MTLCommandQueue? { get }

  func allocate(length: Int) throws -> MTLBuffer

  func binaryOp(
    _ a: Tensor.Data, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func equals(
    _ a: Tensor.Data, _ b: Tensor.Data, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func equals<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
    throws
    -> Tensor.Data

  func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func repeated(
    _ a: Tensor.Data, outerCount: Int, innerCount: Int, repeats: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data

  func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data
}

open class Backend {

  static var defaultBackend: Backend = CPUBackend()

  public func waitForData(_ data: [Tensor.Data]) async throws {
    for x in data {
      if let c = x.completeOnAllDevices, x.backend !== self {
        try await c.value
      }
    }
  }

  public func waitForData(_ a: Tensor.Data) async throws -> Tensor.Data {
    try await waitForData([a])
    return a
  }

  public func waitForData(_ a: Tensor.Data, _ b: Tensor.Data) async throws -> (
    Tensor.Data, Tensor.Data
  ) {
    try await waitForData([a, b])
    return (a, b)
  }

  public func execute<T>(_ work: (BackendHandle) throws -> T) async throws -> T {
    throw BackendError.notImplemented
  }

}

open class CPUBackend: Backend {

  open class Handle: BackendHandle {
    public let backend: Backend
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue?

    init(backend: Backend, device: MTLDevice, commandQueue: MTLCommandQueue? = nil) {
      self.backend = backend
      self.device = device
      self.commandQueue = commandQueue
    }

    public func allocate(length: Int) throws -> MTLBuffer {
      guard let result = device.makeBuffer(length: length, options: [.storageModeShared]) else {
        throw BackendError.allocationFailed(length)
      }
      return result
    }

    public func binaryOp(
      _ a: Tensor.Data, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
    ) throws
      -> Tensor.Data
    {
      func apply<T: NumericTensorElement>(_ x: T) throws -> Tensor.Data {
        var aData = [T](repeating: x, count: count)
        var bData = [T](repeating: x, count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
        let cData = op.apply(aData, bData)
        let buffer = try allocate(length: count * dtype.byteSize)
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(Int64(0))
      } else {
        return try apply(Float(0))
      }
    }

    public func binaryOp<T: NumericTensorElement>(
      _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T1: NumericTensorElement>(_ b: T1) throws -> Tensor.Data {
        var aData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        let cData = op.apply(aData, b)
        let buffer = try allocate(length: count * dtype.byteSize)
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(b.toInt64())
      } else {
        return try apply(b.toFloat())
      }
    }

    public func binaryOp<T: NumericTensorElement>(
      _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T1: NumericTensorElement>(_ a: T1) throws -> Tensor.Data {
        var bData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
        let cData = op.apply(a, bData)
        let buffer = try allocate(length: count * dtype.byteSize)
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(a.toInt64())
      } else {
        return try apply(a.toFloat())
      }
    }

    public func equals(
      _ a: Tensor.Data, _ b: Tensor.Data, count: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T1: NumericTensorElement>(_ x: T1) throws -> Tensor.Data {
        var aData = [T1](repeating: x, count: count)
        var bData = [T1](repeating: x, count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        try pointerToArray(b.buffer.contents(), output: &bData, dtype: dtype)
        let cData = zip(aData, bData).map { $0 == $1 }
        let buffer = try allocate(length: count * Tensor.DType.bool.byteSize)
        try arrayToPointer(cData, output: buffer.contents(), dtype: .bool)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(Int64(0))
      } else {
        return try apply(Float(0))
      }
    }

    public func equals<T: TensorElement>(
      _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T1: NumericTensorElement>(_ b: T1) throws -> Tensor.Data {
        var aData = [T1](repeating: T1(0.0), count: count)
        try pointerToArray(a.buffer.contents(), output: &aData, dtype: dtype)
        let cData = aData.map { $0 == b }
        let buffer = try allocate(length: count * Tensor.DType.bool.byteSize)
        try arrayToPointer(cData, output: buffer.contents(), dtype: .bool)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(b.toInt64())
      } else {
        return try apply(b.toFloat())
      }
    }

    public func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
      throws
      -> Tensor.Data
    {
      func apply<T: TensorElement>(_ x: T) throws -> Tensor.Data {
        var arr = [T](repeating: x, count: count)
        try pointerToArray(a.buffer.contents(), output: &arr, dtype: inType)
        let buffer = try allocate(length: count * outType.byteSize)
        try arrayToPointer(arr, output: buffer.contents(), dtype: outType)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if inType == .int64 {
        return try apply(Int64(0))
      } else {
        return try apply(Float(0))
      }
    }

    public func pow<T: NumericTensorElement>(
      _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T1: NumericTensorElement>(_ x: T1) throws -> Tensor.Data {
        var arr = [T1](repeating: x, count: count)
        try pointerToArray(a.buffer.contents(), output: &arr, dtype: dtype)
        let cData = arr.map { $0.pow(x) }
        let buffer = try allocate(length: count * dtype.byteSize)
        try arrayToPointer(cData, output: buffer.contents(), dtype: dtype)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(Int64(0))
      } else {
        return try apply(Float(0))
      }
    }

    public func reduce(
      _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T: NumericTensorElement>(_ x: T) throws -> Tensor.Data {
        var arr = [T](repeating: x, count: dims.inCount)
        try pointerToArray(a.buffer.contents(), output: &arr, dtype: dtype)

        switch op {
        case .sum:
          var arrOut = [T]()
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
          let buffer = try allocate(length: arrOut.count * dtype.byteSize)
          try arrayToPointer(arrOut, output: buffer.contents(), dtype: dtype)
          return Tensor.Data(backend: backend, buffer: buffer)
        case .argmin, .argmax:
          assert(dims.outCount > 0, "cannot apply op \(self) to empty dimension")
          var arrOut = [Int64]()
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
          let buffer = try allocate(length: arrOut.count * Tensor.DType.int64.byteSize)
          try arrayToPointer(arrOut, output: buffer.contents(), dtype: .int64)
          return Tensor.Data(backend: backend, buffer: buffer)
        }
      }
      if dtype == .int64 {
        return try apply(Int64(0))
      } else {
        return try apply(Float(0))
      }
    }

    public func repeated(
      _ a: Tensor.Data, outerCount: Int, innerCount: Int, repeats: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      let inData = a.buffer.contents()
      let outData = try allocate(length: outerCount * innerCount * repeats * dtype.byteSize)
      let innerBytes = dtype.byteSize * innerCount
      for i in 0..<outerCount {
        for j in 0..<repeats {
          let outBytes = outData.contents().advanced(
            by: (i * repeats + j) * innerBytes)
          let inBytes = inData.advanced(by: i * innerBytes)
          outBytes.copyMemory(from: inBytes, byteCount: innerBytes)
        }
      }
      return Tensor.Data(backend: backend, buffer: outData)
    }

    public func gather(
      _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      var flatIndices = [Int64](repeating: 0, count: s.indicesCount)
      try pointerToArray(s.indices.buffer.contents(), output: &flatIndices, dtype: .int64)

      if s.broadcasted {
        let innerSize = s.innerCount * dtype.byteSize
        let inData = a.buffer.contents()
        let outBuffer = try allocate(
          length: s.innerCount * flatIndices.count * innerSize)
        let outData = outBuffer.contents()
        for i in 0..<s.outerCount {
          for (j, idx) in flatIndices.enumerated() {
            let source = inData.advanced(by: i * s.middleCount * innerSize + Int(idx) * innerSize)
            let dst = outData.advanced(
              by: i * flatIndices.count * innerSize + j * innerSize)
            dst.copyMemory(from: source, byteCount: innerSize)
          }
        }
        return Tensor.Data(backend: backend, buffer: outBuffer)
      }

      func apply<T: TensorElement>(_ zero: T) throws -> Tensor.Data {
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
        let buffer = try allocate(length: s.scatterOutCount * dtype.byteSize)
        try arrayToPointer(outArr, output: buffer.contents(), dtype: dtype)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(Int64(0))
      } else {
        return try apply(Float(0))
      }
    }

    public func scatter(
      _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      var flatIndices = [Int64](repeating: 0, count: s.indicesCount)
      try pointerToArray(s.indices.buffer.contents(), output: &flatIndices, dtype: .int64)
      func apply<T: NumericTensorElement>(_ zero: T) throws -> Tensor.Data {
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
        let buffer = try allocate(length: s.scatterInCount * dtype.byteSize)
        try arrayToPointer(outArr, output: buffer.contents(), dtype: dtype)
        return Tensor.Data(backend: backend, buffer: buffer)
      }
      if dtype == .int64 {
        return try apply(Int64(0))
      } else {
        return try apply(Float(0))
      }
    }
  }

  private var queue = DispatchQueue(label: "cpu-backend-worker")
  private static var _global = CPUBackend()
  public static var global: CPUBackend { CPUBackend._global }

  private var _allocatorDevice: MTLDevice?
  private var allocatorDevice: MTLDevice {
    get throws {
      if let d = _allocatorDevice {
        return d
      }
      if let d = MTLCreateSystemDefaultDevice() {
        _allocatorDevice = d
        return d
      }
      throw BackendError.failedToCreateMTLDevice
    }
  }

  override public func execute<T>(_ work: (BackendHandle) throws -> T) async throws -> T {
    return try await withCheckedThrowingContinuation { continuation in
      queue.sync {
        do {
          continuation.resume(
            returning: try work(Handle(backend: self, device: try allocatorDevice)))
        } catch {
          continuation.resume(throwing: error)
        }
      }
    }
  }

}

open class MPSBackend: Backend {

  open class Handle: CPUBackend.Handle {
  }

  private var queue = DispatchQueue(label: "mps-backend-worker")
  public let device: MTLDevice
  private let commandQueue: MTLCommandQueue

  public init(device: MTLDevice? = nil, commandQueue: MTLCommandQueue? = nil) throws {
    if let device = device {
      self.device = device
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
      self.device = d
      guard let q = d.makeCommandQueue() else {
        throw BackendError.failedToCreateCommandQueue
      }
      self.commandQueue = q
    }
  }

  override public func execute<T>(_ work: (BackendHandle) throws -> T) async throws -> T {
    return try await withCheckedThrowingContinuation { continuation in
      queue.sync {
        do {
          continuation.resume(
            returning: try work(Handle(backend: self, device: device, commandQueue: commandQueue)))
        } catch {
          continuation.resume(throwing: error)
        }
      }
    }
  }

}
