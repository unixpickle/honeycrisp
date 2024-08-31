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
  func binaryOp(_ a: Tensor.Data, _ b: Tensor.Data, op: BinaryOp, count: Int, dtype: Tensor.DType)
    throws
    -> Tensor.Data
  func binaryOp<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: BinaryOp, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data
  func binaryOp<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: BinaryOp, count: Int, dtype: Tensor.DType
  )
    throws
    -> Tensor.Data
  func cast(_ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType)
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
      guard let result = device.makeBuffer(length: length) else {
        throw BackendError.allocationFailed(length)
      }
      return result
    }

    public func binaryOp(
      _ a: Tensor.Data, _ b: Tensor.Data, op: BinaryOp, count: Int, dtype: Tensor.DType
    ) throws
      -> Tensor.Data
    {
      func apply<T: TensorElement>(_ x: T) throws -> Tensor.Data {
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

    public func binaryOp<T: TensorElement>(
      _ a: Tensor.Data, _ b: T, op: BinaryOp, count: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T1: TensorElement>(_ b: T1) throws -> Tensor.Data {
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

    public func binaryOp<T: TensorElement>(
      _ a: T, _ b: Tensor.Data, op: BinaryOp, count: Int, dtype: Tensor.DType
    )
      throws
      -> Tensor.Data
    {
      func apply<T1: TensorElement>(_ a: T1) throws -> Tensor.Data {
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
