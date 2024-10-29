import Accelerate
import Foundation
import Metal

public enum BackendError: Error {
  case notImplemented(String)
  case failedToCreateMTLDevice
  case failedToCreateCommandQueue
  case allocationFailed(Int)
  case kernelFailed(String)
}

open class Backend {

  public enum TensorOrScalar<T: TensorElement> {
    case tensor(Tensor.Data)
    case scalar(T)

    internal func getPointer(_ dtype: Tensor.DType) throws -> (UnsafeMutableRawPointer, Bool) {
      switch self {
      case .tensor(let t):
        (t.buffer.contents(), false)
      case .scalar(let s):
        try {
          let output = UnsafeMutableRawPointer.allocate(byteCount: dtype.byteSize, alignment: 16)
          try arrayToPointer([s], output: output, dtype: dtype)
          return (output, true)
        }()
      }
    }
  }

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

  /// A lightweight thread-safe FIFO queue.
  public class Queue<T: Sendable> {
    private var items: [T] = []
    private var lock: NSLock = NSLock()
    private var closed: Bool = false
    private var sem: DispatchSemaphore = DispatchSemaphore(value: 0)

    /// Push an item to the queue.
    /// This should never be called after close().
    public func put(_ x: T) {
      lock.lock()
      alwaysAssert(!closed, "cannot put() on queue after closing")
      items.insert(x, at: 0)
      lock.unlock()
      sem.signal()
    }

    /// Wait for the next item on the queue.
    ///
    /// Returns nil forever once the queue has been closed and depleted.
    public func get() -> T? {
      sem.wait()
      lock.lock()
      let item = items.popLast()
      if closed {
        // We don't want to block future get() calls.
        sem.signal()
      }
      lock.unlock()
      return item
    }

    /// Finish writing to the queue and unblock all future get() calls.
    public func close() {
      lock.lock()
      closed = true
      lock.unlock()
      sem.signal()
    }
  }

  /// A background thread which executes blocks serially.
  ///
  /// Once this is deinitialized, the background thread will complete its
  /// remaining work and then exit.
  public class WorkerThread {
    public typealias Job = () -> Void

    private let queue = Queue<Job>()
    private var thread: Thread? = nil

    deinit {
      queue.close()
    }

    public init() {
      thread = Thread { [queue = self.queue] in
        while true {
          guard let job = queue.get() else {
            return
          }
          autoreleasepool {
            job()
          }
        }
      }
      thread!.name = "Backend.WorkerThread"
      thread!.start()
    }

    public func schedule(_ job: @escaping Job) {
      queue.put(job)
    }
  }

  public func allocate(length: Int) async throws -> MTLBuffer {
    throw BackendError.notImplemented("allocate")
  }

  public func binaryOp(
    a: Tensor.Data, aCount: Int, b: Tensor.Data, bCount: Int, op: NumericBinaryOp, count: Int,
    dtype: Tensor.DType
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

  public func mulAdd(
    input: Tensor.Data, inputCount: Int, coeff: Tensor.Data, coeffCount: Int, bias: Tensor.Data,
    biasCount: Int, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("mulAdd")
  }

  public func addMul(
    input: Tensor.Data, inputCount: Int, bias: Tensor.Data, biasCount: Int, coeff: Tensor.Data,
    coeffCount: Int, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("addMul")
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

  public func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("clamp")
  }

  public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("reduce")
  }

  public func logSoftmax(
    _ a: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmax")
  }

  public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("logSoftmaxGrad")
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

  public func when<T>(
    _ mask: Tensor.Data, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("when")
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

  public func batchedMatmul(
    matrixCount: Int, a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool,
    rows: Int, inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("batchedMatmul")
  }

  public func tril(_ a: Tensor.Data, batch: Int, rows: Int, cols: Int, dtype: Tensor.DType)
    async throws -> Tensor.Data
  {
    throw BackendError.notImplemented("tril")
  }

  public func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1D")
  }

  public func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DTranspose")
  }

  public func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv1DKernelGrad")
  }

  public func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2D")
  }

  public func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2DTranspose")
  }

  public func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    throw BackendError.notImplemented("conv2DKernelGrad")
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

  public enum Allocator {
    case device
    case bucket
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
      -> Tensor.Data
    {
      let buffer = try await backend.allocate(length: count * dtype.byteSize)
      try await cpuBackend.serialize {
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
      try await cpuBackend.serialize {
        let ints = (0..<count).map { _ in Int64.random(in: range) }
        try arrayToPointer(ints, output: buffer.contents(), dtype: .int64)
      }
      return Tensor.Data(backend: backend, buffer: buffer)
    }
  }

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

  // Allocator state.
  internal var allocBucketsLock = NSLock()
  internal var allocBuckets: [Int: [MTLBuffer]]? = nil

  internal var worker = Backend.WorkerThread()

  override public init() {
  }

  public init(allocator: Allocator) throws {
    super.init()
    try initAllocator(allocator)
  }

  internal func initAllocator(_ allocator: Allocator) throws {
    switch allocator {
    case .device:
      ()
    case .bucket:
      allocBuckets = [:]
    }
  }

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

  internal func waitForData(_ xs: Tensor.Data...) async throws {
    for x in xs {
      if let waiter = x.completeOnAllDevices {
        try await waiter.value
      }
    }
  }

  override public func allocate(length: Int) async throws -> MTLBuffer {
    if allocBuckets != nil {
      let bucket = nextAllocatorBucket(length: length)
      let rawResult =
        if let item = allocBucketsLock.withLock({ allocBuckets![bucket]?.popLast() }) {
          item
        } else {
          try await {
            return try await allocRaw(length: bucket)
          }()
        }
      let maybeResult = (try device).makeBuffer(
        bytesNoCopy: rawResult.contents(), length: max(length, 1), options: [.storageModeShared],
        deallocator: { [weak self] _, _ in
          if let self = self {
            self.allocBucketsLock.withLock {
              if self.allocBuckets![bucket] == nil {
                self.allocBuckets![bucket] = [rawResult]
              } else {
                self.allocBuckets![bucket]!.append(rawResult)
              }
            }
          }
        }
      )
      guard let result = maybeResult else {
        throw BackendError.allocationFailed(length)
      }
      return result
    }

    return try await allocRaw(length: length)
  }

  internal func allocRaw(length: Int) async throws -> MTLBuffer {
    return try await serialize { [self] in
      let maybeBuffer = (try device).makeBuffer(
        length: max(1, length), options: [.storageModeShared])
      guard let result = maybeBuffer else {
        throw BackendError.allocationFailed(length)
      }
      #if DEBUG
        // Fill the data with garbage to catch methods that assume
        // zero initialization.
        let bound = result.contents().bindMemory(to: UInt8.self, capacity: length)
        let noise = (0..<3).map({ _ in UInt8.random(in: 0...255) })
        for i in 0..<length {
          bound[i] = noise[i % 3]
        }
      #endif
      return result
    }
  }

  override public func binaryOp(
    a: Tensor.Data, aCount: Int, b: Tensor.Data, bCount: Int, op: NumericBinaryOp, count: Int,
    dtype: Tensor.DType
  ) async throws
    -> Tensor.Data
  {
    try await waitForData(a, b)

    func apply<T: NumericTensorElement>(_: T.Type) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        if dtype == .float32 && (op != .mod) && aCount == count && bCount == count {
          let x = UnsafePointer<Float>(
            a.buffer.contents().bindMemory(to: Float.self, capacity: aCount))
          let y = UnsafePointer<Float>(
            b.buffer.contents().bindMemory(to: Float.self, capacity: bCount))
          let z = buffer.contents().bindMemory(to: Float.self, capacity: count)
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
          try readBuffer(T.self, a.buffer, count: aCount, dtype: dtype) { aData in
            try readBuffer(T.self, b.buffer, count: bCount, dtype: dtype) { bData in
              try writeBuffer(T.self, buffer, count: count, dtype: dtype) { cData in
                for i in 0..<count {
                  cData[i] = op.apply(aData[i % aCount], bData[i % bCount])
                }
              }
            }
          }
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
        if dtype == .float32 && (op != .mod) {
          let x = UnsafePointer<Float>(
            a.buffer.contents().bindMemory(to: Float.self, capacity: count))
          var bScalar =
            switch op {
            case .add, .mul: b.toFloat()
            case .div: 1 / b.toFloat()
            case .sub: -b.toFloat()
            case .mod: fatalError()
            }
          let z = buffer.contents().bindMemory(to: Float.self, capacity: count)
          switch op {
          case .add, .sub:
            vDSP_vsadd(x, 1, &bScalar, z, 1, vDSP_Length(count))
          case .mul, .div:
            vDSP_vsmul(x, 1, &bScalar, z, 1, vDSP_Length(count))
          case .mod:
            fatalError()
          }
        } else {
          try readBuffer(T1.self, a.buffer, count: count, dtype: dtype) { aData in
            try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { cData in
              for (i, x) in aData.enumerated() {
                cData[i] = op.apply(x, b)
              }
            }
          }
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
        if dtype == .float32 && (op != .mod) {
          let x = UnsafePointer<Float>(
            b.buffer.contents().bindMemory(to: Float.self, capacity: count))
          var aFloat = a.toFloat()
          var neg1 = Float(-1)
          let z = buffer.contents().bindMemory(to: Float.self, capacity: count)
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
          try readBuffer(T1.self, b.buffer, count: count, dtype: dtype) { bData in
            try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { cData in
              for (i, x) in bData.enumerated() {
                cData[i] = op.apply(a, x)
              }
            }
          }
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

  override public func mulAdd(
    input: Tensor.Data, inputCount: Int, coeff: Tensor.Data, coeffCount: Int, bias: Tensor.Data,
    biasCount: Int, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(input, coeff, bias)
    func apply<T1: NumericTensorElement>(_: T1.Type) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        if dtype == .float32 && biasCount == count && coeffCount == count && inputCount == count {
          let x = UnsafePointer<Float>(
            input.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let coeff = UnsafePointer<Float>(
            coeff.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let bias = UnsafePointer<Float>(
            bias.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let output = buffer.contents().bindMemory(to: Float.self, capacity: count)
          vDSP_vma(x, 1, coeff, 1, bias, 1, output, 1, vDSP_Length(count))
        } else {
          try readBuffer(T1.self, input.buffer, count: inputCount, dtype: dtype) { inData in
            try readBuffer(T1.self, coeff.buffer, count: coeffCount, dtype: dtype) { coeff in
              try readBuffer(T1.self, bias.buffer, count: biasCount, dtype: dtype) { bias in
                try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { outData in
                  for i in 0..<count {
                    outData[i] =
                      inData[i % inputCount] * coeff[i % coeffCount] + bias[i % biasCount]
                  }
                }
              }
            }
          }
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

  override public func addMul(
    input: Tensor.Data, inputCount: Int, bias: Tensor.Data, biasCount: Int, coeff: Tensor.Data,
    coeffCount: Int, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(input, coeff, bias)
    func apply<T1: NumericTensorElement>(_: T1.Type) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        if dtype == .float32 && biasCount == count && coeffCount == count && inputCount == count {
          let x = UnsafePointer<Float>(
            input.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let coeff = UnsafePointer<Float>(
            coeff.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let bias = UnsafePointer<Float>(
            bias.buffer.contents().bindMemory(to: Float.self, capacity: count))
          let output = buffer.contents().bindMemory(to: Float.self, capacity: count)
          vDSP_vam(x, 1, bias, 1, coeff, 1, output, 1, vDSP_Length(count))
        } else {
          try readBuffer(T1.self, input.buffer, count: inputCount, dtype: dtype) { inData in
            try readBuffer(T1.self, coeff.buffer, count: coeffCount, dtype: dtype) { coeff in
              try readBuffer(T1.self, bias.buffer, count: biasCount, dtype: dtype) { bias in
                try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { outData in
                  for i in 0..<count {
                    outData[i] =
                      (inData[i % inputCount] + bias[i % biasCount]) * coeff[i % coeffCount]
                  }
                }
              }
            }
          }
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
        try readBuffer(T1.self, a.buffer, count: count, dtype: dtype) { aData in
          try readBuffer(T1.self, b.buffer, count: count, dtype: dtype) { bData in
            try writeBuffer(Bool.self, buffer, count: count, dtype: .bool) { cData in
              for (i, (x, y)) in zip(aData, bData).enumerated() {
                cData[i] = op.apply(x, y)
              }
            }
          }
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
        try readBuffer(T1.self, a.buffer, count: count, dtype: dtype) { aData in
          try writeBuffer(Bool.self, buffer, count: count, dtype: .bool) { cData in
            for (i, x) in aData.enumerated() {
              cData[i] = op.apply(x, b)
            }
          }
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
        try readBuffer(T1.self, b.buffer, count: count, dtype: dtype) { bData in
          try writeBuffer(Bool.self, buffer, count: count, dtype: .bool) { cData in
            for (i, x) in bData.enumerated() {
              cData[i] = op.apply(a, x)
            }
          }
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
          try readBuffer(T1.self, a.buffer, count: count, dtype: dtype) { arr in
            try writeBuffer(T1.self, buffer, count: count, dtype: dtype) { out in
              for (i, x) in arr.enumerated() {
                out[i] = x.pow(b)
              }
            }
          }
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

  override public func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)

    alwaysAssert(min != nil || max != nil, "cannot use clamp() without bounds")

    func apply<T1: NumericTensorElement>(_ min: T1?, _ max: T1?) async throws -> Tensor.Data {
      let buffer = try await allocate(length: count * dtype.byteSize)
      try await serialize {
        try readBuffer(T1.self, a.buffer, count: count, dtype: dtype) { arr in
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
      return Tensor.Data(backend: self, buffer: buffer)
    }
    if dtype == .int64 {
      return try await apply(min?.toInt64(), max?.toInt64())
    } else {
      return try await apply(min?.toFloat(), max?.toFloat())
    }
  }

  override public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)

    if dtype == .float32 && op == .sum {
      let buffer = try await allocate(length: dims.outCount * dtype.byteSize)
      for i in 0..<dims.outerCount {
        for j in 0..<dims.innerCount {
          let inPtr = UnsafePointer<Float>(
            a.buffer.contents().advanced(by: 4 * (j + i * dims.reduceCount * dims.innerCount))
              .bindMemory(
                to: Float.self, capacity: dims.reduceCount))
          let y = UnsafeMutablePointer<Float>(
            buffer.contents().advanced(by: 4 * (i * dims.innerCount + j)).bindMemory(
              to: Float.self, capacity: dims.reduceCount))
          vDSP_sve(inPtr, vDSP_Stride(dims.innerCount), y, vDSP_Length(dims.reduceCount))
        }
      }
      return Tensor.Data(backend: self, buffer: buffer)
    }

    func apply<T: NumericTensorElement>(_: T.Type) async throws -> Tensor.Data {
      switch op {
      case .sum:
        let buffer = try await allocate(length: dims.outCount * dtype.byteSize)
        try await serialize {
          try readBuffer(T.self, a.buffer, count: dims.inCount, dtype: dtype) { arr in
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
        return Tensor.Data(backend: self, buffer: buffer)
      case .argmin, .argmax:
        let buffer = try await allocate(length: dims.outCount * Tensor.DType.int64.byteSize)
        try await serialize {
          try readBuffer(T.self, a.buffer, count: dims.inCount, dtype: dtype) { arr in
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
        return Tensor.Data(backend: self, buffer: buffer)
      }
    }
    if dtype == .int64 {
      return try await apply(Int64.self)
    } else {
      return try await apply(Float.self)
    }
  }

  override public func logSoftmax(
    _ a: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let totalCount = outerCount * middleCount * innerCount
    let buffer = try await allocate(length: totalCount * dtype.byteSize)
    try await serialize {
      try readBuffer(Float.self, a.buffer, count: totalCount, dtype: dtype) { arr in
        try writeBuffer(Float.self, buffer, count: totalCount, dtype: dtype) { arrOut in
          for i in 0..<outerCount {
            let outerOffset = i * innerCount * middleCount
            for j in 0..<innerCount {
              var max: Float = 0
              for k in 0..<middleCount {
                let item = arr[j + k * innerCount + outerOffset]
                if k == 0 || item > max {
                  max = item
                }
              }
              var expSum: Float = 0
              for k in 0..<middleCount {
                let item = arr[j + k * innerCount + outerOffset]
                expSum += exp(item - max)
              }
              let logSum = log(expSum) + max
              for k in 0..<middleCount {
                let idx = j + k * innerCount + outerOffset
                let item = arr[idx]
                arrOut[idx] = item - logSum
              }
            }
          }
        }
      }
    }
    return Tensor.Data(backend: self, buffer: buffer)
  }

  override public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let totalCount = outerCount * middleCount * innerCount
    let buffer = try await allocate(length: totalCount * dtype.byteSize)
    try await serialize {
      try readBuffer(Float.self, a.buffer, count: totalCount, dtype: dtype) { arr in
        try readBuffer(Float.self, outGrad.buffer, count: totalCount, dtype: dtype) { arrGrad in
          try writeBuffer(Float.self, buffer, count: totalCount, dtype: dtype) { arrOut in
            for i in 0..<outerCount {
              let outerOffset = i * innerCount * middleCount
              for j in 0..<innerCount {
                var max: Float = 0
                var gradSum: Float = 0
                for k in 0..<middleCount {
                  let idx = j + k * innerCount + outerOffset
                  let item = arr[idx]
                  gradSum += arrGrad[idx]
                  if k == 0 || item > max {
                    max = item
                  }
                }
                var expSum: Float = 0
                for k in 0..<middleCount {
                  let item = arr[j + k * innerCount + outerOffset]
                  expSum += exp(item - max)
                }
                let logSum = log(expSum) + max
                for k in 0..<middleCount {
                  let idx = j + k * innerCount + outerOffset
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
    return Tensor.Data(backend: self, buffer: buffer)
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

    let outBuffer = try await allocate(length: s.gatherOutCount * dtype.byteSize)

    if s.broadcasted {
      let innerSize = s.innerCount * dtype.byteSize
      try await serialize {
        try readBuffer(Int64.self, s.indices.buffer, count: s.indicesCount, dtype: .int64) {
          flatIndices in
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
      }
      return Tensor.Data(backend: self, buffer: outBuffer)
    }

    func apply<T: TensorElement>(_ zero: T) async throws -> Tensor.Data {
      try await serialize {
        try readBuffer(Int64.self, s.indices.buffer, count: s.indicesCount, dtype: .int64) {
          flatIndices in
          try readBuffer(T.self, a.buffer, count: s.gatherInCount, dtype: dtype) { inArr in
            try writeBuffer(T.self, outBuffer, count: s.gatherOutCount, dtype: dtype) { outArr in
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
            }
          }
        }
      }
      return Tensor.Data(backend: self, buffer: outBuffer)
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
    func apply<T: NumericTensorElement>(_ zero: T) async throws -> Tensor.Data {
      let buffer = try await allocate(length: s.gatherInCount * dtype.byteSize)
      try await serialize {
        try readBuffer(Int64.self, s.indices.buffer, count: s.indicesCount, dtype: .int64) {
          flatIndices in
          try readBuffer(T.self, a.buffer, count: s.gatherOutCount, dtype: dtype) { inArr in
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
            try arrayToPointer(outArr, output: buffer.contents(), dtype: dtype)
          }
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

  override public func when<T>(
    _ mask: Tensor.Data, _ a: TensorOrScalar<T>, _ b: TensorOrScalar<T>, _: T.Type, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if case .tensor(let t) = a {
      try await waitForData(t)
    }
    if case .tensor(let t) = b {
      try await waitForData(t)
    }

    let (aData, aIsScalar) = try a.getPointer(dtype)
    defer {
      if aIsScalar {
        aData.deallocate()
      }
    }
    let (bData, bIsScalar) = try b.getPointer(dtype)
    defer {
      if bIsScalar {
        bData.deallocate()
      }
    }
    let output = try await allocate(length: count * dtype.byteSize)

    try await serialize {
      let contents = output.contents()
      let bools = mask.buffer.contents().bindMemory(to: UInt8.self, capacity: count)
      for i in 0..<count {
        let off = i * dtype.byteSize
        if bools[i] != 0 {
          contents.advanced(by: off).copyMemory(
            from: aData.advanced(by: aIsScalar ? 0 : off), byteCount: dtype.byteSize)
        } else {
          contents.advanced(by: off).copyMemory(
            from: bData.advanced(by: bIsScalar ? 0 : off), byteCount: dtype.byteSize)
        }
      }
    }

    return Tensor.Data(backend: self, buffer: output)
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
    try await waitForData(a, b)

    let aCount = rows * inner
    let bCount = inner * cols
    func apply<T: NumericTensorElement>(_ zero: T) async throws -> Tensor.Data {
      let buffer = try await allocate(length: matrixCount * rows * cols * dtype.byteSize)
      if !transA && !transB && !transOut && dtype == .float32 {
        try await serialize {
          for i in 0..<matrixCount {
            let x = UnsafePointer<Float>(
              a.buffer.contents().advanced(by: i * aCount * dtype.byteSize).bindMemory(
                to: Float.self, capacity: aCount))
            let y = UnsafePointer<Float>(
              b.buffer.contents().advanced(by: i * bCount * dtype.byteSize).bindMemory(
                to: Float.self, capacity: aCount))
            let z = buffer.contents().advanced(by: i * rows * cols * dtype.byteSize).bindMemory(
              to: Float.self, capacity: rows * cols)
            vDSP_mmul(x, 1, y, 1, z, 1, vDSP_Length(rows), vDSP_Length(cols), vDSP_Length(inner))
          }
        }
      } else {
        try await serialize {
          var arrA = [T](repeating: zero, count: matrixCount * aCount)
          var arrB = [T](repeating: zero, count: matrixCount * bCount)
          try pointerToArray(a.buffer.contents(), output: &arrA, dtype: dtype)
          try pointerToArray(b.buffer.contents(), output: &arrB, dtype: dtype)
          var arrC = [T](repeating: zero, count: matrixCount * rows * cols)

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

  override public func tril(_ a: Tensor.Data, batch: Int, rows: Int, cols: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)

    let rowSize = cols * dtype.byteSize
    let outBuf = try await allocate(length: batch * rows * rowSize)
    try await serialize {
      let inPtr = a.buffer.contents()
      let outPtr = outBuf.contents()
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
    return Tensor.Data(backend: self, buffer: outBuf)
  }

  internal func convNd<Dim: SpatialDim>(
    _ config: ConvConfig<Dim>, batch: Int, image: Tensor.Data, kernel: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(kernel, image)

    let imageShape = config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape = config.outputTensorShape(batch: batch)

    let outBuf = try await allocate(length: outShape.product() * dtype.byteSize)

    func apply<T: NumericTensorElement>(_ zero: T) async throws -> Tensor.Data {
      try await serialize {
        assert(kernel.buffer.allocatedSize >= dtype.byteSize * kernelShape.product())
        assert(image.buffer.allocatedSize >= dtype.byteSize * imageShape.product())
        try readBuffer(T.self, kernel.buffer, count: kernelShape.product(), dtype: dtype) {
          arrKernel in
          try readBuffer(T.self, image.buffer, count: imageShape.product(), dtype: dtype) {
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
      return Tensor.Data(backend: self, buffer: outBuf)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
    }
  }

  internal func convNdTranspose<Dim: SpatialDim>(
    _ config: ConvConfig<Dim>, batch: Int, image: Tensor.Data, kernel: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(kernel, image)

    let imageShape = config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape = config.outputTensorShape(batch: batch)

    let outBuf = try await allocate(length: imageShape.product() * dtype.byteSize)

    func apply<T: NumericTensorElement>(_ zero: T) async throws -> Tensor.Data {
      try await serialize {
        assert(kernel.buffer.allocatedSize >= dtype.byteSize * kernelShape.product())
        assert(image.buffer.allocatedSize >= dtype.byteSize * outShape.product())
        try readBuffer(T.self, kernel.buffer, count: kernelShape.product(), dtype: dtype) {
          arrKernel in
          try readBuffer(T.self, image.buffer, count: outShape.product(), dtype: dtype) {
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
      return Tensor.Data(backend: self, buffer: outBuf)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
    }
  }

  internal func convNdKernelGrad<Dim: SpatialDim>(
    _ config: ConvConfig<Dim>, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForData(image, outGrad)

    let imageShape = config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape = config.outputTensorShape(batch: batch)

    let outBuf = try await allocate(length: kernelShape.product() * dtype.byteSize)

    func apply<T: NumericTensorElement>(_ zero: T) async throws -> Tensor.Data {
      try await serialize {
        try readBuffer(T.self, image.buffer, count: imageShape.product(), dtype: dtype) {
          arrImage in
          try readBuffer(T.self, outGrad.buffer, count: outShape.product(), dtype: dtype) {
            arrOutGrad in
            try writeBuffer(T.self, outBuf, count: kernelShape.product(), dtype: dtype) { arrOut in
              let getImage = config.lazy(from: arrImage, shape: imageShape)
              let getOutGrad = config.lazy(from: arrOutGrad, shape: outShape)
              let outputFn = config.lazyKernelGrad(image: getImage, outGrad: getOutGrad)
              outputFn.unlazify(to: &arrOut, channelsLast: false)
            }
          }
        }
      }
      return Tensor.Data(backend: self, buffer: outBuf)
    }
    if dtype == .int64 {
      return try await apply(Int64(0))
    } else {
      return try await apply(Float(0))
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

  override public func elemwise(_ a: Tensor.Data, op: ElemwiseOp, count: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    try await waitForData(a)
    let buffer = try await allocate(length: count * dtype.byteSize)
    try await serialize {
      try readBuffer(Float.self, a.buffer, count: count, dtype: dtype) { arr in
        try writeBuffer(Float.self, buffer, count: count, dtype: dtype) { out in
          for (i, x) in arr.enumerated() {
            out[i] = op.apply(x)
          }
        }
      }
    }
    return Tensor.Data(backend: self, buffer: buffer)
  }

  override public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(inputs.count == innerCounts.count)
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
        var flatIndex = 0
        for j in 0..<newStrides.count {
          flatIndex += permutedStrides[j] * ((i / newStrides[j]) % newShape[j])
        }
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

func nextAllocatorBucket(length: Int) -> Int {
  if length < 4096 {
    return 4096
  } else {
    var i = 4096
    while i < length && i >= 4096 {
      i *= 2
    }
    alwaysAssert(i >= 4096, "allocation size overflow: \(length)")
    return i
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
  _: T1.Type, _ buf: MTLBuffer, count: Int, dtype: Tensor.DType,
  _ fn: (UnsafeBufferPointer<T1>) throws -> T
) throws -> T {
  assert(buf.allocatedSize >= count * dtype.byteSize)
  if dtype == T1.dtype {
    return try fn(
      UnsafeBufferPointer(
        start: buf.contents().bindMemory(to: T1.self, capacity: count), count: count))
  }
  var arr = [T1](repeating: T1(0.0), count: count)
  try pointerToArray(buf.contents(), output: &arr, dtype: dtype)
  return try arr.withUnsafeBufferPointer(fn)
}

func writeBuffer<T, T1: TensorElement>(
  _: T1.Type, _ buf: MTLBuffer, count: Int, dtype: Tensor.DType,
  _ fn: (inout UnsafeMutableBufferPointer<T1>) throws -> T
) throws -> T {
  assert(buf.allocatedSize >= count * dtype.byteSize)
  if dtype == T1.dtype {
    var buf = UnsafeMutableBufferPointer(
      start: buf.contents().bindMemory(to: T1.self, capacity: count), count: count)
    return try fn(&buf)
  }
  var arr = [T1](repeating: T1(0.0), count: count)
  try pointerToArray(buf.contents(), output: &arr, dtype: dtype)
  let result = try arr.withUnsafeMutableBufferPointer(fn)
  try arrayToPointer(arr, output: buf.contents(), dtype: dtype)
  return result
}
