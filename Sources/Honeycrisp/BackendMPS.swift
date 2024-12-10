import Accelerate
import Foundation
import HCBacktrace
@preconcurrency import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// A ``Backend`` which uses Metal Performance Shaders and custom kernels to
/// implement operations on the GPU.
open class MPSBackend: CPUBackend, @unchecked Sendable {

  internal class Deallocator: @unchecked Sendable {
    let fn: () -> Void
    let trace: [CodeLocation]
    var called: Bool = false
    let lock = NSLock()

    public init(
      fn: @escaping () -> Void,
      function: StaticString = #function,
      file: StaticString = #filePath,
      line: UInt = #line
    ) {
      self.fn = fn
      self.trace = Backtrace.current + [CodeLocation(function: function, file: file, line: line)]
    }

    @recordCaller
    private func _callAsFunction() {
      lock.withLock {
        #alwaysAssert(!called, "double free")
        called = true
      }
      fn()
    }

    deinit {
      #if DEBUG
        // In debug mode, this will help us catch bugs where we forgot to release resources.
        assert(
          called, "deallocator was never called, trace of creation:\n\n\(Backtrace.format(trace))")
      #else
        // In release mode, this is more likely caused by an exception returning from a backend
        // method before using a buffer, so we want to actually free the buffer.
        if !called {
          fn()
        }
      #endif
    }
  }

  internal class GPUData: Tensor.Data, @unchecked Sendable {

    public let backend: MPSBackend
    public let buffer: MTLBuffer
    public let completion: Task<Void, Error>?
    public let deallocator: Deallocator?

    public init(
      backend: MPSBackend,
      buffer: MTLBuffer,
      completion: Task<Void, Error>?,
      deallocator: Deallocator?
    ) {
      self.backend = backend
      self.buffer = buffer
      self.completion = completion
      self.deallocator = deallocator
    }

    public var byteCount: Int { buffer.allocatedSize }

    public func onCPU<T>(_ fn: (_: UnsafeRawPointer) async throws -> T) async throws -> T {
      if let completion = completion {
        let _ = try await completion.value
      }
      return try await fn(buffer.contents())
    }

    public func mutateOnCPU<T>(_ fn: (_: UnsafeMutableRawPointer) async throws -> T) async throws
      -> T
    {
      if let completion = completion {
        let _ = try await completion.value
      }
      return try await fn(buffer.contents())
    }

    deinit {
      if let deallocator = deallocator {
        let completion = completion
        Task.detached {
          // Don't deallocate until we have actually completed the kernel call.
          if let completion = completion {
            let _ = try await completion.value
          }
          deallocator()
        }
      }
    }
  }

  public enum BroadcastError: Error {
    case shapeTooLarge(String)
    case dimTooLarge(String)
  }

  internal typealias StridesItem = (
    UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
  )

  @recordCaller
  internal static func _createStridesItem(_ arr: [Int], padding: UInt32) throws -> StridesItem {
    if arr.count > 8 {
      throw BroadcastError.shapeTooLarge(
        "maximum supported number of dims in broadcast shape is 8, got \(arr.count) in shape \(arr)"
      )
    }
    func at(_ i: Int) throws -> UInt32 {
      if i >= arr.count {
        return padding
      }
      if arr[i] >= Int(UInt32.max) {
        throw BroadcastError.dimTooLarge("dimension \(arr[i]) is too large to fit in UInt32")
      }
      return UInt32(arr[i])
    }
    return (
      try at(0),
      try at(1),
      try at(2),
      try at(3),
      try at(4),
      try at(5),
      try at(6),
      try at(7)
    )
  }

  internal typealias Strides = (dimCount: UInt32, shape: StridesItem, strides: StridesItem)

  @recordCaller
  internal static func _createStrides(_ s: BroadcastStrides) throws -> Strides {
    if s.isNoOp {
      return (
        dimCount: UInt32(1),
        shape: try createStridesItem([s.shape.product()], padding: 1),
        strides: try createStridesItem([1], padding: 0)
      )
    }

    // Coalesce consecutive contiguous dimensions to produce a simpler
    // shape for the kernel to handle.
    var shape = s.shape
    var strides = s.strides
    var i = shape.count - 1
    while i > 0 {
      if strides[i] != 0 && strides[i - 1] != 0 && (strides[i - 1] == shape[i] * strides[i]) {
        strides[i - 1] = strides[i]
        shape[i - 1] *= shape[i]
        shape.remove(at: i)
        strides.remove(at: i)
      } else if strides[i] == 0 && strides[i - 1] == 0 {
        shape[i - 1] *= shape[i]
        shape.remove(at: i)
        strides.remove(at: i)
      }
      i -= 1
    }

    return (
      dimCount: UInt32(strides.count),
      shape: try createStridesItem(shape, padding: 1),
      strides: try createStridesItem(strides, padding: 0)
    )
  }

  public enum Allocator {
    case device
    case bucket
    case heap(Int)
  }

  private struct MatmulKey: Hashable {
    let transA: Bool
    let transB: Bool
    let batch: Int
    let rows: Int
    let cols: Int
    let inner: Int
    let dtype: MPSDataType
  }

  private struct Conv2DKey: Hashable {
    enum Kind {
      case forward
      case transpose
      case kernelGrad
    }

    let conv: Conv2DConfig
    let batch: Int
    let kind: Kind
    let dtype: MPSDataType
  }

  private struct TwoToOneGraph {
    let graph: MPSGraph
    let inputA: MPSGraphTensor
    let inputB: MPSGraphTensor
    let output: MPSGraphTensor
  }

  private struct OneToOneGraph {
    let graph: MPSGraph
    let input: MPSGraphTensor
    let output: MPSGraphTensor
  }

  private struct ReduceKey: Hashable {
    let op: ReduceOp
    let dims: ReduceDims
    let dtype: MPSDataType
  }

  private struct SoftmaxKey: Hashable {
    let outerCount: Int
    let middleCount: Int
    let innerCount: Int
    let dtype: MPSDataType
  }

  private struct ScatterKey: Hashable {
    var valueShape: [Int]
    var axis: Int
    var indexShape: [Int]
    var indexStrides: [Int]
    let dtype: MPSDataType
  }

  public class MPSRandomGenerator: RandomGenerator, @unchecked Sendable {
    public let mpsBackend: MPSBackend

    override open var stateCount: Int {
      2
    }

    override open var stateDType: Tensor.DType {
      .int64
    }

    init(mpsBackend: MPSBackend, seed: Int) {
      self.mpsBackend = mpsBackend
      super.init(
        backend: mpsBackend,
        state: mpsBackend.use { Tensor(data: [seed, 0], dtype: .int64) }
      )
    }

    override open func _seed(_ x: Int) async throws -> Tensor.Data {
      try await mpsBackend.collection([x, 0], reverse: false, dtype: .int64)
    }

    override open func _sample(
      state: Tensor.Data, count: Int, dist: RandomDist, dtype: Tensor.DType
    )
      async throws -> (
        sample: Tensor.Data, state: Tensor.Data
      )
    {
      #alwaysAssert(count <= 0xffff_ffff, "count exceeds UInt32 size: \(count)")

      let functionName =
        "\(dist == .normal ? "randn" : "rand")_\(MPSBackend.CastTypes[dtype]!)"
      let (inState, inStateCb) = try await mpsBackend.gpuBuffer(state)
      let (output, outputCb) = try await mpsBackend.allocateBuf(count * dtype.byteSize)
      let (outState, outStateCb) = try await mpsBackend.allocateBuf(16)

      return try await mpsBackend.serialize { [self] in
        let completion = try mpsBackend.completionBufferAndEncoder(
          label: "sample", deallocators: [inStateCb]
        ) { buf, enc in
          let state = try mpsBackend.getFunction(name: functionName)
          try mpsBackend.setArguments(
            enc, .buffer(inState), .buffer(outState), .buffer(output), .uint(UInt32(count)))
          let chunkSize = dist == .normal ? 2 : 4
          let totalThreads = (count + chunkSize - 1) / chunkSize
          mpsBackend.dispatch1D(enc, state: state, threadCount: totalThreads)
        }
        return (
          sample: GPUData(
            backend: mpsBackend, buffer: output, completion: completion, deallocator: outputCb),
          state: GPUData(
            backend: mpsBackend, buffer: outState, completion: completion, deallocator: outStateCb)
        )
      }
    }

    override open func _sample(state: Tensor.Data, count: Int, in range: Range<Int64>)
      async throws -> (
        sample: Tensor.Data, state: Tensor.Data
      )
    {
      #alwaysAssert(count <= 0xffff_ffff, "count exceeds UInt32 size: \(count)")

      let (inState, inStateCb) = try await mpsBackend.gpuBuffer(state)
      let (output, outputCb) = try await mpsBackend.allocateBuf(count * Tensor.DType.int64.byteSize)
      let (outState, outStateCb) = try await mpsBackend.allocateBuf(16)

      return try await mpsBackend.serialize { [self] in
        let completion = try mpsBackend.completionBufferAndEncoder(
          label: "sampleInt", deallocators: [inStateCb]
        ) { buf, enc in
          let state = try mpsBackend.getFunction(name: "rand_long")
          try mpsBackend.setArguments(
            enc,
            .buffer(inState),
            .buffer(outState),
            .buffer(output),
            .int64(range.lowerBound),
            .int64(range.upperBound &- range.lowerBound),
            .uint(UInt32(count))
          )
          mpsBackend.dispatch1D(enc, state: state, threadCount: count)
        }
        return (
          sample: GPUData(
            backend: mpsBackend, buffer: output, completion: completion, deallocator: outputCb),
          state: GPUData(
            backend: mpsBackend, buffer: outState, completion: completion, deallocator: outStateCb)
        )
      }
    }
  }

  private static let CastTypes: [Tensor.DType: String] = [
    .float16: "half", .float32: "float", .int64: "long",
  ]

  public let device: MTLDevice

  private var queue = DispatchQueue(label: "mps-backend-worker")
  private var commandQueue: MTLCommandQueue? = nil
  private var matmuls: [MatmulKey: TwoToOneGraph] = [:]
  private var conv2D: [Conv2DKey: TwoToOneGraph] = [:]
  private var scatters: [ScatterKey: TwoToOneGraph] = [:]
  private var reductions: [ReduceKey: OneToOneGraph] = [:]
  private var logSoftmaxes: [SoftmaxKey: OneToOneGraph] = [:]
  private var logSoftmaxGrads: [SoftmaxKey: TwoToOneGraph] = [:]
  private var _defaultRandomMPS: MPSRandomGenerator? = nil
  private var functions: [String: MTLComputePipelineState] = [:]

  // Allocator state.
  internal var allocBucketsLock = NSLock()
  internal var allocBuckets: [Int: [MTLBuffer]]? = nil
  internal var heap: MTLHeap? = nil
  internal var argumentHeap: MTLHeap? = nil

  public init(
    device: MTLDevice? = nil, commandQueue: MTLCommandQueue? = nil, allocator: Allocator = .device
  ) throws {
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
    super.init()
    try self.initAllocator(allocator)
    try self.initFunctions()
    self._defaultRandomMPS = MPSRandomGenerator(
      mpsBackend: self, seed: Int.random(in: 0..<1_000_000_000))
  }

  internal func initFunctions() throws {
    guard let fileURL = Bundle.module.url(forResource: "kernels", withExtension: "txt") else {
      throw BackendError.failedToLoadKernels("kernels.txt could not be found")
    }
    guard let sourceStr = String(data: try Data(contentsOf: fileURL), encoding: .utf8) else {
      throw BackendError.failedToLoadKernels("kernels source code was not a utf-8 string")
    }
    let library = try device.makeLibrary(source: sourceStr, options: MTLCompileOptions())

    // Lookup all functions in the library.
    var names = ["axis_permutation"]
    for type in ["half", "float"] {
      for op in [
        "vector_pow", "vector_pow_scaled", "log", "recip", "exp", "sigmoid", "sigmoid_grad",
        "gelu_approx", "gelu_approx_grad", "gelu_exact", "gelu_exact_grad", "erf", "erf_grad",
        "sin", "cos", "minus_sin", "relu", "relu_grad", "abs", "abs_grad", "rand",
        "randn", "normalize_inner", "normalize_inner_grad", "log_softmax",
        "log_softmax_grad", "adamw",
      ] {
        names.append("\(op)_\(type)")
      }
    }
    names.append("rand_long")
    for type in ["char", "short", "int", "long"] {
      names.append("when_\(type)")
      names.append("repeat_\(type)")
      names.append("strided_copy_\(type)")
      names.append("broadcast_\(type)")
      for op in ["gather", "scatter"] {
        names.append("\(op)_\(type)")
      }
      for op in ["xor", "and", "or"] {
        names.append("\(op)_\(type)")
      }
    }
    for op in ["add", "sub", "mul", "div", "mod", "lt", "gt", "le", "ge", "eq"] {
      for type in ["half", "float", "long"] {
        for args in ["vv", "sv", "vs"] {
          names.append("\(op)_\(args)_\(type)")
        }
      }
    }
    for t1 in ["half", "float", "long"] {
      names.append("add_mul_\(t1)")
      names.append("mul_add_\(t1)")
      names.append("clamp_\(t1)")
      for t2 in ["half", "float", "long"] {
        if t1 == t2 {
          continue
        }
        names.append("cast_\(t1)_\(t2)")
      }
    }
    for name in names {
      guard let f = library.makeFunction(name: name) else {
        throw BackendError.kernelFailed("could not create kernel with name '\(name)'")
      }
      functions[name] = try device.makeComputePipelineState(function: f)
    }
  }

  internal func initAllocator(_ allocator: Allocator) throws {
    let argDesc = MTLHeapDescriptor()
    argDesc.size = 1_048_576
    argDesc.hazardTrackingMode = .tracked
    argDesc.storageMode = .shared
    guard let argHeap = device.makeHeap(descriptor: argDesc) else {
      throw BackendError.allocationFailed(argDesc.size)
    }
    argumentHeap = argHeap

    switch allocator {
    case .device:
      ()
    case .bucket:
      allocBuckets = [:]
    case .heap(let size):
      let desc = MTLHeapDescriptor()
      desc.size = size
      desc.hazardTrackingMode = .tracked
      desc.storageMode = .shared
      guard let h = device.makeHeap(descriptor: desc) else {
        throw BackendError.allocationFailed(size)
      }
      heap = h
    }
  }

  override open func allocate(_ byteCount: Int) async throws -> Tensor.Data {
    let (buf, deallocator) = try await allocateBuf(byteCount)
    return GPUData(backend: self, buffer: buf, completion: nil, deallocator: deallocator)
  }

  internal func allocateBuf(_ byteCount: Int) async throws -> (
    MTLBuffer, Deallocator?
  ) {
    if allocBuckets != nil {
      let bucket = nextAllocatorBucket(byteCount)
      let rawResult =
        if let item = allocBucketsLock.withLock({ allocBuckets![bucket]?.popLast() }) {
          item
        } else {
          try await allocRaw(bucket)
        }
      return (
        rawResult,
        Deallocator { [weak self] in
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
    }
    return (try await allocRaw(byteCount), nil)
  }

  private func allocRaw(_ byteCount: Int) async throws -> MTLBuffer {
    return try await serialize { [self] in
      try allocRawSync(byteCount)
    }
  }

  private func allocRawSync(_ byteCount: Int) throws -> MTLBuffer {
    let maybeBuffer =
      if let heap = heap {
        heap.makeBuffer(length: max(1, byteCount), options: [.storageModeShared])
      } else {
        device.makeBuffer(length: max(1, byteCount), options: [.storageModeShared])
      }

    guard let result = maybeBuffer else {
      throw BackendError.allocationFailed(byteCount)
    }
    #if DEBUG
      // Fill the data with garbage to catch methods that assume
      // zero initialization.
      let bound = result.contents().bindMemory(to: UInt8.self, capacity: byteCount)
      let noise = (0..<3).map({ _ in UInt8.random(in: 0...255) })
      for i in 0..<byteCount {
        bound[i] = noise[i % 3]
      }
    #endif
    return result
  }

  internal func allocateArgument(_ byteCount: Int) throws -> MTLBuffer {
    if let heap = argumentHeap,
      let result = heap.makeBuffer(length: max(1, byteCount), options: [.storageModeShared])
    {
      return result
    }
    throw BackendError.allocationFailed(byteCount)
  }

  @recordCaller
  internal func _gpuBuffer(_ data: Tensor.Data) async throws -> (MTLBuffer, Deallocator) {
    if let data = data as? GPUData, data.backend === self {
      return (data.buffer, Deallocator { [data] () in let _ = data })
    }

    // Create an MPSBuffer which, while in use, maintains a background Task
    // inside of an onCPU call on the original data.
    struct SendableResult: @unchecked Sendable {
      let value: (MTLBuffer, Deallocator)
    }
    let result: SendableResult = try await withCheckedThrowingContinuation { outerContinuation in
      Task.detached {
        try await data.onCPU { cpuBuffer in
          try await withCheckedThrowingContinuation {
            (innerContinuation: CheckedContinuation<(), Error>) in
            guard
              let buffer = self.device.makeBuffer(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: cpuBuffer),
                length: data.byteCount,
                options: .init(arrayLiteral: [.storageModeShared])
              )
            else {
              outerContinuation.resume(throwing: BackendError.failedToCreateMTLBuffer)
              innerContinuation.resume(throwing: BackendError.failedToCreateMTLBuffer)
              return
            }
            outerContinuation.resume(
              returning: SendableResult(value: (buffer, Deallocator { innerContinuation.resume() }))
            )
          }
        }
      }
    }
    return result.value
  }

  internal func gpuBuffer(_ data: Tensor.Data?) async throws -> (MTLBuffer?, Deallocator?) {
    guard let data = data else {
      return (nil, nil)
    }
    let (x, y) = try await gpuBuffer(data)
    return (x, y)
  }

  internal func getFunction(name: String) throws -> MTLComputePipelineState {
    guard let f = functions[name] else {
      throw BackendError.kernelFailed("no kernel with name '\(name)'")
    }
    return f
  }

  override open func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, scale: T, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.pow(a, b, scale: scale, scales: scales, count: count, dtype: dtype)
    }

    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")
    let (aBuf, aCb) = try await gpuBuffer(a)
    let (scalesBuf, scalesCb) = try await gpuBuffer(scales)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "pow",
        deallocators: [aCb] + (scalesCb == nil ? [] : [scalesCb!])
      ) {
        buf, enc in
        let funcName =
          "vector_pow\(scalesBuf == nil ? "" : "_scaled")_\(MPSBackend.CastTypes[dtype]!)"
        let state = try getFunction(name: funcName)
        if let scalesBuf = scalesBuf {
          try setArguments(
            enc,
            .buffer(aBuf),
            .buffer(scalesBuf),
            .buffer(output),
            .float(b.toFloat()),
            .float(scale.toFloat()),
            .uint(UInt32(count)))
        } else {
          try setArguments(
            enc,
            .buffer(aBuf),
            .buffer(output),
            .float(b.toFloat()),
            .float(scale.toFloat()),
            .uint(UInt32(count)))
        }
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func elemwise(
    _ a: Tensor.Data, op: ElemwiseOp, scales: Tensor.Data?, count: Int, dtype: Tensor.DType
  ) async throws -> Tensor.Data {
    if dtype != .float16 && dtype != .float32 {
      return try await super.elemwise(a, op: op, scales: scales, count: count, dtype: dtype)
    }

    let typeName = MPSBackend.CastTypes[dtype]!

    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let namePrefix =
      switch op {
      case .sin:
        "sin"
      case .cos:
        "cos"
      case .minusSin:
        "minus_sin"
      case .relu:
        "relu"
      case .reluGrad:
        "relu_grad"
      case .abs:
        "abs"
      case .absGrad:
        "abs_grad"
      case .log:
        "log"
      case .recip:
        "recip"
      case .geluApprox:
        "gelu_approx"
      case .geluApproxGrad:
        "gelu_approx_grad"
      case .geluExact:
        "gelu_exact"
      case .geluExactGrad:
        "gelu_exact_grad"
      case .erf:
        "erf"
      case .erfGrad:
        "erf_grad"
      case .exp:
        "exp"
      case .sigmoid:
        "sigmoid"
      case .sigmoidGrad:
        "sigmoid_grad"
      }
    let functionName = "\(namePrefix)_\(typeName)"

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (scalesBuf, scalesCb) = try await gpuBuffer(scales)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "elemwise",
        deallocators: [aCb] + (scalesCb == nil ? [] : [scalesCb!])
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(enc, .buffer(aBuf), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
        if let scalesBuf = scalesBuf {
          let state = try getFunction(name: "mul_vv_\(typeName)")
          try setArguments(
            enc,
            .buffer(output),
            .buffer(scalesBuf),
            .buffer(output),
            .opaque(
              (
                MPSBackend.createStrides(BroadcastStrides(shape: [count], strides: [1])),
                MPSBackend.createStrides(BroadcastStrides(shape: [count], strides: [1])),
                UInt32(count)
              )))
          dispatch1D(enc, state: state, threadCount: count)
        }
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  internal static func binaryOpType(_ dtype: Tensor.DType) -> String? {
    switch dtype {
    case .float16:
      "half"
    case .float32:
      "float"
    case .int64:
      "long"
    default:
      nil
    }
  }

  internal static func binaryOpName(_ op: NumericBinaryOp) -> String {
    switch op {
    case .add:
      "add"
    case .mul:
      "mul"
    case .sub:
      "sub"
    case .div:
      "div"
    case .mod:
      "mod"
    }
  }

  override open func binaryOp(
    _ a: BroadcastData, _ b: BroadcastData, op: NumericBinaryOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = Self.binaryOpType(dtype) else {
      return try await super.binaryOp(a, b, op: op, dtype: dtype)
    }
    let count = a.strides.shape.product()
    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.binaryOpName(op)
    let functionName = "\(opName)_vv_\(typeName)"

    let (aBuf, aCb) = try await gpuBuffer(a.data)
    let (bBuf, bCb) = try await gpuBuffer(b.data)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)
    let aStrides = try MPSBackend.createStrides(a.strides)
    let bStrides = try MPSBackend.createStrides(b.strides)
    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "binaryOp", deallocators: [aCb, bCb]
      ) {
        buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc,
          .buffer(aBuf),
          .buffer(bBuf),
          .buffer(output),
          .opaque((aStrides, bStrides, UInt32(count)))
        )
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = Self.binaryOpType(dtype) else {
      return try await super.binaryOp(a, b, op: op, count: count, dtype: dtype)
    }

    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.binaryOpName(op)
    let functionName = "\(opName)_vs_\(typeName)"

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "binaryOp", deallocators: [aCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(aBuf), .float(b.toFloat()), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = Self.binaryOpType(dtype) else {
      return try await super.binaryOp(a, b, op: op, count: count, dtype: dtype)
    }

    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.binaryOpName(op)
    let functionName = "\(opName)_sv_\(typeName)"

    let (bBuf, bCb) = try await gpuBuffer(b)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "binaryOp", deallocators: [bCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .float(a.toFloat()), .buffer(bBuf), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  internal static func comparisonOpName(_ op: ComparisonOp) -> String {
    switch op {
    case .less:
      "lt"
    case .lessEqual:
      "le"
    case .greater:
      "gt"
    case .greaterEqual:
      "ge"
    case .equal:
      "eq"
    }
  }

  override open func compare(
    _ a: BroadcastData, _ b: BroadcastData, op: ComparisonOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = Self.binaryOpType(dtype) else {
      return try await super.compare(a, b, op: op, dtype: dtype)
    }

    let count = a.strides.shape.product()
    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.comparisonOpName(op)
    let functionName = "\(opName)_vv_\(typeName)"

    let (aBuf, aCb) = try await gpuBuffer(a.data)
    let (bBuf, bCb) = try await gpuBuffer(b.data)
    let (output, outputCb) = try await allocateBuf(count * Tensor.DType.bool.byteSize)
    let aStrides = try MPSBackend.createStrides(a.strides)
    let bStrides = try MPSBackend.createStrides(b.strides)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "compare", deallocators: [aCb, bCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(aBuf), .buffer(bBuf), .buffer(output),
          .opaque((aStrides, bStrides, UInt32(count)))
        )
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func compare<T: TensorElement>(
    _ a: Tensor.Data, _ b: T, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = Self.binaryOpType(dtype) else {
      return try await super.compare(a, b, op: op, count: count, dtype: dtype)
    }

    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.comparisonOpName(op)
    let functionName = "\(opName)_vs_\(typeName)"

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (output, outputCb) = try await allocateBuf(count * Tensor.DType.bool.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "compare", deallocators: [aCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(aBuf), .float(b.toFloat()), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func compare<T: TensorElement>(
    _ a: T, _ b: Tensor.Data, op: ComparisonOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = Self.binaryOpType(dtype) else {
      return try await super.compare(a, b, op: op, count: count, dtype: dtype)
    }

    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.comparisonOpName(op)
    let functionName = "\(opName)_sv_\(typeName)"

    let (bBuf, bCb) = try await gpuBuffer(b)
    let (output, outputCb) = try await allocateBuf(count * Tensor.DType.bool.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "compare", deallocators: [bCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .float(a.toFloat()), .buffer(bBuf), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  public static func bitwiseOpName(_ op: BitwiseOp) -> String {
    switch op {
    case .or:
      "or"
    case .and:
      "and"
    case .xor:
      "xor"
    }
  }

  override open func bitwiseOp(
    _ a: BroadcastData, _ b: BroadcastData, op: BitwiseOp, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let count = a.strides.shape.product()
    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.bitwiseOpName(op)
    let functionName = "\(opName)_\(dtype.metalSizeType)"

    let (aBuf, aCb) = try await gpuBuffer(a.data)
    let (bBuf, bCb) = try await gpuBuffer(b.data)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)
    let aStrides = try MPSBackend.createStrides(a.strides)
    let bStrides = try MPSBackend.createStrides(b.strides)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "bitwiseOp", deallocators: [aCb, bCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(aBuf), .buffer(bBuf), .buffer(output),
          .opaque((aStrides, bStrides, UInt32(count)))
        )
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func bitwiseOp<T: TensorElementBitPattern>(
    _ a: Tensor.Data, _ b: T, op: BitwiseOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    #alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName = Self.bitwiseOpName(op)
    let functionName = "\(opName)_\(dtype.metalSizeType)"

    let (aBuf, aCb) = try await gpuBuffer(a)
    let bData = b.bitsForBitwiseOp
    #alwaysAssert(bData.count == dtype.byteSize)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "bitwiseOp", deallocators: [aCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(aBuf), .data(Data(bData)), .buffer(output),
          .opaque(
            (
              MPSBackend.createStrides(BroadcastStrides(shape: [count], strides: [1])),
              MPSBackend.createStrides(BroadcastStrides(shape: [count], strides: [0])),
              UInt32(count)
            )))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await addMulOrMulAdd(
      input: input, a: coeff, b: bias, method: "mul_add", dtype: dtype)
  }

  override open func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await addMulOrMulAdd(
      input: input, a: bias, b: coeff, method: "add_mul", dtype: dtype)
  }

  private func addMulOrMulAdd(
    input: BroadcastData, a: BroadcastData, b: BroadcastData, method: String, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let count = input.strides.shape.product()
    #alwaysAssert(count <= UInt32.max, "\(method) cannot operate on as many as \(count) values")
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      if method == "mul_add" {
        return try await super.mulAdd(input: input, coeff: a, bias: b, dtype: dtype)
      } else {
        return try await super.addMul(input: input, bias: a, coeff: b, dtype: dtype)
      }
    }
    let functionName = "\(method)_\(typeName)"

    let (inBuf, inCb) = try await gpuBuffer(input.data)
    let (aBuf, aCb) = try await gpuBuffer(a.data)
    let (bBuf, bCb) = try await gpuBuffer(b.data)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)
    let inputStrides = try MPSBackend.createStrides(input.strides)
    let aStrides = try MPSBackend.createStrides(a.strides)
    let bStrides = try MPSBackend.createStrides(b.strides)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "addMulOrMulAdd", deallocators: [inCb, aCb, bCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(inBuf), .buffer(aBuf), .buffer(bBuf),
          .buffer(output),
          .opaque(inputStrides),
          .opaque(aStrides),
          .opaque(bStrides),
          .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  /// Perform an AdamW update.
  override open func adamW(
    param: Tensor.Data,
    grad: Tensor.Data,
    moment1: Tensor.Data,
    moment2: Tensor.Data,
    beta1: Float,
    beta2: Float,
    eps: Float,
    weightDecay: Float,
    lr: Float,
    step: Float,
    count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> (param: Tensor.Data, moment1: Tensor.Data, moment2: Tensor.Data)
  {
    #alwaysAssert(count <= UInt32.max, "normalize cannot operate on as many as \(count) values")
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      return try await super.adamW(
        param: param,
        grad: grad,
        moment1: moment1,
        moment2: moment2,
        beta1: beta1,
        beta2: beta2,
        eps: eps,
        weightDecay: weightDecay,
        lr: lr,
        step: step,
        count: count,
        dtype: dtype
      )
    }
    let functionName = "adamw_\(typeName)"

    let (paramBuf, paramCb) = try await gpuBuffer(param)
    let (gradBuf, gradCb) = try await gpuBuffer(grad)
    let (moment1Buf, moment1Cb) = try await gpuBuffer(moment1)
    let (moment2Buf, moment2Cb) = try await gpuBuffer(moment2)
    let (paramOutBuf, paramOutCb) = try await allocateBuf(count * dtype.byteSize)
    let (moment1OutBuf, moment1OutCb) = try await allocateBuf(count * dtype.byteSize)
    let (moment2OutBuf, moment2OutCb) = try await allocateBuf(count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "adamw", deallocators: [paramCb, gradCb, moment1Cb, moment2Cb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(paramBuf), .buffer(gradBuf), .buffer(moment1Buf), .buffer(moment2Buf),
          .buffer(paramOutBuf), .buffer(moment1OutBuf), .buffer(moment2OutBuf),
          .float(beta1), .float(beta2), .float(eps), .float(weightDecay), .float(lr),
          .float(step), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return (
        param: GPUData(
          backend: self, buffer: paramOutBuf, completion: completion, deallocator: paramOutCb),
        moment1: GPUData(
          backend: self, buffer: moment1OutBuf, completion: completion, deallocator: moment1OutCb),
        moment2: GPUData(
          backend: self, buffer: moment2OutBuf, completion: completion, deallocator: moment2OutCb)
      )
    }
  }

  override open func normalize<T: NumericTensorElement>(
    input: Tensor.Data, dims: ReduceDims, eps: T, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    if dims.innerCount == 1 && (dtype == .float16 || dtype == .float32) {
      let totalCount = dims.outerCount * dims.reduceCount * dims.innerCount
      let (aBuf, aCb) = try await gpuBuffer(input)
      let (output, outputCb) = try await allocateBuf(totalCount * dtype.byteSize)
      return try await serialize { [self] in
        let completion = try completionBufferAndEncoder(
          label: "normalize", deallocators: [aCb]
        ) { buf, enc in
          try setArguments(
            enc,
            .buffer(aBuf),
            .buffer(output),
            .float(eps.toFloat()),
            .uint(UInt32(dims.reduceCount))
          )
          let functionName = "normalize_inner_\(MPSBackend.CastTypes[dtype]!)"
          let state = try getFunction(name: functionName)
          let gridSize = MTLSize(width: dims.outerCount * 256, height: 1, depth: 1)
          let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
          enc.setComputePipelineState(state)
          enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        }
        return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
      }
    } else {
      return try await super.normalize(input: input, dims: dims, eps: eps, dtype: dtype)
    }
  }

  override open func normalizeGrad<T: NumericTensorElement>(
    input: Tensor.Data, outGrad: Tensor.Data, dims: ReduceDims, eps: T, dtype: Tensor.DType
  )
    async throws -> Tensor.Data
  {
    if dims.innerCount == 1 && (dtype == .float16 || dtype == .float32) {
      let totalCount = dims.outerCount * dims.reduceCount * dims.innerCount
      let (aBuf, aCb) = try await gpuBuffer(input)
      let (gradBuf, gradCb) = try await gpuBuffer(outGrad)
      let (output, outputCb) = try await allocateBuf(totalCount * dtype.byteSize)
      return try await serialize { [self] in
        let completion = try completionBufferAndEncoder(
          label: "normalizeGrad", deallocators: [aCb, gradCb]
        ) { buf, enc in
          try setArguments(
            enc,
            .buffer(aBuf),
            .buffer(gradBuf),
            .buffer(output),
            .float(eps.toFloat()),
            .uint(UInt32(dims.reduceCount))
          )
          let functionName = "normalize_inner_grad_\(MPSBackend.CastTypes[dtype]!)"
          let state = try getFunction(name: functionName)
          let gridSize = MTLSize(width: dims.outerCount * 256, height: 1, depth: 1)
          let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
          enc.setComputePipelineState(state)
          enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        }
        return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
      }
    } else {
      return try await super.normalizeGrad(
        input: input, outGrad: outGrad, dims: dims, eps: eps, dtype: dtype)
    }
  }

  override open func cast(
    _ a: Tensor.Data, count: Int, inType: Tensor.DType, outType: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let inTypeName = MPSBackend.CastTypes[inType],
      let outTypeName = MPSBackend.CastTypes[outType]
    else {
      return try await super.cast(a, count: count, inType: inType, outType: outType)
    }

    let functionName = "cast_\(inTypeName)_\(outTypeName)"

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (output, outputCb) = try await allocateBuf(count * outType.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "cast", deallocators: [aCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(enc, .buffer(aBuf), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      return try await super.clamp(a, min: min, max: max, count: count, dtype: dtype)
    }
    #alwaysAssert(count <= UInt32.max, "cannot apply clamp() to \(count) values")

    let functionName = "clamp_\(typeName)"

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "clamp", deallocators: [aCb]
      ) { buf, enc in
        let state = try getFunction(name: functionName)
        if dtype == .int64 {
          try setArguments(
            enc, .buffer(aBuf), .buffer(output), .int64(min?.toInt64() ?? 0),
            .int64(max?.toInt64() ?? 0), .uint(UInt32(min != nil ? 1 : 0)),
            .uint(UInt32(max != nil ? 1 : 1)),
            .uint(UInt32(count)))
        } else {
          try setArguments(
            enc, .buffer(aBuf), .buffer(output), .float(min?.toFloat() ?? 0),
            .float(max?.toFloat() ?? 0), .uint(UInt32(min != nil ? 1 : 0)),
            .uint(UInt32(max != nil ? 1 : 0)),
            .uint(UInt32(count)))
        }
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  internal func tensorOrScalarBuffer<T>(_ obj: TensorOrScalar<T>, _ dtype: Tensor.DType)
    async throws -> (MTLBuffer, Deallocator)
  {
    switch obj {
    case .tensor(let t):
      try await gpuBuffer(t.data)
    case .scalar(let s, _):
      try await {
        let (buf, cb) = try await allocateBuf(dtype.byteSize)
        try arrayToPointer([s], output: buf.contents(), dtype: dtype)
        return (buf, cb ?? Deallocator { () in () })
      }()
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
    #alwaysAssert(count < UInt32.max, "cannot apply when() to \(count) elements")

    let (maskBuf, maskCb) = try await gpuBuffer(mask.data)
    let (aBuf, aCb) = try await tensorOrScalarBuffer(a, dtype)
    let (bBuf, bCb) = try await tensorOrScalarBuffer(b, dtype)

    let maskStrides = try MPSBackend.createStrides(mask.strides)
    let aStrides = try MPSBackend.createStrides(a.strides)
    let bStrides = try MPSBackend.createStrides(b.strides)

    let (output, outputCb) = try await allocateBuf(count * dtype.byteSize)
    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "when", deallocators: [maskCb, aCb, bCb]
      ) { buf, enc in
        let typeName = dtype.metalSizeType
        let functionName = "when_\(typeName)"
        let state = try getFunction(name: functionName)
        try setArguments(
          enc,
          .buffer(maskBuf),
          .buffer(aBuf),
          .buffer(bBuf),
          .buffer(output),
          .opaque(
            (
              maskStrides,
              aStrides,
              bStrides,
              UInt32(count)
            ))
        )
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func broadcast(_ a: BroadcastData, dtype: Tensor.DType) async throws -> Tensor.Data
  {
    let outCount = a.strides.shape.product()
    #alwaysAssert(
      outCount <= Int(UInt32.max),
      "cannot apply repeat kernel to this many values")

    let typeName = dtype.metalSizeType

    let (aBuf, aCb) = try await gpuBuffer(a.data)
    let (output, outputCb) = try await allocateBuf(outCount * dtype.byteSize)
    let aStrides = try MPSBackend.createStrides(a.strides)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "broadcast", deallocators: [aCb]
      ) { buf, enc in
        let state = try getFunction(name: "broadcast_\(typeName)")
        try setArguments(
          enc,
          .buffer(aBuf),
          .buffer(output),
          .opaque(aStrides),
          .uint(UInt32(outCount))
        )
        dispatch1D(enc, state: state, threadCount: outCount)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let outCount = dims.outCount
    #alwaysAssert(
      outCount <= Int(UInt32.max),
      "cannot apply repeat kernel to this many values")

    let typeName = dtype.metalSizeType

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (output, outputCb) = try await allocateBuf(outCount * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "repeated", deallocators: [aCb]
      ) { buf, enc in
        let state = try getFunction(name: "repeat_\(typeName)")
        try setArguments(
          enc,
          .buffer(aBuf),
          .buffer(output),
          .opaque((UInt32(dims.innerCount), UInt32(dims.outerCount), UInt32(dims.repeatCount)))
        )
        dispatch1D(enc, state: state, threadCount: outCount)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    #alwaysAssert(inputs.count == innerCounts.count)
    var inBufs = [MTLBuffer]()
    var inDeallocators = [Deallocator]()
    for input in inputs {
      let (buf, d) = try await gpuBuffer(input)
      inBufs.append(buf)
      inDeallocators.append(d)
    }
    let inDeallocatorsFinal = inDeallocators
    let inBufsFinal = inBufs

    let totalInner = innerCounts.sum()
    let (buffer, bufferCb) = try await allocateBuf(outerCount * totalInner * dtype.byteSize)

    let typeName = dtype.metalSizeType

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "concat", deallocators: inDeallocatorsFinal
      ) { buf, enc in
        for (i, innerCount) in innerCounts.enumerated() {
          let offset = innerCounts[..<i].sum()
          let state = try getFunction(name: "strided_copy_\(typeName)")
          try setArguments(
            enc,
            .buffer(inBufsFinal[i]),
            .buffer(buffer),
            .opaque((UInt32(innerCount), UInt32(totalInner), UInt32(outerCount), UInt32(offset)))
          )
          dispatch1D(enc, state: state, threadCount: innerCount * outerCount)
        }
      }
      return GPUData(backend: self, buffer: buffer, completion: completion, deallocator: bufferCb)
    }
  }

  override open func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let (aBuf, aCb) = try await gpuBuffer(a)
    let (idxBuf, idxCb) = try await gpuBuffer(s.indices.data)
    let (output, outputCb) = try await allocateBuf(s.gatherOutCount * dtype.byteSize)
    let indexStrides = try MPSBackend.createStrides(s.indices.strides)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(
        label: "gather", deallocators: [aCb, idxCb]
      ) { buf, enc in
        let typeName = dtype.metalSizeType
        let functionName = "gather_\(typeName)"
        let state = try getFunction(name: functionName)
        try setArguments(
          enc,
          .buffer(aBuf),
          .buffer(idxBuf),
          .buffer(output),
          .uint(UInt32(s.outerCount)),
          .uint(UInt32(s.outCount)),
          .uint(UInt32(s.middleCount)),
          .uint(UInt32(s.innerCount)),
          .opaque(indexStrides)
        )
        dispatch1D(enc, state: state, threadCount: s.gatherOutCount)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  override open func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let (aBuf, aCb) = try await gpuBuffer(a)
    let (idxBuf, idxCb) = try await gpuBuffer(s.indices.data)
    let indexStrides = try MPSBackend.createStrides(s.indices.strides)

    let outBytes = s.gatherInCount * dtype.byteSize
    let (output, outputCb) = try await allocateBuf(outBytes)

    if !s.indicesAreUnique, let mpsDType = dtype.mpsDType {
      return try await serialize { [self] in
        let scatter = self.createScatter(scatter: s, dtype: mpsDType)
        let completion = completionBuffer(
          label: "scatter", deallocators: [aCb, idxCb]
        ) { buf in
          scatter.graph.encode(
            to: buf as! MPSCommandBuffer,
            feeds: [
              scatter.inputA: MPSGraphTensorData(
                MPSVector(
                  buffer: aBuf,
                  descriptor: MPSVectorDescriptor(
                    length: s.gatherOutCount, dataType: mpsDType))),
              scatter.inputB: MPSGraphTensorData(
                MPSVector(
                  buffer: idxBuf,
                  descriptor: MPSVectorDescriptor(length: s.indices.dataCount, dataType: .int64))),
            ],
            targetOperations: nil,
            resultsDictionary: [
              scatter.output: MPSGraphTensorData(
                MPSVector(
                  buffer: output,
                  descriptor: MPSVectorDescriptor(length: s.gatherInCount, dataType: mpsDType)))
            ],
            executionDescriptor: nil)
        }
        return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
      }
    }

    if !s.indicesAreUnique {
      // Fallback on CPU implementation if for some reason there is a dtype
      // that MPS cannot handle above.
      return try await super.scatter(a, s, dtype: dtype)
    }

    return try await serialize { [self] in
      let completion = try completionBuffer(
        label: "scatterUnique", deallocators: [aCb, idxCb]
      ) { buf in
        let typeName = dtype.metalSizeType
        let functionName = "scatter_\(typeName)"
        let state = try getFunction(name: functionName)

        let blitEncoder = buf.makeBlitCommandEncoder()!
        blitEncoder.fill(buffer: output, range: 0..<outBytes, value: 0)
        blitEncoder.endEncoding()

        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .buffer(aBuf), .buffer(idxBuf), .buffer(output),
          .uint(UInt32(s.outerCount)), .uint(UInt32(s.outCount)),
          .uint(UInt32(s.middleCount)), .uint(UInt32(s.innerCount)),
          .opaque(indexStrides))
        dispatch1D(computeEncoder, state: state, threadCount: s.gatherOutCount)
        computeEncoder.endEncoding()
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  private func createScatter(scatter: ScatterGatherIndices, dtype: MPSDataType) -> TwoToOneGraph {
    let key = ScatterKey(
      valueShape: scatter.valueShape,
      axis: scatter.axis,
      indexShape: scatter.indices.strides.shape,
      indexStrides: scatter.indices.strides.strides,
      dtype: dtype
    )
    if let r = scatters[key] {
      return r
    }

    let graph = MPSGraph()
    let inputShape = key.indexShape
    var indexShape = scatter.indices.strides.dataShape
    if indexShape[..<scatter.axis].allSatisfy({ $0 == 1 })
      && indexShape[(scatter.axis + 1)...].allSatisfy({ $0 == 1 })
    {
      indexShape = [indexShape[scatter.axis]]
    }
    let shape3DIn = [
      inputShape[..<scatter.axis].product(), inputShape[scatter.axis],
      inputShape[(scatter.axis + 1)...].product(),
    ]
    let shape3DOut = [
      key.valueShape[..<scatter.axis].product(), key.valueShape[scatter.axis],
      key.valueShape[(scatter.axis + 1)...].product(),
    ]

    assert(inputShape.product() == scatter.gatherOutCount)
    assert(shape3DOut.product() == scatter.gatherInCount)
    assert(indexShape.product() == scatter.indices.dataCount)

    let input = graph.placeholder(
      shape: mpsShape([inputShape.product()]), dataType: dtype, name: "input")
    let indices = graph.placeholder(
      shape: mpsShape([indexShape.product()]), dataType: .int64, name: "indices")
    let inputReshaped = graph.reshape(input, shape: mpsShape(shape3DIn), name: "inputReshaped")
    let indicesReshaped = graph.reshape(
      indices, shape: mpsShape(indexShape), name: "indexReshaped")
    let unshapedOutput =
      if indexShape.count == 1 {
        graph.scatter(
          inputReshaped,
          indices: indicesReshaped,
          shape: mpsShape(shape3DOut),
          axis: 1,
          mode: .add,
          name: "scatter"
        )
      } else {
        graph.scatterAlongAxis(
          1,
          updates: inputReshaped,
          indices: graph.reshape(
            (inputShape != indexShape
              ? graph.broadcast(indicesReshaped, shape: mpsShape(inputShape), name: "bcast")
              : indicesReshaped),
            shape: mpsShape(shape3DIn),
            name: "bcastReshaped"
          ),
          shape: mpsShape(shape3DOut),
          mode: .add,
          name: "scatter"
        )
      }
    let output = graph.reshape(
      unshapedOutput,
      shape: mpsShape([shape3DOut.product()]),
      name: "output"
    )
    let r = TwoToOneGraph(graph: graph, inputA: input, inputB: indices, output: output)
    scatters[key] = r
    return r
  }

  override open func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data {
    let outputCount = shape.product()
    #alwaysAssert(outputCount <= Int(UInt32.max), "cannot apply kernel to this many values")

    let (buffer, bufferCb) = try await allocateBuf(outputCount * Tensor.DType.int64.byteSize)
    return try await serialize { [self] in
      let oldStrides = stridesForShape(shape)
      let permutedStrides = permutation.map { oldStrides[$0] }
      let newShape = permutation.map { shape[$0] }
      let newStrides = stridesForShape(newShape)

      let completion = try completionBufferAndEncoder(
        label: "axisPermutation", deallocators: []
      ) { buf, enc in
        let functionName = "axis_permutation"
        let state = try getFunction(name: functionName)
        try setArguments(
          enc,
          .uints(newStrides.map { UInt32($0) }),
          .uints(newShape.map { UInt32($0) }),
          .uints(permutedStrides.map { UInt32($0) }),
          .buffer(buffer),
          .uint(UInt32(newShape.count)),
          .uint(UInt32(outputCount))
        )
        dispatch1D(enc, state: state, threadCount: outputCount)
      }
      return GPUData(backend: self, buffer: buffer, completion: completion, deallocator: bufferCb)
    }
  }

  override open func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.reduce(a, op: op, dims: dims, dtype: dtype)
    }

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (output, outputCb) = try await allocateBuf(
      dims.outCount * (op == .sum ? dtype : Tensor.DType.int64).byteSize)
    return try await serialize { [self] in
      let red = self.createReduction(op: op, dims: dims, dtype: mpsDType)
      let completion = completionBuffer(
        label: "reduce", deallocators: [aCb]
      ) { buf in
        red.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            red.input: MPSGraphTensorData(
              MPSVector(
                buffer: aBuf,
                descriptor: MPSVectorDescriptor(
                  length: dims.inCount, dataType: mpsDType)))
          ],
          targetOperations: nil,
          resultsDictionary: [
            red.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(
                  length: dims.outCount, dataType: op == .sum ? mpsDType : .int64)))
          ],
          executionDescriptor: nil)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  private func createReduction(op: ReduceOp, dims: ReduceDims, dtype: MPSDataType) -> OneToOneGraph
  {
    let key = ReduceKey(op: op, dims: dims, dtype: dtype)
    if let r = reductions[key] {
      return r
    } else {
      let graph = MPSGraph()
      let input = graph.placeholder(
        shape: mpsShape([dims.inCount]), dataType: dtype, name: "input")
      let reshaped = graph.reshape(input, shape: mpsShape(dims.shape), name: "reshaped")
      let unshapedOutput =
        switch op {
        case .sum:
          graph.reductionSum(with: reshaped, axis: 1, name: "unshapedOutput")
        case .argmax:
          graph.cast(
            graph.reductionArgMaximum(with: reshaped, axis: 1, name: "unshapedOutput"),
            to: .int64, name: "castedOutput")
        case .argmin:
          graph.cast(
            graph.reductionArgMinimum(with: reshaped, axis: 1, name: "unshapedOutput"),
            to: .int64, name: "castedOutput")
        }
      let output = graph.reshape(unshapedOutput, shape: mpsShape([dims.outCount]), name: "output")
      let r = OneToOneGraph(graph: graph, input: input, output: output)
      reductions[key] = r
      return r
    }
  }

  override open func logSoftmax(
    _ a: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.logSoftmax(a, dims: dims, dtype: dtype)
    }

    let totalCount = dims.outerCount * dims.reduceCount * dims.innerCount
    let (aBuf, aCb) = try await gpuBuffer(a)
    let (output, outputCb) = try await allocateBuf(totalCount * dtype.byteSize)

    if dims.innerCount == 1 && (dtype == .float16 || dtype == .float32) {
      return try await serialize { [self] in
        let completion = try completionBufferAndEncoder(
          label: "logSoftmax", deallocators: [aCb]
        ) { buf, enc in
          try setArguments(enc, .buffer(aBuf), .buffer(output), .uint(UInt32(dims.reduceCount)))

          let functionName = "log_softmax_\(MPSBackend.CastTypes[dtype]!)"
          let state = try getFunction(name: functionName)
          let gridSize = MTLSize(width: dims.outerCount * 256, height: 1, depth: 1)
          let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
          enc.setComputePipelineState(state)
          enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        }
        return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
      }
    }

    return try await serialize { [self] in
      let op = self.createLogSoftmax(
        outerCount: dims.outerCount, middleCount: dims.reduceCount, innerCount: dims.innerCount,
        dtype: mpsDType)
      let completion = completionBuffer(
        label: "logSoftmax", deallocators: [aCb]
      ) { buf in
        op.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            op.input: MPSGraphTensorData(
              MPSVector(
                buffer: aBuf,
                descriptor: MPSVectorDescriptor(length: totalCount, dataType: mpsDType)))
          ],
          targetOperations: nil,
          resultsDictionary: [
            op.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(length: totalCount, dataType: mpsDType)
              ))
          ],
          executionDescriptor: nil)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  private func createLogSoftmax(
    outerCount: Int, middleCount: Int, innerCount: Int, dtype: MPSDataType
  ) -> OneToOneGraph {
    let key = SoftmaxKey(
      outerCount: outerCount, middleCount: middleCount, innerCount: innerCount, dtype: dtype)
    if let r = logSoftmaxes[key] {
      return r
    } else {
      let graph = MPSGraph()
      let totalCount = outerCount * middleCount * innerCount
      let input = graph.placeholder(shape: mpsShape([totalCount]), dataType: dtype, name: "input")
      let reshaped = graph.reshape(
        input, shape: mpsShape([outerCount, middleCount, innerCount]), name: "reshaped")
      let maxes = graph.reshape(
        graph.reductionMaximum(with: reshaped, axes: mpsShape([1]), name: "reduceMax"),
        shape: mpsShape([outerCount, 1, innerCount]), name: "reshapeMax")
      let exps = graph.exponent(
        with: graph.subtraction(reshaped, maxes, name: "subMax"), name: "exp")
      let sumExp = graph.reshape(
        graph.reductionSum(with: exps, axes: mpsShape([1]), name: "sumExp"),
        shape: mpsShape([outerCount, 1, innerCount]), name: "reshapedSumExp")
      let logSumExp = graph.addition(
        graph.logarithm(with: sumExp, name: "logSumExp"), maxes, name: "logSumExpWithMaxes")
      let results = graph.subtraction(reshaped, logSumExp, name: "unshapedResult")
      let output = graph.reshape(results, shape: mpsShape([totalCount]), name: "flatOut")
      let r = OneToOneGraph(graph: graph, input: input, output: output)
      logSoftmaxes[key] = r
      return r
    }
  }

  override open func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.logSoftmaxGrad(a, outGrad, dims: dims, dtype: dtype)
    }

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (outGradBuf, outGradCb) = try await gpuBuffer(outGrad)

    let totalCount = dims.outerCount * dims.reduceCount * dims.innerCount
    let (output, outputCb) = try await allocateBuf(totalCount * dtype.byteSize)

    if dims.innerCount == 1 && (dtype == .float16 || dtype == .float32) {
      return try await serialize { [self] in
        let completion = try completionBufferAndEncoder(
          label: "logSoftmaxGrad", deallocators: [aCb, outGradCb]
        ) { buf, enc in
          try setArguments(
            enc,
            .buffer(aBuf),
            .buffer(outGradBuf),
            .buffer(output),
            .uint(UInt32(dims.reduceCount))
          )

          let functionName = "log_softmax_grad_\(MPSBackend.CastTypes[dtype]!)"
          let state = try getFunction(name: functionName)
          let gridSize = MTLSize(width: dims.outerCount * 256, height: 1, depth: 1)
          let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
          enc.setComputePipelineState(state)
          enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        }
        return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
      }
    }

    return try await serialize { [self] in
      let op = self.createLogSoftmaxGrad(
        outerCount: dims.outerCount, middleCount: dims.reduceCount, innerCount: dims.innerCount,
        dtype: mpsDType)
      let completion = completionBuffer(
        label: "logSoftmaxGrad", deallocators: [aCb, outGradCb]
      ) { buf in
        op.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            op.inputA: MPSGraphTensorData(
              MPSVector(
                buffer: aBuf,
                descriptor: MPSVectorDescriptor(length: totalCount, dataType: mpsDType))),
            op.inputB: MPSGraphTensorData(
              MPSVector(
                buffer: outGradBuf,
                descriptor: MPSVectorDescriptor(length: totalCount, dataType: mpsDType))),
          ],
          targetOperations: nil,
          resultsDictionary: [
            op.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(length: totalCount, dataType: mpsDType)
              ))
          ],
          executionDescriptor: nil)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  private func createLogSoftmaxGrad(
    outerCount: Int, middleCount: Int, innerCount: Int, dtype: MPSDataType
  ) -> TwoToOneGraph {
    let key = SoftmaxKey(
      outerCount: outerCount, middleCount: middleCount, innerCount: innerCount, dtype: dtype)
    if let r = logSoftmaxGrads[key] {
      return r
    } else {
      let graph = MPSGraph()
      let totalCount = outerCount * middleCount * innerCount
      let input = graph.placeholder(shape: mpsShape([totalCount]), dataType: dtype, name: "input")
      let gradInput = graph.placeholder(
        shape: mpsShape([totalCount]), dataType: dtype, name: "gradInput")
      let reshaped = graph.reshape(
        input, shape: mpsShape([outerCount, middleCount, innerCount]), name: "reshaped")
      let reshapedGrad = graph.reshape(
        gradInput, shape: mpsShape([outerCount, middleCount, innerCount]), name: "reshapedGrad")
      let softmaxOutput = graph.softMax(with: reshaped, axis: 1, name: "softmax")
      let gradSum = graph.reshape(
        graph.reductionSum(with: reshapedGrad, axes: mpsShape([1]), name: "sumGrad"),
        shape: mpsShape([outerCount, 1, innerCount]), name: "sumGradReshaped")
      let results = graph.subtraction(
        reshapedGrad, graph.multiplication(gradSum, softmaxOutput, name: "gradSumProduct"),
        name: "unshapedResult")
      let output = graph.reshape(results, shape: mpsShape([totalCount]), name: "flatOut")
      let r = TwoToOneGraph(graph: graph, inputA: input, inputB: gradInput, output: output)
      logSoftmaxGrads[key] = r
      return r
    }
  }

  override open func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await batchedMatmul(
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
    guard let mpsDType = dtype.mpsDType else {
      return try await super.batchedMatmul(
        matrixCount: matrixCount, a: a, transA: transA, b: b, transB: transB, transOut: transOut,
        rows: rows, inner: inner,
        cols: cols, dtype: dtype)
    }

    if transOut {
      return try await batchedMatmul(
        matrixCount: matrixCount, a: b, transA: !transB, b: a, transB: !transA, transOut: false,
        rows: cols, inner: inner,
        cols: rows, dtype: dtype)
    }

    let aShape = transA ? (inner, rows) : (rows, inner)
    let bShape = transB ? (cols, inner) : (inner, cols)

    let (aBuf, aCb) = try await gpuBuffer(a)
    let (bBuf, bCb) = try await gpuBuffer(b)
    let (output, outputCb) = try await allocateBuf(matrixCount * rows * cols * dtype.byteSize)

    return try await serialize { [self] in
      let mm = self.createMatmul(
        transA: transA, transB: transB, batch: matrixCount, rows: rows, inner: inner, cols: cols,
        dtype: mpsDType)
      let completion = completionBuffer(
        label: "batchedMatmul", deallocators: [aCb, bCb]
      ) { buf in
        #alwaysAssert(
          aBuf.allocatedSize >= matrixCount * aShape.0 * aShape.1 * dtype.byteSize,
          "matrix A buffer underflow")
        #alwaysAssert(
          bBuf.allocatedSize >= matrixCount * bShape.0 * bShape.1 * dtype.byteSize,
          "matrix B buffer underflow \(matrixCount) * \(bShape) * \(dtype.byteSize) vs \(bBuf.allocatedSize)"
        )
        #alwaysAssert(
          output.allocatedSize >= matrixCount * rows * cols * dtype.byteSize,
          "output matrix buffer underflow")
        mm.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            mm.inputA: MPSGraphTensorData(
              MPSMatrix(
                buffer: aBuf,
                descriptor: MPSMatrixDescriptor(
                  rows: aShape.0, columns: aShape.1, matrices: matrixCount,
                  rowBytes: aShape.1 * dtype.byteSize,
                  matrixBytes: aShape.0 * aShape.1 * dtype.byteSize, dataType: mpsDType)),
              rank: matrixCount > 1 ? 3 : 2),
            mm.inputB: MPSGraphTensorData(
              MPSMatrix(
                buffer: bBuf,
                descriptor: MPSMatrixDescriptor(
                  rows: bShape.0, columns: bShape.1, matrices: matrixCount,
                  rowBytes: bShape.1 * dtype.byteSize,
                  matrixBytes: bShape.0 * bShape.1 * dtype.byteSize, dataType: mpsDType)),
              rank: matrixCount > 1 ? 3 : 2),
          ],
          targetOperations: nil,
          resultsDictionary: [
            mm.output: MPSGraphTensorData(
              MPSMatrix(
                buffer: output,
                descriptor: MPSMatrixDescriptor(
                  rows: rows, columns: cols, matrices: matrixCount, rowBytes: cols * dtype.byteSize,
                  matrixBytes: rows * cols * dtype.byteSize, dataType: mpsDType)),
              rank: matrixCount > 1 ? 3 : 2)
          ],
          executionDescriptor: nil)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  private func createMatmul(
    transA: Bool, transB: Bool, batch: Int, rows: Int, inner: Int, cols: Int, dtype: MPSDataType
  )
    -> TwoToOneGraph
  {
    let key = MatmulKey(
      transA: transA, transB: transB, batch: batch, rows: rows, cols: cols, inner: inner,
      dtype: dtype)
    if let matmul = matmuls[key] {
      return matmul
    } else {
      let graph = MPSGraph()
      let batchShape = batch > 1 ? [batch] : []
      let inputA = graph.placeholder(
        shape: mpsShape(batchShape + [transA ? inner : rows, transA ? rows : inner]),
        dataType: dtype, name: "inputA")
      let inputB = graph.placeholder(
        shape: mpsShape(batchShape + [transB ? cols : inner, transB ? inner : cols]),
        dataType: dtype, name: "inputB")
      let transIndices = batch > 1 ? (1, 2) : (0, 1)
      let output = graph.matrixMultiplication(
        primary: !transA
          ? inputA
          : graph.transposeTensor(
            inputA, dimension: transIndices.0, withDimension: transIndices.1, name: nil),
        secondary: !transB
          ? inputB
          : graph.transposeTensor(
            inputB, dimension: transIndices.0, withDimension: transIndices.1, name: nil),
        name: "output")
      let mm = TwoToOneGraph(graph: graph, inputA: inputA, inputB: inputB, output: output)
      matmuls[key] = mm
      return mm
    }
  }

  internal func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data,
    dtype: Tensor.DType, transpose: Bool
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      if transpose {
        return try await super.conv2DTranspose(
          config, batch: batch, image: image, kernel: kernel, dtype: dtype)
      } else {
        return try await super.conv2D(
          config, batch: batch, image: image, kernel: kernel, dtype: dtype)
      }
    }

    let imageShape =
      transpose ? config.outputTensorShape(batch: batch) : config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape =
      transpose ? config.imageTensorShape(batch: batch) : config.outputTensorShape(batch: batch)
    let (imageBuf, imageCb) = try await gpuBuffer(image)
    let (kernelBuf, kernelCb) = try await gpuBuffer(kernel)
    let (output, outputCb) = try await allocateBuf(outShape.product() * dtype.byteSize)

    return try await serialize { [self] in
      let op = try self.createConv2D(
        config, batch: batch, kind: transpose ? .transpose : .forward, dtype: mpsDType)
      let completion = completionBuffer(
        label: "conv2D", deallocators: [imageCb, kernelCb]
      ) { buf in
        #alwaysAssert(
          imageBuf.allocatedSize >= imageShape.product() * dtype.byteSize,
          "input image buffer underflow")
        #alwaysAssert(
          kernelBuf.allocatedSize >= kernelShape.product() * dtype.byteSize,
          "kernel buffer underflow")
        op.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            op.inputA: MPSGraphTensorData(
              MPSVector(
                buffer: imageBuf,
                descriptor: MPSVectorDescriptor(length: imageShape.product(), dataType: mpsDType))),
            op.inputB: MPSGraphTensorData(
              MPSVector(
                buffer: kernelBuf,
                descriptor: MPSVectorDescriptor(length: kernelShape.product(), dataType: mpsDType))),
          ], targetOperations: nil,
          resultsDictionary: [
            op.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(length: outShape.product(), dataType: mpsDType)))
          ], executionDescriptor: nil)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  internal func conv1DToConv2D(_ config: Conv1DConfig) throws -> Conv2DConfig {
    try Conv2DConfig(
      inChannels: config.inChannels, outChannels: config.outChannels,
      kernelSize: .init(x: config.kernelSize.x, y: 1),
      imageSize: .init(x: config.imageSize.x, y: 1), stride: .init(x: config.stride.x, y: 1),
      dilation: .init(x: config.dilation.x, y: 1),
      padding: .init(
        before: .init(x: config.padding.before.x, y: 0),
        after: .init(x: config.padding.after.x, y: 0)), groups: config.groups,
      channelsLast: config.channelsLast)
  }

  override open func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2D(
      try conv1DToConv2D(config), batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override open func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2DTranspose(
      try conv1DToConv2D(config), batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override open func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2DKernelGrad(
      try conv1DToConv2D(config), batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override open func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2D(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype, transpose: false)
  }

  override open func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2D(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype, transpose: true)
  }

  override open func conv2DKernelGrad(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.conv2DKernelGrad(
        config, batch: batch, image: image, outGrad: outGrad, dtype: dtype)
    }

    let imageShape = config.imageTensorShape(batch: batch)
    let kernelShape = config.kernelTensorShape()
    let outShape = config.outputTensorShape(batch: batch)
    let (imageBuf, imageCb) = try await gpuBuffer(image)
    let (outGradBuf, outGradCb) = try await gpuBuffer(outGrad)
    let (output, outputCb) = try await allocateBuf(kernelShape.product() * dtype.byteSize)
    return try await serialize { [self] in
      let op = try self.createConv2D(config, batch: batch, kind: .kernelGrad, dtype: mpsDType)
      let completion = completionBuffer(
        label: "conv2DKernelGrad", deallocators: [imageCb, outGradCb]
      ) { buf in
        #alwaysAssert(
          imageBuf.allocatedSize >= imageShape.product() * dtype.byteSize,
          "input image buffer underflow")
        #alwaysAssert(
          outGradBuf.allocatedSize >= outShape.product() * dtype.byteSize,
          "output gradient buffer underflow")
        op.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            op.inputA: MPSGraphTensorData(
              MPSVector(
                buffer: imageBuf,
                descriptor: MPSVectorDescriptor(length: imageShape.product(), dataType: mpsDType))),
            op.inputB: MPSGraphTensorData(
              MPSVector(
                buffer: outGradBuf,
                descriptor: MPSVectorDescriptor(length: outShape.product(), dataType: mpsDType))),
          ], targetOperations: nil,
          resultsDictionary: [
            op.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(length: kernelShape.product(), dataType: mpsDType)))
          ], executionDescriptor: nil)
      }
      return GPUData(backend: self, buffer: output, completion: completion, deallocator: outputCb)
    }
  }

  private func createConv2D(
    _ conv: Conv2DConfig, batch: Int, kind: Conv2DKey.Kind, dtype: MPSDataType
  ) throws -> TwoToOneGraph {
    let key = Conv2DKey(conv: conv, batch: batch, kind: kind, dtype: dtype)
    if let op = conv2D[key] {
      return op
    }

    let graph = MPSGraph()
    guard
      let convDesc = MPSGraphConvolution2DOpDescriptor(
        strideInX: conv.stride.x, strideInY: conv.stride.y, dilationRateInX: conv.dilation.x,
        dilationRateInY: conv.dilation.y, groups: conv.groups, paddingLeft: conv.padding.before.x,
        paddingRight: conv.padding.after.x, paddingTop: conv.padding.before.y,
        paddingBottom: conv.padding.after.y, paddingStyle: .explicit,
        dataLayout: conv.channelsLast ? .NHWC : .NCHW, weightsLayout: .OIHW)
    else {
      throw BackendError.kernelFailed("failed to create MPSGraph convolution for: \(conv)")
    }

    let kernelShape = conv.kernelTensorShape()

    if kind == .kernelGrad {
      let imageShape = conv.imageTensorShape(batch: batch)
      let outShape = conv.outputTensorShape(batch: batch)
      let inputA = graph.placeholder(
        shape: mpsShape([imageShape.product()]), dataType: dtype, name: "image")
      let inputB = graph.placeholder(
        shape: mpsShape([outShape.product()]), dataType: dtype, name: "outGrad")
      let image = graph.reshape(inputA, shape: mpsShape(imageShape), name: "imageWithShape")
      let outGrad = graph.reshape(inputB, shape: mpsShape(outShape), name: "outGradWithShape")
      let output =
        graph.convolution2DWeightsGradient(
          outGrad, source: image, outputShape: mpsShape(kernelShape),
          forwardConvolutionDescriptor: convDesc, name: "output")
      let flatOutput = graph.reshape(
        output, shape: mpsShape([kernelShape.product()]), name: "flatOutput")
      let op = TwoToOneGraph(graph: graph, inputA: inputA, inputB: inputB, output: flatOutput)
      conv2D[key] = op
      return op
    } else {
      let imageShape =
        kind == .transpose
        ? conv.outputTensorShape(batch: batch) : conv.imageTensorShape(batch: batch)
      let outShape =
        kind == .transpose
        ? conv.imageTensorShape(batch: batch) : conv.outputTensorShape(batch: batch)
      let inputA = graph.placeholder(
        shape: mpsShape([imageShape.product()]), dataType: dtype, name: "image")
      let inputB = graph.placeholder(
        shape: mpsShape([kernelShape.product()]), dataType: dtype, name: "kernel")
      let image = graph.reshape(inputA, shape: mpsShape(imageShape), name: "imageWithShape")
      let kernel = graph.reshape(inputB, shape: mpsShape(kernelShape), name: "kernelWithShape")
      let output =
        if kind == .transpose {
          graph.convolutionTranspose2D(
            image, weights: kernel, outputShape: mpsShape(outShape), descriptor: convDesc,
            name: "output")
        } else {
          graph.convolution2D(image, weights: kernel, descriptor: convDesc, name: "output")
        }
      let flatOutput = graph.reshape(
        output, shape: mpsShape([outShape.product()]), name: "flatOutput")
      let op = TwoToOneGraph(graph: graph, inputA: inputA, inputB: inputB, output: flatOutput)
      conv2D[key] = op
      return op
    }
  }

  override open func defaultRandom() -> RandomGenerator {
    _defaultRandomMPS!
  }

  override open func createRandom() -> RandomGenerator {
    MPSRandomGenerator(mpsBackend: self, seed: Int.random(in: 0..<1_000_000_000))
  }

  internal enum KernelArgument {
    case uint(UInt32)
    case uints([UInt32])
    case float(Float)
    case int64(Int64)
    case uint64(UInt64)
    case buffer(any MTLBuffer)
    case data(Data)

    internal func intoBuffer(_ b: MPSBackend) throws -> MTLBuffer {
      switch self {
      case .uint(let x):
        try b.makeUIntBuffer(x)
      case .uints(let x):
        try b.makeUIntBuffer(x)
      case .float(let x):
        try b.makeFloatBuffer(x)
      case .int64(let x):
        try b.makeInt64Buffer(x)
      case .uint64(let x):
        try b.makeUInt64Buffer(x)
      case .buffer(let x):
        x
      case .data(let x):
        try b.makeDataBuffer(x)
      }
    }

    static func opaque<T>(_ x: T) -> KernelArgument {
      var x = x
      return .data(withUnsafeBytes(of: &x) { Data($0) })
    }
  }

  internal func setArguments(_ c: MTLComputeCommandEncoder, _ args: KernelArgument...) throws {
    for (i, arg) in args.enumerated() {
      c.setBuffer(try arg.intoBuffer(self), offset: 0, index: i)
    }
  }

  internal func dispatch1D(
    _ c: MTLComputeCommandEncoder, state: MTLComputePipelineState, threadCount count: Int
  ) {
    let groupSize = min(state.maxTotalThreadsPerThreadgroup, nextPowerOf2(count, min: 32))
    let gridSize = MTLSize(width: nextMultiple(count, divisor: groupSize), height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: groupSize, height: 1, depth: 1)
    c.setComputePipelineState(state)
    c.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
  }

  internal func nextPowerOf2(_ x: Int, min: Int = 1) -> Int {
    var y = min
    while y < x {
      y *= 2
    }
    return y
  }

  internal func nextMultiple(_ x: Int, divisor: Int) -> Int {
    if x % divisor == 0 {
      x
    } else {
      x + divisor - (x % divisor)
    }
  }

  internal func makeUIntBuffer(_ x: UInt32) throws -> MTLBuffer {
    try makeUIntBuffer([x])
  }

  internal func makeUIntBuffer(_ x: [UInt32]) throws -> MTLBuffer {
    return try x.withUnsafeBytes { bytes in
      let result = try allocateArgument(4 * x.count)
      result.contents().copyMemory(from: bytes.baseAddress!, byteCount: 4 * x.count)
      return result
    }
  }

  internal func makeFloatBuffer(_ x: Float) throws -> MTLBuffer {
    var x = x
    let result = try allocateArgument(4)
    result.contents().copyMemory(from: &x, byteCount: 4)
    return result
  }

  internal func makeInt64Buffer(_ x: Int64) throws -> MTLBuffer {
    var x = x
    let result = try allocateArgument(8)
    result.contents().copyMemory(from: &x, byteCount: 8)
    return result
  }

  internal func makeUInt64Buffer(_ x: UInt64) throws -> MTLBuffer {
    var x = x
    let result = try allocateArgument(8)
    result.contents().copyMemory(from: &x, byteCount: 8)
    return result
  }

  internal func makeDataBuffer(_ x: Data) throws -> MTLBuffer {
    let result = try allocateArgument(x.count)
    x.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
      result.contents().copyMemory(from: bytes.baseAddress!, byteCount: x.count)
    }
    return result
  }

  class CompletionCallback {
    private var lock: NSLock = NSLock()
    private var _result: Result<(), Error>? = nil
    private var _continue: (@Sendable (Result<(), Error>) -> Void)? = nil

    init() {
    }

    func putResult(_ x: Result<(), Error>) {
      lock.lock()
      if let c = _continue {
        lock.unlock()
        c(x)
      } else {
        _result = x
        lock.unlock()
      }
    }

    func putContinuation(_ f: @escaping @Sendable (Result<(), Error>) -> Void) {
      lock.lock()
      if let r = _result {
        lock.unlock()
        f(r)
      } else {
        _continue = f
        lock.unlock()
      }
    }
  }

  internal func completionBuffer(
    label: String, deallocators: [Deallocator], _ action: (MTLCommandBuffer) throws -> Void
  ) rethrows -> Task<(), Error> {
    let rawBuf = commandQueue!.makeCommandBuffer()!
    rawBuf.label = label
    let buf = MPSCommandBuffer(commandBuffer: rawBuf)
    try action(buf)
    let cb = CompletionCallback()
    buf.addCompletedHandler { _ in
      let result: Result<(), Error> =
        if let e = buf.error {
          Result.failure(e)
        } else {
          Result.success(())
        }
      for x in deallocators {
        x()
      }
      cb.putResult(result)
    }
    buf.commit()
    return Task.detached {
      try await withCheckedThrowingContinuation { continuation in
        cb.putContinuation({ x in continuation.resume(with: x) })
      }
    }
  }

  internal func completionBufferAndEncoder(
    label: String, deallocators: [Deallocator],
    _ action: (MTLCommandBuffer, MTLComputeCommandEncoder) throws -> Void
  ) throws -> Task<
    (), Error
  > {
    try completionBuffer(label: label, deallocators: deallocators) { buf in
      guard let computeEncoder = buf.makeComputeCommandEncoder() else {
        throw BackendError.kernelFailed("could not create compute encoder")
      }
      do {
        try action(buf, computeEncoder)
        computeEncoder.endEncoding()
      } catch {
        computeEncoder.endEncoding()
        throw error
      }
    }
  }

}

extension Tensor.DType {
  var mpsDType: MPSDataType? {
    switch self {
    case .float32:
      .float32
    case .float16:
      .float16
    case .int64:
      .int64
    default:
      nil
    }
  }
}

func mpsShape(_ x: [Int]) -> [NSNumber] {
  x.map { NSNumber(value: $0) }
}

extension Tensor.DType {
  fileprivate var metalSizeType: String {
    switch self {
    case .bool:
      "char"
    case .float16:
      "short"
    case .float32:
      "int"
    case .int64:
      "long"
    }
  }
}

func nextAllocatorBucket(_ length: Int) -> Int {
  if length < 4096 {
    return 4096
  } else {
    var i = 4096
    while i < length && i >= 4096 {
      i *= 2
    }
    #alwaysAssert(i >= 4096, "allocation size overflow: \(length)")
    return i
  }
}
