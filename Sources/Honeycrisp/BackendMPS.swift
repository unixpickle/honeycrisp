import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

open class MPSBackend: CPUBackend {

  internal struct GPUData: Tensor.Data {
    public let backend: MPSBackend
    public let buffer: MTLBuffer
    public let completion: Task<Void, Error>?

    public var cpuBuffer: MTLBuffer {
      get async throws {
        if let completion = completion {
          let _ = try await completion.value
        }
        return buffer
      }
    }
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
    let broadcasted: Bool
    let outCount: Int

    let outerCount: Int
    let middleCount: Int
    let innerCount: Int

    let dtype: MPSDataType
  }

  public class MPSRandomGenerator: NativeRandomGenerator {
    public let mpsBackend: MPSBackend
    private var seed: UInt64
    private var offset: UInt64

    init(mpsBackend: MPSBackend, seed: Int) {
      self.mpsBackend = mpsBackend
      self.seed = UInt64(seed)
      self.offset = 0
      super.init(cpuBackend: mpsBackend)
    }

    override public func save() async throws -> Data {
      let info = try await mpsBackend.serialize {
        return [self.seed, self.offset]
      }
      return Data(info.flatMap { x in (0..<8).map { UInt8((x >> (8 * $0)) & 0xff) } })
    }

    override public func restore(_ x: Data) async throws {
      var seed: UInt64 = 0
      var offset: UInt64 = 0
      for (i, x) in x.enumerated() {
        if i < 8 {
          seed |= UInt64(x) << (8 * i)
        } else {
          offset |= UInt64(x) << (8 * (i - 8))
        }
      }
      try await mpsBackend.serialize {
        self.seed = seed
        self.offset = offset
      }
    }

    override public func seed(_ seed: Int) async throws {
      try await mpsBackend.serialize {
        self.seed = UInt64(seed)
        self.offset = 0
      }
    }

    override public func sample(count: Int, dist: RandomDist, dtype: Tensor.DType) async throws
      -> Tensor.Data
    {
      if dtype != .float16 && dtype != .float32 {
        return try await super.sample(count: count, dist: dist, dtype: dtype)
      }
      alwaysAssert(count <= 0xffff_ffff, "count exceeds UInt32 size: \(count)")

      let functionName =
        "\(dist == .normal ? "randn" : "rand")_\(MPSBackend.CastTypes[dtype]!)"
      let output = try await mpsBackend.allocate(length: count * dtype.byteSize)

      return try await mpsBackend.serialize { [self] in
        let completion = try mpsBackend.completionBufferAndEncoder(label: "sample") { buf, enc in
          let state = try mpsBackend.getFunction(name: functionName)
          try mpsBackend.setArguments(
            enc, .buffer(output), .uint(UInt32(seed)), .uint(UInt32(seed >> 32)),
            .uint(UInt32(offset)), .uint(UInt32(count)))
          let chunkSize = dist == .normal ? 2 : 4
          let totalThreads = (count + chunkSize - 1) / chunkSize
          mpsBackend.dispatch1D(enc, state: state, threadCount: totalThreads)
          offset += UInt64(totalThreads)
        }
        return GPUData(backend: mpsBackend, buffer: output, completion: completion)
      }
    }
  }

  private static let CastTypes: [Tensor.DType: String] = [
    .float16: "half", .float32: "float", .int64: "long",
  ]

  private var queue = DispatchQueue(label: "mps-backend-worker")
  private var commandQueue: MTLCommandQueue? = nil
  private var matmuls: [MatmulKey: TwoToOneGraph] = [:]
  private var conv2D: [Conv2DKey: TwoToOneGraph] = [:]
  private var scatters: [ScatterKey: TwoToOneGraph] = [:]
  private var reductions: [ReduceKey: OneToOneGraph] = [:]
  private var logSoftmaxes: [SoftmaxKey: OneToOneGraph] = [:]
  private var logSoftmaxGrads: [SoftmaxKey: TwoToOneGraph] = [:]
  private var defaultRNG: MPSRandomGenerator? = nil
  private var functions: [String: MTLComputePipelineState] = [:]

  public init(
    device: MTLDevice? = nil, commandQueue: MTLCommandQueue? = nil, allocator: Allocator = .device
  ) throws {
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
    try self.initAllocator(allocator)

    guard let fileURL = Bundle.module.url(forResource: "kernels", withExtension: "metal") else {
      throw BackendError.failedToLoadKernels("kernels.metal could not be found")
    }
    guard let sourceStr = String(data: try Data(contentsOf: fileURL), encoding: .utf8) else {
      throw BackendError.failedToLoadKernels("kernels source code was not a utf-8 string")
    }
    let library = try (self._device!).makeLibrary(source: sourceStr, options: MTLCompileOptions())

    // Lookup all functions in the library.
    var names = ["axis_permutation"]
    for type in ["half", "float"] {
      for op in [
        "vector_pow", "log", "recip", "exp", "sigmoid", "sigmoid_grad", "gelu", "gelu_grad", "sin",
        "cos", "minus_sin", "relu", "relu_grad", "abs", "abs_grad", "rand", "randn", "normalize",
        "normalize_var_grad", "normalize_x_grad",
      ] {
        names.append("\(op)_\(type)")
      }
    }
    for type in ["char", "short", "int", "long"] {
      names.append("repeat_\(type)")
      names.append("strided_copy_\(type)")
      for mode in ["", "_bcast"] {
        for op in ["gather", "scatter"] {
          names.append("\(op)\(mode)_\(type)")
        }
      }
    }
    for op in ["add", "sub", "mul", "div", "mod"] {
      for type in ["half", "float"] {
        for args in ["vv", "sv", "vs"] {
          names.append("\(op)\(args)_\(type)")
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
      functions[name] = try (self._device!).makeComputePipelineState(function: f)
    }
  }

  internal func gpuBuffer(_ data: Tensor.Data) async throws -> MTLBuffer {
    if let data = data as? GPUData, data.backend === self {
      return data.buffer
    }
    return try await data.cpuBuffer
  }

  internal func getFunction(name: String) throws -> MTLComputePipelineState {
    guard let f = functions[name] else {
      throw BackendError.kernelFailed("no kernel with name '\(name)'")
    }
    return f
  }

  override public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, outScale: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.pow(a, b, outScale: outScale, count: count, dtype: dtype)
    }

    alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")
    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "pow") { buf, enc in
        let funcName = "vector_pow_\(MPSBackend.CastTypes[dtype]!)"
        let state = try getFunction(name: funcName)
        try setArguments(
          enc, .buffer(aBuf), .buffer(output), .float(b.toFloat()), .float(outScale.toFloat()),
          .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func elemwise(_ a: Tensor.Data, op: ElemwiseOp, count: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.elemwise(a, op: op, count: count, dtype: dtype)
    }

    alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

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
      case .gelu:
        "gelu"
      case .geluGrad:
        "gelu_grad"
      case .exp:
        "exp"
      case .sigmoid:
        "sigmoid"
      case .sigmoidGrad:
        "sigmoid_grad"
      }
    let functionName = "\(namePrefix)_\(MPSBackend.CastTypes[dtype]!)"

    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "elemwise") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(enc, .buffer(aBuf), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func binaryOp(
    a: BroadcastData, b: BroadcastData, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.binaryOp(a: a, b: b, op: op, count: count, dtype: dtype)
    }

    alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName =
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
    let functionName = "\(opName)vv_\(MPSBackend.CastTypes[dtype]!)"

    let aBuf = try await gpuBuffer(a.data)
    let bBuf = try await gpuBuffer(b.data)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "binaryOp") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(aBuf), .buffer(bBuf), .buffer(output),
          .opaque(
            (
              UInt32(a.strides.dataCount),
              UInt32(a.strides.innerRepeats),
              UInt32(b.strides.dataCount),
              UInt32(b.strides.innerRepeats),
              UInt32(count)
            )))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.binaryOp(a, b, op: op, count: count, dtype: dtype)
    }

    alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName =
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
    let functionName = "\(opName)vs_\(MPSBackend.CastTypes[dtype]!)"

    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "binaryOp") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(aBuf), .float(b.toFloat()), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func binaryOp<T: NumericTensorElement>(
    _ a: T, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.binaryOp(a, b, op: op, count: count, dtype: dtype)
    }

    alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

    let opName =
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
    let functionName = "\(opName)sv_\(MPSBackend.CastTypes[dtype]!)"

    let bBuf = try await gpuBuffer(b)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "binaryOp") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .float(a.toFloat()), .buffer(bBuf), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func mulAdd(
    input: BroadcastData, coeff: BroadcastData, bias: BroadcastData, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await addMulOrMulAdd(
      input: input, a: coeff, b: bias, method: "mul_add", count: count, dtype: dtype)
  }

  override public func addMul(
    input: BroadcastData, bias: BroadcastData, coeff: BroadcastData, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await addMulOrMulAdd(
      input: input, a: bias, b: coeff, method: "add_mul", count: count, dtype: dtype)
  }

  private func addMulOrMulAdd(
    input: BroadcastData, a: BroadcastData, b: BroadcastData, method: String, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(count <= UInt32.max, "\(method) cannot operate on as many as \(count) values")
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      if method == "mul_add" {
        return try await super.mulAdd(input: input, coeff: a, bias: b, count: count, dtype: dtype)
      } else {
        return try await super.addMul(input: input, bias: a, coeff: b, count: count, dtype: dtype)
      }
    }
    let functionName = "\(method)_\(typeName)"

    let inBuf = try await gpuBuffer(input.data)
    let aBuf = try await gpuBuffer(a.data)
    let bBuf = try await gpuBuffer(b.data)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "addMulOrMulAdd") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(inBuf), .buffer(aBuf), .buffer(bBuf),
          .buffer(output),
          .uint(UInt32(input.strides.dataCount)), .uint(UInt32(input.strides.innerRepeats)),
          .uint(UInt32(a.strides.dataCount)), .uint(UInt32(a.strides.innerRepeats)),
          .uint(UInt32(b.strides.dataCount)), .uint(UInt32(b.strides.innerRepeats)),
          .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func normalize<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, epsilon: T, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(count <= UInt32.max, "normalize cannot operate on as many as \(count) values")
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      return try await super.normalize(
        input: input, mean: mean, variance: variance, epsilon: epsilon, count: count, dtype: dtype)
    }
    let functionName = "normalize_\(typeName)"

    let inBuf = try await gpuBuffer(input.data)
    let meanBuf = try await gpuBuffer(mean.data)
    let varianceBuf = try await gpuBuffer(variance.data)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "normalize") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(inBuf), .buffer(meanBuf), .buffer(varianceBuf),
          .buffer(output),
          .opaque(
            (
              epsilon.toFloat(),
              UInt32(input.strides.dataCount),
              UInt32(input.strides.innerRepeats),
              UInt32(mean.strides.dataCount),
              UInt32(mean.strides.innerRepeats),
              UInt32(variance.strides.dataCount),
              UInt32(variance.strides.innerRepeats),
              UInt32(count)
            ))
        )
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func normalizeXGrad<T: TensorElement>(
    variance: BroadcastData, outGrad: BroadcastData, epsilon: T, sign: Float, count: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(
      count <= UInt32.max, "normalizeXGrad cannot operate on as many as \(count) values")
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      return try await super.normalizeXGrad(
        variance: variance, outGrad: outGrad, epsilon: epsilon, sign: sign, count: count,
        dtype: dtype)
    }
    let functionName = "normalize_x_grad_\(typeName)"

    let varianceBuf = try await gpuBuffer(variance.data)
    let outGradBuf = try await gpuBuffer(outGrad.data)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "normalizeXGrad") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(varianceBuf), .buffer(outGradBuf),
          .buffer(output),
          .opaque(
            (
              epsilon.toFloat(),
              sign.toFloat(),
              UInt32(variance.strides.dataCount),
              UInt32(variance.strides.innerRepeats),
              UInt32(outGrad.strides.dataCount),
              UInt32(outGrad.strides.innerRepeats),
              UInt32(count)
            )))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func normalizeVarianceGrad<T: TensorElement>(
    input: BroadcastData, mean: BroadcastData, variance: BroadcastData, outGrad: BroadcastData,
    epsilon: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(
      count <= UInt32.max, "normalizeVarianceGrad cannot operate on as many as \(count) values")
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      return try await super.normalize(
        input: input, mean: mean, variance: variance, epsilon: epsilon, count: count, dtype: dtype)
    }
    let functionName = "normalize_var_grad_\(typeName)"

    let inBuf = try await gpuBuffer(input.data)
    let meanBuf = try await gpuBuffer(mean.data)
    let varianceBuf = try await gpuBuffer(variance.data)
    let outGradBuf = try await gpuBuffer(outGrad.data)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "normalizeVarianceGrad") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(
          enc, .buffer(inBuf), .buffer(meanBuf), .buffer(varianceBuf),
          .buffer(outGradBuf),
          .buffer(output),
          .opaque(
            (
              epsilon.toFloat(),
              UInt32(input.strides.dataCount),
              UInt32(input.strides.innerRepeats),
              UInt32(mean.strides.dataCount),
              UInt32(mean.strides.innerRepeats),
              UInt32(variance.strides.dataCount),
              UInt32(variance.strides.innerRepeats),
              UInt32(outGrad.strides.dataCount),
              UInt32(outGrad.strides.innerRepeats),
              UInt32(count)
            )))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func cast(
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

    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(length: count * outType.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "cast") { buf, enc in
        let state = try getFunction(name: functionName)
        try setArguments(enc, .buffer(aBuf), .buffer(output), .uint(UInt32(count)))
        dispatch1D(enc, state: state, threadCount: count)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func clamp<T: NumericTensorElement>(
    _ a: Tensor.Data, min: T?, max: T?, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let typeName = MPSBackend.CastTypes[dtype] else {
      return try await super.clamp(a, min: min, max: max, count: count, dtype: dtype)
    }
    alwaysAssert(count <= UInt32.max, "cannot apply clamp() to \(count) values")

    let functionName = "clamp_\(typeName)"

    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(length: count * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "clamp") { buf, enc in
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
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func repeated(
    _ a: Tensor.Data, dims: RepeatDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let outCount = dims.outCount
    alwaysAssert(
      outCount <= Int(UInt32.max),
      "cannot apply repeat kernel to this many values")

    let typeName = dtype.metalSizeType

    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(length: outCount * dtype.byteSize)

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "repeated") { buf, enc in
        let state = try getFunction(name: "repeat_\(typeName)")
        try setArguments(
          enc,
          .buffer(aBuf),
          .buffer(output),
          .opaque((UInt32(dims.innerCount), UInt32(dims.outerCount), UInt32(dims.repeatCount)))
        )
        dispatch1D(enc, state: state, threadCount: outCount)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func concat(
    _ inputs: [Tensor.Data], outerCount: Int, innerCounts: [Int], dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    alwaysAssert(inputs.count == innerCounts.count)
    var inBufs = [MTLBuffer]()
    for input in inputs {
      inBufs.append(try await gpuBuffer(input))
    }
    let totalInner = innerCounts.sum()
    let buffer = try await allocate(length: outerCount * totalInner * dtype.byteSize)

    let typeName = dtype.metalSizeType

    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "concat") { buf, enc in
        for (i, innerCount) in innerCounts.enumerated() {
          let offset = innerCounts[..<i].sum()
          let state = try getFunction(name: "strided_copy_\(typeName)")
          try setArguments(
            enc,
            .buffer(inBufs[i]),
            .buffer(buffer),
            .opaque((UInt32(innerCount), UInt32(totalInner), UInt32(outerCount), UInt32(offset)))
          )
          dispatch1D(enc, state: state, threadCount: innerCount * outerCount)
        }
      }
      return GPUData(backend: self, buffer: buffer, completion: completion)
    }
  }

  override public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let aBuf = try await gpuBuffer(a)
    let idxBuf = try await gpuBuffer(s.indices)
    let output = try await allocate(length: s.gatherOutCount * dtype.byteSize)
    return try await serialize { [self] in
      let completion = try completionBufferAndEncoder(label: "gather") { buf, enc in
        let typeName = dtype.metalSizeType
        let functionName = "gather\(s.broadcasted ? "_bcast" : "")_\(typeName)"
        let state = try getFunction(name: functionName)
        try setArguments(
          enc,
          .buffer(aBuf),
          .buffer(idxBuf),
          .buffer(output),
          .uint(UInt32(s.outerCount)),
          .uint(UInt32(s.outCount)),
          .uint(UInt32(s.middleCount)),
          .uint(UInt32(s.innerCount))
        )
        dispatch1D(enc, state: state, threadCount: s.gatherOutCount)
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  override public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let aBuf = try await gpuBuffer(a)
    let idxBuf = try await gpuBuffer(s.indices)

    let outBytes = s.gatherInCount * dtype.byteSize
    let output = try await allocate(length: outBytes)

    if !s.indicesAreUnique, let mpsDType = dtype.mpsDType {
      return try await serialize { [self] in
        let scatter = self.createScatter(scatter: s, dtype: mpsDType)
        let completion = completionBuffer(label: "scatter") { buf in
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
                  descriptor: MPSVectorDescriptor(length: s.indicesCount, dataType: .int64))),
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
        return GPUData(backend: self, buffer: output, completion: completion)
      }
    }

    if !s.indicesAreUnique {
      // Fallback on CPU implementation if for some reason there is a dtype
      // that MPS cannot handle above.
      return try await super.scatter(a, s, dtype: dtype)
    }

    return try await serialize { [self] in
      let completion = try completionBuffer(label: "scatterUnique") { buf in
        let typeName = dtype.metalSizeType
        let functionName = "scatter\(s.broadcasted ? "_bcast" : "")_\(typeName)"
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
          .uint(UInt32(s.middleCount)), .uint(UInt32(s.innerCount)))
        dispatch1D(computeEncoder, state: state, threadCount: s.gatherOutCount)
        computeEncoder.endEncoding()
      }
      return GPUData(backend: self, buffer: output, completion: completion)
    }
  }

  private func createScatter(scatter: ScatterGatherIndices, dtype: MPSDataType) -> TwoToOneGraph {
    let key = ScatterKey(
      broadcasted: scatter.broadcasted, outCount: scatter.outCount, outerCount: scatter.outerCount,
      middleCount: scatter.middleCount, innerCount: scatter.innerCount, dtype: dtype)
    if let r = scatters[key] {
      return r
    } else {
      let graph = MPSGraph()
      let inputShape = [scatter.outerCount, scatter.outCount, scatter.innerCount]
      let indexShape =
        scatter.broadcasted
        ? [scatter.outCount] : [scatter.outerCount, scatter.outCount, scatter.innerCount]
      let outputShape = [scatter.outerCount, scatter.middleCount, scatter.innerCount]
      let input = graph.placeholder(
        shape: mpsShape([inputShape.product()]), dataType: dtype, name: "input")
      let indices = graph.placeholder(
        shape: mpsShape([indexShape.product()]), dataType: .int64, name: "indices")
      let inputReshaped = graph.reshape(input, shape: mpsShape(inputShape), name: "inputReshaped")
      let indicesReshaped = graph.reshape(
        indices, shape: mpsShape(indexShape), name: "indexReshaped")
      let unshapedOutput =
        if scatter.broadcasted {
          graph.scatter(
            inputReshaped, indices: indicesReshaped, shape: mpsShape(outputShape), axis: 1,
            mode: .add, name: "scatter")
        } else {
          graph.scatterAlongAxis(
            1, updates: inputReshaped, indices: indicesReshaped, shape: mpsShape(outputShape),
            mode: .add, name: "scatter")
        }
      let output = graph.reshape(
        unshapedOutput, shape: mpsShape([outputShape.product()]), name: "output")
      let r = TwoToOneGraph(graph: graph, inputA: input, inputB: indices, output: output)
      scatters[key] = r
      return r
    }
  }

  override public func axisPermutation(permutation: [Int], shape: [Int]) async throws -> Tensor.Data
  {
    let outputCount = shape.product()
    alwaysAssert(outputCount <= Int(UInt32.max), "cannot apply kernel to this many values")

    let buffer = try await allocate(length: outputCount * Tensor.DType.int64.byteSize)
    return try await serialize { [self] in
      let oldStrides = stridesForShape(shape)
      let permutedStrides = permutation.map { oldStrides[$0] }
      let newShape = permutation.map { shape[$0] }
      let newStrides = stridesForShape(newShape)

      let completion = try completionBufferAndEncoder(label: "axisPermutation") { buf, enc in
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
      return GPUData(backend: self, buffer: buffer, completion: completion)
    }
  }

  override public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.reduce(a, op: op, dims: dims, dtype: dtype)
    }

    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(
      length: dims.outCount * (op == .sum ? dtype : Tensor.DType.int64).byteSize)
    return try await serialize { [self] in
      let red = self.createReduction(op: op, dims: dims, dtype: mpsDType)
      let completion = completionBuffer(label: "reduce") { buf in
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
      return GPUData(backend: self, buffer: output, completion: completion)
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

  override public func logSoftmax(
    _ a: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.logSoftmax(
        a, outerCount: outerCount, middleCount: middleCount, innerCount: innerCount, dtype: dtype)
    }

    let totalCount = outerCount * middleCount * innerCount
    let aBuf = try await gpuBuffer(a)
    let output = try await allocate(length: totalCount * dtype.byteSize)
    return try await serialize { [self] in
      let op = self.createLogSoftmax(
        outerCount: outerCount, middleCount: middleCount, innerCount: innerCount, dtype: mpsDType)
      let completion = completionBuffer(label: "logSoftmax") { buf in
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
      return GPUData(backend: self, buffer: output, completion: completion)
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

  override public func logSoftmaxGrad(
    _ a: Tensor.Data, _ outGrad: Tensor.Data, outerCount: Int, middleCount: Int, innerCount: Int,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.logSoftmax(
        a, outerCount: outerCount, middleCount: middleCount, innerCount: innerCount, dtype: dtype)
    }

    let aBuf = try await gpuBuffer(a)
    let outGradBuf = try await gpuBuffer(outGrad)

    let totalCount = outerCount * middleCount * innerCount
    let output = try await allocate(length: totalCount * dtype.byteSize)

    return try await serialize { [self] in
      let op = self.createLogSoftmaxGrad(
        outerCount: outerCount, middleCount: middleCount, innerCount: innerCount, dtype: mpsDType)
      let completion = completionBuffer(label: "logSoftmaxGrad") { buf in
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
      return GPUData(backend: self, buffer: output, completion: completion)
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

  override public func matmul(
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

  override public func batchedMatmul(
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

    let output = try await allocate(length: matrixCount * rows * cols * dtype.byteSize)
    let aBuf = try await gpuBuffer(a)
    let bBuf = try await gpuBuffer(b)

    return try await serialize { [self] in
      let mm = self.createMatmul(
        transA: transA, transB: transB, batch: matrixCount, rows: rows, inner: inner, cols: cols,
        dtype: mpsDType)
      let completion = completionBuffer(label: "batchedMatmul") { buf in
        alwaysAssert(
          aBuf.allocatedSize >= matrixCount * aShape.0 * aShape.1 * dtype.byteSize,
          "matrix A buffer underflow")
        alwaysAssert(
          bBuf.allocatedSize >= matrixCount * bShape.0 * bShape.1 * dtype.byteSize,
          "matrix B buffer underflow \(matrixCount) * \(bShape) * \(dtype.byteSize) vs \(bBuf.allocatedSize)"
        )
        alwaysAssert(
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
      return GPUData(backend: self, buffer: output, completion: completion)
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
    let output = try await allocate(length: outShape.product() * dtype.byteSize)
    let imageBuf = try await gpuBuffer(image)
    let kernelBuf = try await gpuBuffer(kernel)

    return try await serialize { [self] in
      let op = try self.createConv2D(
        config, batch: batch, kind: transpose ? .transpose : .forward, dtype: mpsDType)
      let completion = completionBuffer(label: "conv2D") { buf in
        alwaysAssert(
          imageBuf.allocatedSize >= imageShape.product() * dtype.byteSize,
          "input image buffer underflow")
        alwaysAssert(
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
      return GPUData(backend: self, buffer: output, completion: completion)
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

  override public func conv1D(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2D(
      try conv1DToConv2D(config), batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv1DTranspose(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2DTranspose(
      try conv1DToConv2D(config), batch: batch, image: image, kernel: kernel, dtype: dtype)
  }

  override public func conv1DKernelGrad(
    _ config: Conv1DConfig, batch: Int, image: Tensor.Data, outGrad: Tensor.Data,
    dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2DKernelGrad(
      try conv1DToConv2D(config), batch: batch, image: image, outGrad: outGrad, dtype: dtype)
  }

  override public func conv2D(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2D(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype, transpose: false)
  }

  override public func conv2DTranspose(
    _ config: Conv2DConfig, batch: Int, image: Tensor.Data, kernel: Tensor.Data, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await conv2D(
      config, batch: batch, image: image, kernel: kernel, dtype: dtype, transpose: true)
  }

  override public func conv2DKernelGrad(
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
    let output = try await allocate(length: kernelShape.product() * dtype.byteSize)
    let imageBuf = try await gpuBuffer(image)
    let outGradBuf = try await gpuBuffer(outGrad)
    return try await serialize { [self] in
      let op = try self.createConv2D(config, batch: batch, kind: .kernelGrad, dtype: mpsDType)
      let completion = completionBuffer(label: "conv2DKernelGrad") { buf in
        alwaysAssert(
          imageBuf.allocatedSize >= imageShape.product() * dtype.byteSize,
          "input image buffer underflow")
        alwaysAssert(
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
      return GPUData(backend: self, buffer: output, completion: completion)
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

  override public func defaultRandom() async throws -> RandomGenerator {
    try await serialize { [self] in
      if let d = defaultRNG {
        return d
      } else {
        defaultRNG = MPSRandomGenerator(mpsBackend: self, seed: Int.random(in: 0..<1_000_000_000))
        return defaultRNG!
      }
    }
  }

  override public func createRandom() async throws -> RandomGenerator {
    try await serialize {
      MPSRandomGenerator(mpsBackend: self, seed: Int.random(in: 0..<1_000_000_000))
    }
  }

  internal enum KernelArgument {
    case uint(UInt32)
    case uints([UInt32])
    case float(Float)
    case int64(Int64)
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
      let result = try allocateSync(length: 4 * x.count)
      result.contents().copyMemory(from: bytes.baseAddress!, byteCount: 4 * x.count)
      return result
    }
  }

  internal func makeFloatBuffer(_ x: Float) throws -> MTLBuffer {
    var x = x
    let result = try allocateSync(length: 4)
    result.contents().copyMemory(from: &x, byteCount: 4)
    return result
  }

  internal func makeInt64Buffer(_ x: Int64) throws -> MTLBuffer {
    var x = x
    let result = try allocateSync(length: 8)
    result.contents().copyMemory(from: &x, byteCount: 8)
    return result
  }

  internal func makeDataBuffer(_ x: Data) throws -> MTLBuffer {
    let result = try allocateSync(length: x.count)
    x.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
      result.contents().copyMemory(from: bytes.baseAddress!, byteCount: x.count)
    }
    return result
  }

  class CompletionCallback {
    private var lock: NSLock = NSLock()
    private var _result: Result<(), Error>? = nil
    private var _continue: ((Result<(), Error>) -> Void)? = nil

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

    func putContinuation(_ f: @escaping (Result<(), Error>) -> Void) {
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

  internal func completionBuffer(label: String, _ action: (MTLCommandBuffer) throws -> Void)
    rethrows -> Task<
      (), Error
    >
  {
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
      cb.putResult(result)
    }
    buf.commit()
    return Task.detached {
      try await withCheckedThrowingContinuation { continuation in
        cb.putContinuation(continuation.resume)
      }
    }
  }

  internal func completionBufferAndEncoder(
    label: String, _ action: (MTLCommandBuffer, MTLComputeCommandEncoder) throws -> Void
  ) throws -> Task<
    (), Error
  > {
    try completionBuffer(label: label) { buf in
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
