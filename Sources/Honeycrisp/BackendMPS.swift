import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

open class MPSBackend: CPUBackend {

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
        "\(dist == .normal ? "randn" : "rand")_\(dtype == .float16 ? "fp16" : "fp32")"
      let output = try await mpsBackend.allocate(length: count * dtype.byteSize)

      return try await mpsBackend.serialize { [self] in
        let completion = try mpsBackend.completionBuffer { buf in
          let state = try mpsBackend.getFunction(name: functionName)
          guard let computeEncoder = buf.makeComputeCommandEncoder() else {
            throw BackendError.kernelFailed("could not create compute encoder")
          }
          computeEncoder.setBuffer(output, offset: 0, index: 0)
          computeEncoder.setBuffer(try mpsBackend.makeUIntBuffer(UInt32(seed)), offset: 0, index: 1)
          computeEncoder.setBuffer(
            try mpsBackend.makeUIntBuffer(UInt32(seed >> 32)), offset: 0, index: 2)
          computeEncoder.setBuffer(
            try mpsBackend.makeUIntBuffer(UInt32(offset)), offset: 0, index: 3)
          computeEncoder.setBuffer(
            try mpsBackend.makeUIntBuffer(UInt32(count)), offset: 0, index: 4)

          let chunkSize = dist == .normal ? 2 : 4
          let totalThreads = (count + chunkSize - 1) / chunkSize
          mpsBackend.dispatch1D(computeEncoder, state: state, threadCount: totalThreads)
          computeEncoder.endEncoding()
          offset += UInt64(totalThreads)
        }
        return Tensor.Data(backend: mpsBackend, buffer: output, completeOnAllDevices: completion)
      }
    }
  }

  private var queue = DispatchQueue(label: "mps-backend-worker")
  private var commandQueue: MTLCommandQueue? = nil
  private var matmuls: [MatmulKey: TwoToOneGraph] = [:]
  private var conv2D: [Conv2DKey: TwoToOneGraph] = [:]
  private var reductions: [ReduceKey: OneToOneGraph] = [:]
  private var defaultRNG: MPSRandomGenerator? = nil
  private var functions: [String: MTLComputePipelineState] = [:]

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
    let library = try (self._device!).makeLibrary(
      source: MPSBackend.KernelCode, options: MTLCompileOptions())

    // Lookup all functions in the library.
    var names = [
      "vector_pow_fp16", "vector_pow_fp32", "log_fp16", "log_fp32", "recip_fp16", "recip_fp32",
      "exp_fp16", "exp_fp32", "sigmoid_fp16", "sigmoid_fp32", "sigmoid_grad_fp16",
      "sigmoid_grad_fp32", "gelu_fp32", "gelu_fp16", "gelu_grad_fp32", "gelu_grad_fp16", "sin_fp32",
      "sin_fp16", "cos_fp32", "cos_fp16", "minus_sin_fp32", "minus_sin_fp16", "relu_fp32",
      "relu_fp16", "relu_grad_fp32", "relu_grad_fp16", "abs_fp16", "abs_fp32", "abs_grad_fp16",
      "abs_grad_fp32", "repeat", "rand_fp32", "rand_fp16", "randn_fp32", "randn_fp16",
    ]
    for type in ["char", "short", "int", "long"] {
      for mode in ["", "_bcast"] {
        for op in ["gather", "scatter"] {
          names.append("\(op)\(mode)_\(type)")
        }
      }
    }
    for op in ["add", "sub", "mul", "div", "mod"] {
      for type in ["fp16", "fp32"] {
        for args in ["vv", "sv", "vs"] {
          names.append("\(op)\(args)_\(type)")
        }
      }
    }
    for name in names {
      guard let f = library.makeFunction(name: name) else {
        throw BackendError.kernelFailed("could not create kernel with name '\(name)'")
      }
      functions[name] = try (self._device!).makeComputePipelineState(function: f)
    }
  }

  internal func waitForGPUData(_ xs: Tensor.Data...) async throws {
    for x in xs {
      if x.backend === self && x.completeOnAllDevices != nil {
        continue
      }
      try await waitForData(x)
    }
  }

  internal func getFunction(name: String) throws -> MTLComputePipelineState {
    guard let f = functions[name] else {
      throw BackendError.kernelFailed("no kernel with name '\(name)'")
    }
    return f
  }

  override public func pow<T: NumericTensorElement>(
    _ a: Tensor.Data, _ b: T, count: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.pow(a, b, count: count, dtype: dtype)
    }

    alwaysAssert(count <= Int(UInt32.max), "cannot apply kernel to this many values")
    let output = try await allocate(length: count * dtype.byteSize)

    try await waitForGPUData(a)
    return try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let funcName = "vector_pow_\(dtype == .float16 ? "fp16" : "fp32")"
        let state = try getFunction(name: funcName)
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .buffer(a.buffer), .buffer(output), .float(b.toFloat()),
          .uint(UInt32(count)))
        dispatch1D(computeEncoder, state: state, threadCount: count)
        computeEncoder.endEncoding()

      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
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
    let functionName = "\(namePrefix)_\(dtype == .float16 ? "fp16" : "fp32")"
    let output = try await allocate(length: count * dtype.byteSize)

    try await waitForGPUData(a)
    return try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let state = try getFunction(name: functionName)
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(computeEncoder, .buffer(a.buffer), .buffer(output), .uint(UInt32(count)))
        dispatch1D(computeEncoder, state: state, threadCount: count)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }
  }

  override public func binaryOp(
    _ a: Tensor.Data, _ b: Tensor.Data, op: NumericBinaryOp, count: Int, dtype: Tensor.DType
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
    let functionName = "\(opName)vv_\(dtype == .float16 ? "fp16" : "fp32")"
    let output = try await allocate(length: count * dtype.byteSize)

    try await waitForGPUData(a)
    return try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let state = try getFunction(name: functionName)
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .buffer(a.buffer), .buffer(b.buffer), .buffer(output),
          .uint(UInt32(count)))
        dispatch1D(computeEncoder, state: state, threadCount: count)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
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
    let functionName = "\(opName)vs_\(dtype == .float16 ? "fp16" : "fp32")"
    let output = try await allocate(length: count * dtype.byteSize)

    try await waitForGPUData(a)
    return try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let state = try getFunction(name: functionName)
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .buffer(a.buffer), .float(b.toFloat()), .buffer(output),
          .uint(UInt32(count)))
        dispatch1D(computeEncoder, state: state, threadCount: count)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
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
    let functionName = "\(opName)sv_\(dtype == .float16 ? "fp16" : "fp32")"
    let output = try await allocate(length: count * dtype.byteSize)

    try await waitForGPUData(b)
    return try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let state = try getFunction(name: functionName)
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .float(a.toFloat()), .buffer(b.buffer), .buffer(output),
          .uint(UInt32(count)))
        dispatch1D(computeEncoder, state: state, threadCount: count)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }
  }

  override public func repeated(
    _ a: Tensor.Data, outerCount: Int, innerCount: Int, repeats: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    let outBytes = innerCount * outerCount * repeats * dtype.byteSize
    alwaysAssert(
      outBytes <= Int(UInt32.max),
      "cannot apply kernel to this many values")

    let output = try await allocate(length: outBytes)

    try await waitForGPUData(a)
    return try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let state = try getFunction(name: "repeat")
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .buffer(a.buffer), .buffer(output),
          .uint(UInt32(innerCount * dtype.byteSize)), .uint(UInt32(outerCount)),
          .uint(UInt32(repeats)))
        dispatch1D(computeEncoder, state: state, threadCount: outBytes)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }
  }

  override public func gather(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForGPUData(a, s.indices)
    let output = try await allocate(length: s.gatherOutCount * dtype.byteSize)
    return try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let typeName =
          switch dtype {
          case .bool:
            "char"
          case .float16:
            "short"
          case .float32:
            "int"
          case .int64:
            "long"
          }
        let functionName = "gather\(s.broadcasted ? "_bcast" : "")_\(typeName)"
        let state = try getFunction(name: functionName)

        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .buffer(a.buffer), .buffer(s.indices.buffer), .buffer(output),
          .uint(UInt32(s.outerCount)), .uint(UInt32(s.outCount)), .uint(UInt32(s.middleCount)),
          .uint(UInt32(s.innerCount)))
        dispatch1D(computeEncoder, state: state, threadCount: s.gatherOutCount)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }
  }

  override public func scatter(
    _ a: Tensor.Data, _ s: ScatterGatherIndices, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    try await waitForGPUData(a, s.indices)
    let output = try await allocate(length: s.gatherInCount * dtype.byteSize)
    let counts = try await allocate(length: s.gatherInCount * 4)
    let needsAddition = try await allocate(length: 4)

    let result = try await serialize { [self] in
      let completion = try completionBuffer { buf in
        let typeName =
          switch dtype {
          case .bool:
            "char"
          case .float16:
            "short"
          case .float32:
            "int"
          case .int64:
            "long"
          }
        let functionName = "scatter\(s.broadcasted ? "_bcast" : "")_\(typeName)"
        let state = try getFunction(name: functionName)

        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        try setArguments(
          computeEncoder, .buffer(a.buffer), .buffer(s.indices.buffer), .buffer(output),
          .buffer(counts), .buffer(needsAddition), .uint(UInt32(s.outerCount)),
          .uint(UInt32(s.outCount)), .uint(UInt32(s.middleCount)), .uint(UInt32(s.innerCount)))
        dispatch1D(computeEncoder, state: state, threadCount: s.gatherOutCount)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }

    // We may need to fallback to the CPU implementation
    // if summation is required.
    let _ = try await result.completeOnAllDevices!.value
    if needsAddition.contents().load(as: UInt32.self) != 0 {
      return try await super.scatter(a, s, dtype: dtype)
    } else {
      return Tensor.Data(backend: self, buffer: result.buffer)
    }
  }

  override public func reduce(
    _ a: Tensor.Data, op: ReduceOp, dims: ReduceDims, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType, op == .sum else {
      return try await super.reduce(a, op: op, dims: dims, dtype: dtype)
    }

    let output = try await allocate(length: dims.outCount * dtype.byteSize)
    try await waitForGPUData(a)
    return try await serialize { [self] in
      let red = self.createReduction(op: op, dims: dims, dtype: mpsDType)
      let completion = completionBuffer { buf in
        red.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            red.input: MPSGraphTensorData(
              MPSVector(
                buffer: a.buffer,
                descriptor: MPSVectorDescriptor(
                  length: dims.inCount, dataType: mpsDType)))
          ],
          targetOperations: nil,
          resultsDictionary: [
            red.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(length: dims.outCount, dataType: mpsDType)))
          ],
          executionDescriptor: nil)
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
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
      let unshapedOutput = graph.reductionSum(with: reshaped, axis: 1, name: "unshapedOutput")
      let output = graph.reshape(unshapedOutput, shape: mpsShape([dims.outCount]), name: "output")
      let r = OneToOneGraph(graph: graph, input: input, output: output)
      reductions[key] = r
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
    try await waitForGPUData(a, b)
    return try await serialize { [self] in
      let mm = self.createMatmul(
        transA: transA, transB: transB, batch: matrixCount, rows: rows, inner: inner, cols: cols,
        dtype: mpsDType)
      let completion = completionBuffer { buf in
        alwaysAssert(
          a.buffer.allocatedSize >= matrixCount * aShape.0 * aShape.1 * dtype.byteSize,
          "matrix A buffer underflow")
        alwaysAssert(
          b.buffer.allocatedSize >= matrixCount * bShape.0 * bShape.1 * dtype.byteSize,
          "matrix B buffer underflow \(matrixCount) * \(bShape) * \(dtype.byteSize) vs \(b.buffer.allocatedSize)"
        )
        alwaysAssert(
          output.allocatedSize >= matrixCount * rows * cols * dtype.byteSize,
          "output matrix buffer underflow")
        mm.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            mm.inputA: MPSGraphTensorData(
              MPSMatrix(
                buffer: a.buffer,
                descriptor: MPSMatrixDescriptor(
                  rows: aShape.0, columns: aShape.1, matrices: matrixCount,
                  rowBytes: aShape.1 * dtype.byteSize,
                  matrixBytes: aShape.0 * aShape.1 * dtype.byteSize, dataType: mpsDType)),
              rank: matrixCount > 1 ? 3 : 2),
            mm.inputB: MPSGraphTensorData(
              MPSMatrix(
                buffer: b.buffer,
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
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
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
    try await waitForGPUData(image, kernel)
    return try await serialize { [self] in
      let op = try self.createConv2D(
        config, batch: batch, kind: transpose ? .transpose : .forward, dtype: mpsDType)
      let completion = completionBuffer { buf in
        alwaysAssert(
          image.buffer.allocatedSize >= imageShape.product() * dtype.byteSize,
          "input image buffer underflow")
        alwaysAssert(
          kernel.buffer.allocatedSize >= kernelShape.product() * dtype.byteSize,
          "kernel buffer underflow")
        op.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            op.inputA: MPSGraphTensorData(
              MPSVector(
                buffer: image.buffer,
                descriptor: MPSVectorDescriptor(length: imageShape.product(), dataType: mpsDType))),
            op.inputB: MPSGraphTensorData(
              MPSVector(
                buffer: kernel.buffer,
                descriptor: MPSVectorDescriptor(length: kernelShape.product(), dataType: mpsDType))),
          ], targetOperations: nil,
          resultsDictionary: [
            op.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(length: outShape.product(), dataType: mpsDType)))
          ], executionDescriptor: nil)
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
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
    try await waitForGPUData(image, outGrad)
    return try await serialize { [self] in
      let op = try self.createConv2D(config, batch: batch, kind: .kernelGrad, dtype: mpsDType)
      let completion = completionBuffer { buf in
        alwaysAssert(
          image.buffer.allocatedSize >= imageShape.product() * dtype.byteSize,
          "input image buffer underflow")
        alwaysAssert(
          outGrad.buffer.allocatedSize >= outShape.product() * dtype.byteSize,
          "output gradient buffer underflow")
        op.graph.encode(
          to: buf as! MPSCommandBuffer,
          feeds: [
            op.inputA: MPSGraphTensorData(
              MPSVector(
                buffer: image.buffer,
                descriptor: MPSVectorDescriptor(length: imageShape.product(), dataType: mpsDType))),
            op.inputB: MPSGraphTensorData(
              MPSVector(
                buffer: outGrad.buffer,
                descriptor: MPSVectorDescriptor(length: outShape.product(), dataType: mpsDType))),
          ], targetOperations: nil,
          resultsDictionary: [
            op.output: MPSGraphTensorData(
              MPSVector(
                buffer: output,
                descriptor: MPSVectorDescriptor(length: kernelShape.product(), dataType: mpsDType)))
          ], executionDescriptor: nil)
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
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
    case float(Float)
    case buffer(any MTLBuffer)

    internal func intoBuffer(_ b: MPSBackend) throws -> MTLBuffer {
      switch self {
      case .uint(let x):
        try b.makeUIntBuffer(x)
      case .float(let x):
        try b.makeFloatBuffer(x)
      case .buffer(let x):
        x
      }
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
    var x = x
    guard let result = (try device).makeBuffer(bytes: &x, length: 4, options: []) else {
      throw BackendError.allocationFailed(4)
    }
    return result
  }

  internal func makeFloatBuffer(_ x: Float) throws -> MTLBuffer {
    var x = x
    guard let result = (try device).makeBuffer(bytes: &x, length: 4, options: []) else {
      throw BackendError.allocationFailed(4)
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

  internal func completionBuffer(_ action: (MTLCommandBuffer) throws -> Void) rethrows -> Task<
    (), Error
  > {
    let buf = MPSCommandBuffer(commandBuffer: commandQueue!.makeCommandBuffer()!)
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

  private static let KernelCode = """
      #include <metal_stdlib>
      using namespace metal;

      inline float safe_tanh(float x) {
          return (x < -10 ? -1 : (x > 10 ? 1 : tanh(x)));
      }

      template <typename T>
      inline T pythonFmod(T lhs, T rhs) {
          if (rhs < 0) {
              return -pythonFmod(-lhs, -rhs);
          } else if (lhs < 0) {
              return fmod(rhs - fmod(rhs - lhs, rhs), rhs);
          } else {
              return fmod(lhs, rhs);
          }
      }

      #define BINARY_KERNELS(name, expr) \
          kernel void name##vv_fp16(device const half* a [[buffer(0)]], \
                                    device const half* b [[buffer(1)]], \
                                    device half* c [[buffer(2)]], \
                                    constant uint &N [[buffer(3)]], \
                                    uint id [[thread_position_in_grid]]) { \
            if (id < N) { \
              half x = a[id]; \
              half y = b[id]; \
              c[id] = expr; \
            } \
          } \
          kernel void name##vv_fp32(device const float* a [[buffer(0)]], \
                                    device const float* b [[buffer(1)]], \
                                    device float* c [[buffer(2)]], \
                                    constant uint &N [[buffer(3)]], \
                                    uint id [[thread_position_in_grid]]) { \
            if (id < N) { \
              float x = a[id]; \
              float y = b[id]; \
              c[id] = expr; \
            } \
          } \
          kernel void name##vs_fp16(device const half* a [[buffer(0)]], \
                                    device const float& b [[buffer(1)]], \
                                    device half* c [[buffer(2)]], \
                                    constant uint &N [[buffer(3)]], \
                                    uint id [[thread_position_in_grid]]) { \
            if (id < N) { \
              half x = a[id]; \
              half y = (half)b; \
              c[id] = expr; \
            } \
          } \
          kernel void name##vs_fp32(device const float* a [[buffer(0)]], \
                                    device const float& b [[buffer(1)]], \
                                    device float* c [[buffer(2)]], \
                                    constant uint &N [[buffer(3)]], \
                                    uint id [[thread_position_in_grid]]) { \
            if (id < N) { \
              float x = a[id]; \
              float y = b; \
              c[id] = expr; \
            } \
          } \
          kernel void name##sv_fp16(device const float& a [[buffer(0)]], \
                                    device const half* b [[buffer(1)]], \
                                    device half* c [[buffer(2)]], \
                                    constant uint &N [[buffer(3)]], \
                                    uint id [[thread_position_in_grid]]) { \
            if (id < N) { \
              half x = (half)a; \
              half y = b[id]; \
              c[id] = expr; \
            } \
          } \
          kernel void name##sv_fp32(device const float& a [[buffer(0)]], \
                                    device const float* b [[buffer(1)]], \
                                    device float* c [[buffer(2)]], \
                                    constant uint &N [[buffer(3)]], \
                                    uint id [[thread_position_in_grid]]) { \
            if (id < N) { \
              float x = a; \
              float y = b[id]; \
              c[id] = expr; \
            } \
          } \

      BINARY_KERNELS(add, x+y)
      BINARY_KERNELS(sub, x-y)
      BINARY_KERNELS(mul, x*y)
      BINARY_KERNELS(div, x/y)
      BINARY_KERNELS(mod, (pythonFmod(x,y)))

      kernel void vector_pow_fp16(device const half* input [[buffer(0)]],
                                  device half* output [[buffer(1)]],
                                  constant float &exponent [[buffer(2)]],
                                  constant uint &N [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = half(pow(float(input[id]), exponent));
        }
      }

      kernel void vector_pow_fp32(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  constant float &exponent [[buffer(2)]],
                                  constant uint &N [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = pow(input[id], exponent);
        }
      }

      kernel void log_fp16(device const half* input [[buffer(0)]],
                           device half* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = log(input[id]);
        }
      }

      kernel void log_fp32(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = log(input[id]);
        }
      }

      kernel void recip_fp16(device const half* input [[buffer(0)]],
                             device half* output [[buffer(1)]],
                             constant uint &N [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = 1 / input[id];
        }
      }

      kernel void recip_fp32(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint &N [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = 1 / input[id];
        }
      }

      kernel void exp_fp16(device const half* input [[buffer(0)]],
                           device half* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = exp(input[id]);
        }
      }

      kernel void exp_fp32(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = exp(input[id]);
        }
      }

      kernel void sigmoid_fp16(device const half* input [[buffer(0)]],
                               device half* output [[buffer(1)]],
                               constant uint &N [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = (safe_tanh(input[id] / 2) + 1) / 2;
        }
      }

      kernel void sigmoid_fp32(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint &N [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = (safe_tanh(input[id] / 2) + 1) / 2;
        }
      }

      kernel void sigmoid_grad_fp16(device const half* input [[buffer(0)]],
                                    device half* output [[buffer(1)]],
                                    constant uint &N [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id < N) {
            half s = (safe_tanh(input[id] / 2) + 1) / 2;
            output[id] = s * (1 - s);
        }
      }

      kernel void sigmoid_grad_fp32(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint &N [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id < N) {
            half s = (safe_tanh(input[id] / 2) + 1) / 2;
            output[id] = s * (1 - s);
        }
      }

      kernel void gelu_fp32(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint &N [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = input[id];
              output[id] = 0.5 * x * (1.0 + safe_tanh(0.797884561 * (x + 0.044715 * pow(x, 3.0))));
          }
      }

      kernel void gelu_fp16(device const half* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant uint &N [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = float(input[id]);
              output[id] = half(0.5 * x * (1.0 + safe_tanh(0.797884561 * (x + 0.044715 * pow(x, 3.0)))));
          }
      }

      kernel void gelu_grad_fp32(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint &N [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = input[id];
              float tanhTerm = safe_tanh(0.035677408145115 * pow(x, 3.0) + 0.797884561 * x);
              output[id] = 0.5 * x * (1.0 - pow(tanhTerm, 2.0)) * (0.107032224435345 * pow(x, 2.0) + 0.797884561)
                          + 0.5 * tanhTerm + 0.5;
          }
      }

      kernel void gelu_grad_fp16(device const half* input [[buffer(0)]],
                                 device half* output [[buffer(1)]],
                                 constant uint &N [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = float(input[id]);
              float tanhTerm = safe_tanh(0.035677408145115 * pow(x, 3.0) + 0.797884561 * x);
              output[id] = half(
                  0.5 * x * (1.0 - pow(tanhTerm, 2.0)) * (0.107032224435345 * pow(x, 2.0) + 0.797884561)
                  + 0.5 * tanhTerm + 0.5);
          }
      }

      kernel void sin_fp32(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = sin(input[id]);
          }
      }

      kernel void sin_fp16(device const half* input [[buffer(0)]],
                           device half* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = sin(input[id]);
          }
      }

      kernel void cos_fp32(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = cos(input[id]);
          }
      }


      kernel void cos_fp16(device const half* input [[buffer(0)]],
                           device half* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = cos(input[id]);
          }
      }

      kernel void minus_sin_fp32(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint &N [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = -sin(input[id]);
          }
      }


      kernel void minus_sin_fp16(device const half* input [[buffer(0)]],
                                 device half* output [[buffer(1)]],
                                 constant uint &N [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = -sin(input[id]);
          }
      }

      kernel void relu_fp32(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint &N [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = input[id];
              output[id] = x > 0.0f ? x : 0.0f;
          }
      }

      kernel void relu_fp16(device const half* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant uint &N [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
          if (id < N) {
              half x = input[id];
              output[id] = x > half(0.0) ? x : half(0.0);
          }
      }

      kernel void relu_grad_fp32(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint &N [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = input[id];
              output[id] = x > 0.0f ? 1.0f : 0.0f;
          }
      }


      kernel void relu_grad_fp16(device const half* input [[buffer(0)]],
                                 device half* output [[buffer(1)]],
                                 constant uint &N [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
          if (id < N) {
              half x = input[id];
              output[id] = x > half(0.0) ? half(1.0) : half(0.0);
          }
      }

      kernel void abs_fp32(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = abs(input[id]);
          }
      }


      kernel void abs_fp16(device const half* input [[buffer(0)]],
                           device half* output [[buffer(1)]],
                           constant uint &N [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = abs(input[id]);
          }
      }

      kernel void abs_grad_fp32(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint &N [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = input[id] < 0 ? -1 : 1;
          }
      }


      kernel void abs_grad_fp16(device const half* input [[buffer(0)]],
                                device half* output [[buffer(1)]],
                                constant uint &N [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
          if (id < N) {
              output[id] = input[id] < 0 ? -1 : 1;
          }
      }

      kernel void repeat(device const char* input [[buffer(0)]],
                         device char* output [[buffer(1)]],
                         constant uint &inner [[buffer(2)]],
                         constant uint &outer [[buffer(3)]],
                         constant uint &reps [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
          if (id < inner * outer * reps) {
              uint sourceIdx = (id % inner) + (id / (inner * reps)) * inner;
              output[id] = input[sourceIdx];
          }
      }

      constant uint PHILOX_ROUND_A = 0xD2511F53;
      constant uint PHILOX_ROUND_B = 0xCD9E8D57;
      constant uint PHILOX_KEY_A = 0x9E3779B9;
      constant uint PHILOX_KEY_B = 0xBB67AE85;
      constant constexpr float M_PI = 3.14159265358979323846264338327950288;

      inline uint umulhi(uint x, uint y) {
        uint x0 = x & 0xffff;
        uint x1 = x >> 16;
        uint y0 = y & 0xffff;
        uint y1 = y >> 16;

        // Three terms that sum to the total product
        uint p0 = x0 * y0;
        uint p1 = x1 * y0 + x0 * y1; // downshifted by 16
        uint p2 = x1 * y1; // downshifted by 32
        p1 += p0 >> 16;
        p2 += p1 >> 16;
        return p2;
      }

      inline void philox(uint seed0, uint seed1, uint offset, thread uint* c) {
        c[0] = offset;
        c[1] = 0;
        c[2] = 0;
        c[3] = 0;
        uint k0 = seed0;
        uint k1 = seed1;

        for (int i = 0; i < 10; i++) {
            uint prev_c0 = c[0];
            uint prev_c2 = c[2];
            c[0] = umulhi(PHILOX_ROUND_B, c[2]) ^ c[1] ^ k0;
            c[2] = umulhi(PHILOX_ROUND_A, prev_c0) ^ c[3] ^ k1;
            c[1] = PHILOX_ROUND_B * prev_c2;
            c[3] = PHILOX_ROUND_A * prev_c0;
            k0 = (k0 + PHILOX_KEY_A);
            k1 = (k1 + PHILOX_KEY_B);
        }
      }

      kernel void rand_fp32(device float* output [[buffer(0)]],
                            constant uint &seed0 [[buffer(1)]],
                            constant uint &seed1 [[buffer(2)]],
                            constant uint &offset [[buffer(3)]],
                            constant uint &size [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
          uint c[4];
          philox(seed0, seed1, offset + id, c);
          for (int i = 0; i < 4; i++) {
              uint outIdx = id * 4 + i;
              if (outIdx < size) {
                  output[outIdx] = float(c[i]) / float(0xffffffff);
              }
          }
      }

      kernel void rand_fp16(device half* output [[buffer(0)]],
                            constant uint &seed0 [[buffer(1)]],
                            constant uint &seed1 [[buffer(2)]],
                            constant uint &offset [[buffer(3)]],
                            constant uint &size [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
          uint c[4];
          philox(seed0, seed1, offset + id, c);
          for (int i = 0; i < 4; i++) {
              uint outIdx = id * 4 + i;
              if (outIdx < size) {
                  output[outIdx] = half(float(c[i]) / float(0xffffffff));
              }
          }
      }

      kernel void randn_fp32(device float* output [[buffer(0)]],
                             constant uint &seed0 [[buffer(1)]],
                             constant uint &seed1 [[buffer(2)]],
                             constant uint &offset [[buffer(3)]],
                             constant uint &size [[buffer(4)]],
                             uint id [[thread_position_in_grid]]) {
          uint c[4];
          philox(seed0, seed1, offset + id, c);
          float u1 = float(c[0]) / float(0xffffffff);
          if (u1 < 1e-5) {
              u1 = 1e-5;
          }
          float u2 = float(c[1]) / float(0xffffffff);
          float r = sqrt(-2 * log(u1));
          float phi = 2 * M_PI * u2;
          float z[2];
          z[0] = r * cos(phi);
          z[1] = r * sin(phi);

          for (int i = 0; i < 2; i++) {
              uint outIdx = id * 2 + i;
              if (outIdx < size) {
                  output[outIdx] = z[i];
              }
          }
      }

      kernel void randn_fp16(device half* output [[buffer(0)]],
                             constant uint &seed0 [[buffer(1)]],
                             constant uint &seed1 [[buffer(2)]],
                             constant uint &offset [[buffer(3)]],
                             constant uint &size [[buffer(4)]],
                             uint id [[thread_position_in_grid]]) {
          uint c[4];
          philox(seed0, seed1, offset + id, c);
          float u1 = float(c[0]) / float(0xffffffff);
          if (u1 < 1e-5) {
              u1 = 1e-5;
          }
          float u2 = float(c[1]) / float(0xffffffff);
          float r = sqrt(-2 * log(u1));
          float phi = 2 * M_PI * u2;
          float z[2];
          z[0] = r * cos(phi);
          z[1] = r * sin(phi);

          for (int i = 0; i < 2; i++) {
              uint outIdx = id * 2 + i;
              if (outIdx < size) {
                  output[outIdx] = half(z[i]);
              }
          }
      }

      template <typename T>
      void gather_bcast_impl(device const T* input,
                             device const ulong* indices,
                             device T* output,
                             uint outerCount,
                             uint indexCount,
                             uint middleCount,
                             uint innerCount,
                             uint id) {
          uint innerIdx = id % innerCount;
          uint indexIdx = (id / innerCount) % indexCount;
          uint outerIdx = (id / innerCount) / indexCount;
          if (outerIdx < outerCount) {
              ulong index = indices[indexIdx];
              ulong srcIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
              output[id] = input[srcIndex];
          }
      }

      template <typename T>
      void gather_impl(device const T* input,
                       device const ulong* indices,
                       device T* output,
                       uint outerCount,
                       uint indexCount,
                       uint middleCount,
                       uint innerCount,
                       uint id) {
          uint innerIdx = id % innerCount;
          uint indexIdx = (id / innerCount) % indexCount;
          uint outerIdx = (id / innerCount) / indexCount;
          if (outerIdx < outerCount) {
              ulong index = indices[innerIdx + indexIdx*innerCount + outerIdx*innerCount*indexCount];
              ulong srcIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
              output[id] = input[srcIndex];
          }
      }

      #define DEFINE_GATHER(gather, type) \
      kernel void gather##_##type(device const type* input [[buffer(0)]], \
                                  device const ulong* indices [[buffer(1)]], \
                                  device type* output [[buffer(2)]], \
                                  constant uint &outerCount [[buffer(3)]], \
                                  constant uint &indexCount [[buffer(4)]], \
                                  constant uint &middleCount [[buffer(5)]], \
                                  constant uint &innerCount [[buffer(6)]], \
                                  uint id [[thread_position_in_grid]]) { \
          gather##_impl<type>(input, indices, output, outerCount, indexCount, middleCount, \
                                innerCount, id); \
      }

      DEFINE_GATHER(gather, char)
      DEFINE_GATHER(gather_bcast, char)
      DEFINE_GATHER(gather, short)
      DEFINE_GATHER(gather_bcast, short)
      DEFINE_GATHER(gather, int)
      DEFINE_GATHER(gather_bcast, int)
      DEFINE_GATHER(gather, long)
      DEFINE_GATHER(gather_bcast, long)

      template <typename T>
      void scatter_bcast_impl(device const T* input,
                              device const ulong* indices,
                              device T* output,
                              device atomic_int* counts,
                              device int* needsAddition,
                              uint outerCount,
                              uint indexCount,
                              uint middleCount,
                              uint innerCount,
                              uint id) {
          uint innerIdx = id % innerCount;
          uint indexIdx = (id / innerCount) % indexCount;
          uint outerIdx = (id / innerCount) / indexCount;
          if (outerIdx < outerCount) {
              ulong index = indices[indexIdx];
              ulong dstIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
              output[dstIndex] = input[id];
              if (atomic_fetch_add_explicit(&counts[dstIndex], 1, memory_order_relaxed)) {
                  *needsAddition = 1;
              }
          }
      }

      template <typename T>
      void scatter_impl(device const T* input,
                        device const ulong* indices,
                        device T* output,
                        device atomic_int* counts,
                        device int* needsAddition,
                        uint outerCount,
                        uint indexCount,
                        uint middleCount,
                        uint innerCount,
                        uint id) {
          uint innerIdx = id % innerCount;
          uint indexIdx = (id / innerCount) % indexCount;
          uint outerIdx = (id / innerCount) / indexCount;
          if (outerIdx < outerCount) {
              ulong index = indices[innerIdx + indexIdx*innerCount + outerIdx*innerCount*indexCount];
              ulong dstIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
              output[dstIndex] = input[id];
              if (atomic_fetch_add_explicit(&counts[dstIndex], 1, memory_order_relaxed)) {
                  *needsAddition = 1;
              }
          }
      }

      #define DEFINE_SCATTER(scatter, type) \
      kernel void scatter##_##type(device const type* input [[buffer(0)]], \
                                   device const ulong* indices [[buffer(1)]], \
                                   device type* output [[buffer(2)]], \
                                   device atomic_int* counts [[buffer(3)]], \
                                   device int* needsAddition [[buffer(4)]], \
                                   constant uint &outerCount [[buffer(5)]], \
                                   constant uint &indexCount [[buffer(6)]], \
                                   constant uint &middleCount [[buffer(7)]], \
                                   constant uint &innerCount [[buffer(8)]], \
                                   uint id [[thread_position_in_grid]]) { \
          scatter##_impl<type>(input, indices, output, counts, needsAddition, outerCount, \
                               indexCount, middleCount, innerCount, id); \
      }

      DEFINE_SCATTER(scatter, char)
      DEFINE_SCATTER(scatter_bcast, char)
      DEFINE_SCATTER(scatter, short)
      DEFINE_SCATTER(scatter_bcast, short)
      DEFINE_SCATTER(scatter, int)
      DEFINE_SCATTER(scatter_bcast, int)
      DEFINE_SCATTER(scatter, long)
      DEFINE_SCATTER(scatter_bcast, long)
    """

}

extension Tensor.DType {
  var mpsDType: MPSDataType? {
    switch self {
    case .float32:
      .float32
    case .float16:
      .float16
    default:
      nil
    }
  }
}

func mpsShape(_ x: [Int]) -> [NSNumber] {
  x.map { NSNumber(value: $0) }
}
