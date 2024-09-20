import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

open class MPSBackend: CPUBackend {

  private struct MatmulKey: Hashable {
    let transA: Bool
    let transB: Bool
    let rows: Int
    let cols: Int
    let inner: Int
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
      assert(count <= 0xffff_ffff, "count exceeds UInt32 size: \(count)")

      let functionName =
        "\(dist == .normal ? "randn" : "rand")_\(dtype == .float16 ? "fp16" : "fp32")"
      let output = try await mpsBackend.allocate(length: count * dtype.byteSize)

      return try await mpsBackend.serialize {
        let device = try mpsBackend.device
        let completion = try mpsBackend.completionBuffer { buf in
          guard let function = mpsBackend.library!.makeFunction(name: functionName) else {
            throw BackendError.notImplemented(
              "function \(functionName) not found in shader library")
          }
          let state = try device.makeComputePipelineState(function: function)
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
          let groupSize = min(
            state.maxTotalThreadsPerThreadgroup, mpsBackend.nextPowerOf2(totalThreads, min: 32))
          let gridSize = MTLSize(
            width: mpsBackend.nextMultiple(totalThreads, divisor: groupSize), height: 1, depth: 1)
          let threadGroupSize = MTLSize(width: groupSize, height: 1, depth: 1)
          computeEncoder.setComputePipelineState(state)
          computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
          computeEncoder.endEncoding()
          offset += UInt64(totalThreads)
        }
        return Tensor.Data(backend: mpsBackend, buffer: output, completeOnAllDevices: completion)
      }
    }
  }

  private var queue = DispatchQueue(label: "mps-backend-worker")
  private var commandQueue: MTLCommandQueue? = nil
  private var matmuls: [MatmulKey: MPSMatrixMultiplication] = [:]
  private var defaultRNG: MPSRandomGenerator? = nil
  private var library: MTLLibrary? = nil

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
    library = try (self._device!).makeLibrary(
      source: MPSBackend.KernelCode, options: MTLCompileOptions())
  }

  internal func waitForGPUData(_ xs: Tensor.Data...) async throws {
    for x in xs {
      if x.backend === self && x.completeOnAllDevices != nil {
        continue
      }
      try await waitForData(x)
    }
  }

  override public func elemwise(_ a: Tensor.Data, op: ElemwiseOp, count: Int, dtype: Tensor.DType)
    async throws
    -> Tensor.Data
  {
    if dtype != .float16 && dtype != .float32 {
      return try await super.elemwise(a, op: op, count: count, dtype: dtype)
    }

    assert(count <= Int(UInt32.max), "cannot apply kernel to this many values")

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
    return try await serialize {
      let device = try device
      let completion = try completionBuffer { buf in
        guard let function = library!.makeFunction(name: functionName) else {
          throw BackendError.notImplemented("function \(functionName) not found in shader library")
        }
        let state = try device.makeComputePipelineState(function: function)
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        var n = UInt32(count)
        let nBuffer = device.makeBuffer(bytes: &n, length: MemoryLayout<UInt32>.stride, options: [])
        computeEncoder.setBuffer(a.buffer, offset: 0, index: 0)
        computeEncoder.setBuffer(output, offset: 0, index: 1)
        computeEncoder.setBuffer(nBuffer, offset: 0, index: 2)

        let groupSize = min(state.maxTotalThreadsPerThreadgroup, nextPowerOf2(count, min: 32))
        let gridSize = MTLSize(width: nextMultiple(count, divisor: groupSize), height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: groupSize, height: 1, depth: 1)
        computeEncoder.setComputePipelineState(state)
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
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
    assert(
      outBytes <= Int(UInt32.max),
      "cannot apply kernel to this many values")

    let output = try await allocate(length: outBytes * dtype.byteSize)

    try await waitForGPUData(a)
    return try await serialize {
      let device = try device
      let completion = try completionBuffer { buf in
        guard let function = library!.makeFunction(name: "repeat") else {
          throw BackendError.notImplemented("function 'repeat' not found in shader library")
        }
        let state = try device.makeComputePipelineState(function: function)
        guard let computeEncoder = buf.makeComputeCommandEncoder() else {
          throw BackendError.kernelFailed("could not create compute encoder")
        }
        var inner = UInt32(innerCount * dtype.byteSize)
        var outer = UInt32(outerCount)
        var reps = UInt32(repeats)
        computeEncoder.setBuffer(a.buffer, offset: 0, index: 0)
        computeEncoder.setBuffer(output, offset: 0, index: 1)
        computeEncoder.setBuffer(
          device.makeBuffer(bytes: &inner, length: MemoryLayout<UInt32>.stride, options: []),
          offset: 0, index: 2)
        computeEncoder.setBuffer(
          device.makeBuffer(bytes: &outer, length: MemoryLayout<UInt32>.stride, options: []),
          offset: 0, index: 3)
        computeEncoder.setBuffer(
          device.makeBuffer(bytes: &reps, length: MemoryLayout<UInt32>.stride, options: []),
          offset: 0, index: 4)

        let groupSize = min(state.maxTotalThreadsPerThreadgroup, nextPowerOf2(outBytes, min: 32))
        let gridSize = MTLSize(
          width: nextMultiple(outBytes, divisor: groupSize), height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: groupSize, height: 1, depth: 1)
        computeEncoder.setComputePipelineState(state)
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }
  }

  override public func matmul(
    a: Tensor.Data, transA: Bool, b: Tensor.Data, transB: Bool, transOut: Bool, rows: Int,
    inner: Int, cols: Int, dtype: Tensor.DType
  )
    async throws
    -> Tensor.Data
  {
    guard let mpsDType = dtype.mpsDType else {
      return try await super.matmul(
        a: a, transA: transA, b: b, transB: transB, transOut: transOut, rows: rows, inner: inner,
        cols: cols, dtype: dtype)
    }

    if transOut {
      return try await matmul(
        a: b, transA: !transB, b: a, transB: !transA, transOut: false, rows: cols, inner: inner,
        cols: rows, dtype: dtype)
    }

    let aShape = transA ? (inner, rows) : (rows, inner)
    let bShape = transB ? (cols, inner) : (inner, cols)

    let output = try await allocate(length: rows * cols * dtype.byteSize)
    try await waitForGPUData(a, b)
    return try await serialize {
      let mm = self.createMatmul(
        transA: transA, transB: transB, rows: rows, inner: inner, cols: cols)
      let completion = completionBuffer { buf in
        mm.encode(
          commandBuffer: buf,
          leftMatrix: MPSMatrix(
            buffer: a.buffer,
            descriptor: MPSMatrixDescriptor(
              rows: aShape.0, columns: aShape.1, rowBytes: aShape.1 * dtype.byteSize,
              dataType: mpsDType)),
          rightMatrix: MPSMatrix(
            buffer: b.buffer,
            descriptor: MPSMatrixDescriptor(
              rows: bShape.0, columns: bShape.1, rowBytes: bShape.1 * dtype.byteSize,
              dataType: mpsDType)),
          resultMatrix: MPSMatrix(
            buffer: output,
            descriptor: MPSMatrixDescriptor(
              rows: rows, columns: cols, rowBytes: cols * dtype.byteSize, dataType: mpsDType))
        )
      }
      return Tensor.Data(backend: self, buffer: output, completeOnAllDevices: completion)
    }
  }

  private func createMatmul(transA: Bool, transB: Bool, rows: Int, inner: Int, cols: Int)
    -> MPSMatrixMultiplication
  {
    let key = MatmulKey(transA: transA, transB: transB, rows: rows, cols: cols, inner: inner)
    if let matmul = matmuls[key] {
      return matmul
    } else {
      let mm = MPSMatrixMultiplication(
        device: try! device, transposeLeft: transA, transposeRight: transB, resultRows: rows,
        resultColumns: cols, interiorColumns: inner, alpha: 1, beta: 0)
      matmuls[key] = mm
      return mm
    }
  }

  override public func defaultRandom() async throws -> RandomGenerator {
    try await serialize {
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

  internal func completionBuffer(_ action: (MTLCommandBuffer) throws -> Void) rethrows -> Task<
    (), Error
  > {
    let buf = commandQueue!.makeCommandBuffer()!
    try action(buf)
    return Task {
      return try await withCheckedThrowingContinuation { continuation in
        buf.addCompletedHandler { _ in
          if let e = buf.error {
            continuation.resume(throwing: e)
          } else {
            continuation.resume(returning: ())
          }
        }
        // I'm not sure we need to commit() from the main queue,
        // but let's do it to be safe.
        Task {
          try await serialize { buf.commit() }
        }
      }
    }
  }

  private static let KernelCode = """
      #include <metal_stdlib>
      using namespace metal;

      inline float safeTanh(float x) {
          return (x < -10 ? -1 : (x > 10 ? 1 : tanh(x)));
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
            output[id] = (safeTanh(input[id] / 2) + 1) / 2;
        }
      }

      kernel void sigmoid_fp32(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint &N [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
        if (id < N) {
            output[id] = (safeTanh(input[id] / 2) + 1) / 2;
        }
      }

      kernel void sigmoid_grad_fp16(device const half* input [[buffer(0)]],
                                    device half* output [[buffer(1)]],
                                    constant uint &N [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id < N) {
            half s = (safeTanh(input[id] / 2) + 1) / 2;
            output[id] = s * (1 - s);
        }
      }

      kernel void sigmoid_grad_fp32(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint &N [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
        if (id < N) {
            half s = (safeTanh(input[id] / 2) + 1) / 2;
            output[id] = s * (1 - s);
        }
      }

      kernel void gelu_fp32(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint &N [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = input[id];
              output[id] = 0.5 * x * (1.0 + safeTanh(0.797884561 * (x + 0.044715 * pow(x, 3.0))));
          }
      }

      kernel void gelu_fp16(device const half* input [[buffer(0)]],
                            device half* output [[buffer(1)]],
                            constant uint &N [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = float(input[id]);
              output[id] = half(0.5 * x * (1.0 + safeTanh(0.797884561 * (x + 0.044715 * pow(x, 3.0)))));
          }
      }

      kernel void gelu_grad_fp32(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint &N [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
          if (id < N) {
              float x = input[id];
              float tanhTerm = safeTanh(0.035677408145115 * pow(x, 3.0) + 0.797884561 * x);
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
              float tanhTerm = safeTanh(0.035677408145115 * pow(x, 3.0) + 0.797884561 * x);
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
