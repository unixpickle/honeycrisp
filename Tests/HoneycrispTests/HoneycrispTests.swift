import Gzip
import HCTestUtils
import XCTest

@testable import Honeycrisp

final class HoneycrispTests: XCTestCase {

  var _backends: [Backend] = []
  var backends: [Backend] {
    if _backends.count > 0 {
      return _backends
    } else {
      _backends = [CPUBackend.global, try! MPSBackend()]
      return _backends
    }
  }
  var coremlBackend: CoreMLBackend? = nil

  func runInBackends(coreML: Bool = false, _ fn: () async throws -> Void) async throws {
    let start = Backend.defaultBackend
    defer {
      Backend.defaultBackend = start
    }
    for backend in backends {
      Backend.defaultBackend = backend
      try await fn()
    }
    if coreML {
      Backend.defaultBackend =
        coremlBackend
        ?? {
          coremlBackend = CoreMLBackend(wrapping: backends[0])
          return coremlBackend!
        }()
      try await fn()
    }
  }

  func testCast() async throws {
    try await runInBackends {
      try await BackendTests.testCast()
    }
  }

  func testAdd() async throws {
    try await runInBackends {
      try await BackendTests.testAdd()
    }
  }

  func testSub() async throws {
    try await runInBackends {
      try await BackendTests.testSub()
    }
  }

  func testMul() async throws {
    try await runInBackends {
      try await BackendTests.testMul()
    }
  }

  func testDiv() async throws {
    try await runInBackends {
      try await BackendTests.testDiv()
    }
  }

  func testFusedAddMul() async throws {
    try await runInBackends {
      try await BackendTests.testFusedAddMul()
    }
  }

  func testFusedNormalize() async throws {
    try await runInBackends {
      try await BackendTests.testFusedNormalize()
    }
  }

  func testMod() async throws {
    try await runInBackends {
      try await BackendTests.testMod()
    }
  }

  func testMulGrad() async throws {
    try await runInBackends {
      try await BackendTests.testMulGrad()
    }
  }

  func testMSEGrad() async throws {
    try await runInBackends {
      try await BackendTests.testMSEGrad()
    }
  }

  func testEquals() async throws {
    try await runInBackends {
      try await BackendTests.testEquals()
    }
  }

  func testComparison() async throws {
    try await runInBackends {
      try await BackendTests.testComparison()
    }
  }

  func testBitwise() async throws {
    try await runInBackends {
      try await BackendTests.testBitwise()
    }
  }

  func testSum() async throws {
    try await runInBackends {
      try await BackendTests.testSum()
    }
  }

  func testRepeat() async throws {
    try await runInBackends {
      try await BackendTests.testRepeat()
    }
  }

  func testGather() async throws {
    try await runInBackends {
      try await BackendTests.testGather()
    }
  }

  func testMatrixMatrixProduct() async throws {
    try await runInBackends(coreML: true) {
      try await BackendTests.testMatrixMatrixProduct()
    }
  }

  func testBatchedMatmul() async throws {
    try await runInBackends(coreML: true) {
      try await BackendTests.testBatchedMatmul()
    }
  }

  func testMatrixVectorProduct() async throws {
    try await runInBackends {
      try await BackendTests.testMatrixVectorProduct()
    }
  }

  func testTriangular() async throws {
    try await runInBackends {
      try await BackendTests.testTriangular()
    }
  }

  func testIndexing() async throws {
    try await runInBackends {
      try await BackendTests.testIndexing()
    }
  }

  func testChunk() async throws {
    try await runInBackends {
      try await BackendTests.testChunk()
    }
  }

  func testElemwise() async throws {
    try await runInBackends {
      try await BackendTests.testElemwise()
    }
  }

  func testMinMax() async throws {
    try await runInBackends {
      try await BackendTests.testMinMax()
    }
  }

  func testExpandAndRepeat() async throws {
    try await runInBackends {
      try await BackendTests.testExpandAndRepeat()
    }
  }

  func testBinaryBroadcast() async throws {
    try await runInBackends {
      try await BackendTests.testBinaryBroadcast()
    }
  }

  func testFusedBroadcast() async throws {
    try await runInBackends {
      try await BackendTests.testFusedBroadcast()
    }
  }

  func testSoftmax() async throws {
    try await runInBackends {
      try await BackendTests.testSoftmax()
    }
  }

  func testConcatInner() async throws {
    try await runInBackends {
      try await BackendTests.testConcatInner()
    }
  }

  func testConcatOuter() async throws {
    try await runInBackends {
      try await BackendTests.testConcatOuter()
    }
  }

  func testWhen() async throws {
    try await runInBackends {
      try await BackendTests.testWhen()
    }
  }

  func testOneHot() async throws {
    try await runInBackends {
      try await BackendTests.testOneHot()
    }
  }

  func testAdam() async throws {
    try await runInBackends {
      try await BackendTests.testAdam()
    }
  }

  func testRandom() async throws {
    try await runInBackends {
      try await BackendTests.testRandom()
    }
  }

  func testConv2D() async throws {
    try await runInBackends {
      try await BackendTests.testConv2D()
    }
  }

  func testConv2DTransposeGrads() async throws {
    try await runInBackends {
      try await BackendTests.testConv2DTransposeGrads()
    }
  }

  func testConv1D() async throws {
    try await runInBackends {
      try await BackendTests.testConv1D()
    }
  }

  func testGroupNorm() async throws {
    try await runInBackends {
      try await BackendTests.testGroupNorm()
    }
  }

  func testReshape() async throws {
    let x = Tensor(ones: [3, 5, 8])
    XCTAssertEqual(x.reshape([5, 3, 8]).shape, [5, 3, 8])
    XCTAssertEqual(x.reshape([-1, 3, 8]).shape, [5, 3, 8])
    XCTAssertEqual(x.reshape([5, -1, 8]).shape, [5, 3, 8])
    XCTAssertEqual(x.reshape([5, 3, -1]).shape, [5, 3, 8])
    XCTAssertEqual(x.reshape([5 * 3, -1]).shape, [5 * 3, 8])
    let y = Tensor(ones: [3, 0, 15])
    XCTAssertEqual(y.reshape([5 * 3, -1]).shape, [5 * 3, 0])
    XCTAssertEqual(y.reshape([3, 15, -1]).shape, [3, 15, 0])
  }

  func testStack() async throws {
    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
    let y = Tensor(data: [7, 8, 9, 10, 11, 12], shape: [2, 3])
    let stack0 = Tensor(stack: [x, y], axis: 0)
    let stack1 = Tensor(stack: [x, y], axis: 1)
    let stack2 = Tensor(stack: [x, y], axis: 2)
    XCTAssertEqual(stack0.shape, [2, 2, 3])
    XCTAssertEqual(stack1.shape, [2, 2, 3])
    XCTAssertEqual(stack2.shape, [2, 3, 2])
    try await assertDataEqual(stack0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    try await assertDataEqual(stack1, [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12])
    try await assertDataEqual(stack2, [1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12])
    XCTAssertEqual(stack0.shape, Tensor(stack: [x, y], axis: -3).shape)
    try await assertDataEqual(stack0, Tensor(stack: [x, y], axis: -3))
    XCTAssertEqual(stack1.shape, Tensor(stack: [x, y], axis: -2).shape)
    try await assertDataEqual(stack1, Tensor(stack: [x, y], axis: -2))
    XCTAssertEqual(stack2.shape, Tensor(stack: [x, y], axis: -1).shape)
    try await assertDataEqual(stack2, Tensor(stack: [x, y], axis: -1))
  }

  func testTensorState() async throws {
    for t1 in [
      Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [4], dtype: .float32),
      Tensor(data: [1, 2, 3, 4], shape: [4], dtype: .int64),
      Tensor(data: [false, true, false, false], shape: [4], dtype: .bool),
      Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [4], dtype: .float16),
    ] {
      let t1State = try await t1.state()
      let encoder = PropertyListEncoder()
      let data = try encoder.encode(t1State)
      let decoded = try PropertyListDecoder().decode(TensorState.self, from: data)
      let t1Copy = Tensor(state: decoded)
      XCTAssertEqual(t1.shape, t1Copy.shape)
      XCTAssertEqual(t1.dtype, t1Copy.dtype)
      try await assertDataEqual(t1, t1Copy)
    }
  }

  func testTrainable() async throws {
    class Linear: Trainable {
      @Param(name: "weight") var weight: Tensor
      @Param(name: "bias") var bias: Tensor
      @Param(name: "optional") var optional: Tensor?
      @Buf(name: "someBuf") var someBuf: Tensor

      init(inSize: Int, outSize: Int) {
        super.init()
        weight = Tensor(rand: [inSize, outSize])
        bias = Tensor(rand: [outSize])
        someBuf = Tensor(rand: [1])
      }
    }

    class Network: Trainable {
      @Child(name: "layer0") var layer0: Linear
      @Child(name: "layer1") var layer1: Linear
      @Child(name: "layer2") var layer2: Linear?

      override init() {
        super.init()
        layer0 = Linear(inSize: 3, outSize: 5)
        layer1 = Linear(inSize: 5, outSize: 7)
      }
    }

    let instance = Linear(inSize: 3, outSize: 5)
    for _ in 0..<2 {
      var params = instance.parameters
      XCTAssertEqual(params.count, 2)
      XCTAssertEqual(params[0].0, "bias")
      XCTAssertEqual(params[0].1.data!.shape, [5])
      XCTAssertEqual(params[1].0, "weight")
      XCTAssertEqual(params[1].1.data!.shape, [3, 5])
      XCTAssert(instance.$bias.data != nil)
      XCTAssert(instance.$weight.data != nil)
      XCTAssertEqual(instance.bias.shape, [5])
      XCTAssertEqual(instance.weight.shape, [3, 5])

      let buffers = instance.buffers
      XCTAssertEqual(buffers.count, 1)
      XCTAssertEqual(buffers[0].1.data!.shape, [1])

      instance.optional = Tensor(zeros: [7])
      params = instance.parameters
      XCTAssertEqual(params.count, 3)
      XCTAssertEqual(params[1].0, "optional")
      XCTAssertEqual(params[1].1.data!.shape, [7])
      instance.optional = nil
    }

    let net = Network()
    for _ in 0..<2 {
      var netParams = net.parameters
      XCTAssertEqual(netParams.count, 4)
      XCTAssertEqual(netParams[0].0, "layer0.bias")
      XCTAssertEqual(netParams[0].1.data!.shape, [5])
      XCTAssertEqual(netParams[1].0, "layer0.weight")
      XCTAssertEqual(netParams[1].1.data!.shape, [3, 5])
      XCTAssertEqual(netParams[2].0, "layer1.bias")
      XCTAssertEqual(netParams[2].1.data!.shape, [7])
      XCTAssertEqual(netParams[3].0, "layer1.weight")
      XCTAssertEqual(netParams[3].1.data!.shape, [5, 7])

      net.layer2 = Linear(inSize: 3, outSize: 3)
      netParams = net.parameters
      XCTAssertEqual(netParams.count, 6)
      XCTAssertEqual(netParams[4].0, "layer2.bias")
      XCTAssertEqual(netParams[4].1.data!.shape, [3])
      XCTAssertEqual(netParams[5].0, "layer2.weight")
      XCTAssertEqual(netParams[5].1.data!.shape, [3, 3])
      net.layer2 = nil
    }

    let net1 = Network()
    try net1.loadState(try await net.state())
    for ((name1, param1), (name2, param2)) in zip(
      net.buffersAndParameters, net1.buffersAndParameters)
    {
      XCTAssertEqual(param1.data!.dtype, param2.data!.dtype)
      XCTAssertEqual(param1.data!.shape, param2.data!.shape)
      try await assertClose(param1.data!, param2.data!, "params differ for names \(name1)/\(name2)")
    }
  }

  func testConv1DTrainable() async throws {
    let cFirst = Conv1D(inChannels: 3, outChannels: 5, kernelSize: 3, stride: 2)
    let inFirst = Tensor(zeros: [2, 3, 16])
    let outFirst = cFirst(inFirst)
    XCTAssertEqual(outFirst.shape, [2, 5, 16 / 2 - 1])

    let cLast = Conv1D(inChannels: 3, outChannels: 5, kernelSize: 3, stride: 2, channelsLast: true)
    let inLast = Tensor(zeros: [2, 16, 3])
    let outLast = cLast(inLast)
    XCTAssertEqual(outLast.shape, [2, 16 / 2 - 1, 5])

    let cLastSame = Conv1D(
      inChannels: 3, outChannels: 5, kernelSize: 3, padding: .same, channelsLast: true)
    let outLastSame = cLastSame(inLast)
    XCTAssertEqual(outLastSame.shape, [2, 16, 5])
  }

  func testNoGrad() async throws {
    let x = Tensor(ones: [3]).onGradUnsafe { grad in assert(false) }
    XCTAssert(!Tensor.withGrad(enabled: false) { x.sum() }.needsGrad)
    Tensor.withGrad(enabled: false) {
      XCTAssert(!Tensor.isGradEnabled)
      XCTAssert(Tensor.withGrad(enabled: true) { Tensor.isGradEnabled })
      XCTAssert(Tensor.withGrad(enabled: true) { x.sum() }.needsGrad)
      XCTAssert(!x.sum().needsGrad)
    }
    XCTAssert(x.sum().needsGrad)
    XCTAssert(!Tensor.withGrad(enabled: false) { x * x }.needsGrad)
    XCTAssert((x * x).needsGrad)
  }

  func testCheckpointSimple() async throws {
    let xData = Tensor(data: [1.0, 2.0, 3.0])
    var xGrad: Tensor?
    let x = xData.onGradUnsafe { g in xGrad = g }

    let outputs = Tensor.checkpoint([x]) { xs in
      [xs[0].pow(2).sum(), xs[0].pow(3).sum()]
    }
    (outputs[0] + outputs[1]).backward()

    let expectedGrad = 2 * x + 3 * x.pow(2)
    try await assertDataEqual(xGrad!, expectedGrad)
  }

  func testCheckpointRepeatedOutput() async throws {
    let xData = Tensor(data: [1.0, 2.0, 3.0])
    var xGrad: Tensor?
    let x = xData.onGradUnsafe { g in xGrad = g }

    let outputs = Tensor.checkpoint([x]) { xs in
      let out1 = xs[0].pow(2).sum()
      return [out1, out1]
    }
    (outputs[0] + outputs[1]).backward()

    let expectedGrad = 4 * x
    try await assertDataEqual(xGrad!, expectedGrad)
  }

  func testCheckpointUnusedOutput() async throws {
    let xData = Tensor(data: [1.0, 2.0, 3.0])
    var xGrad: Tensor?
    let x = xData.onGradUnsafe { g in xGrad = g }

    let out = Tensor.checkpoint([x]) { xs in
      let out1 = xs[0].pow(2).sum()
      let out2 = out1 * 2
      return [out1, out2, out1]
    }[0]
    out.backward()

    let expectedGrad = 2 * x
    try await assertDataEqual(xGrad!, expectedGrad)
  }

  func testCheckpointNoGradInOut() async throws {
    let xData = Tensor(data: [1.0, 2.0, 3.0])
    var xGrad: Tensor?
    let x = xData.onGradUnsafe { g in xGrad = g }

    let noGrad1 = Tensor(data: [1.0, 2.0, 3.0])
    let noGrad2 = Tensor(data: [1, 2, 3])

    let out = Tensor.checkpoint([x, noGrad1, noGrad2]) { xs in
      let out1 = xs[0].pow(2).sum()
      let out2 = out1 * 2
      return [out1, out2.noGrad(), xs[1], xs[2], noGrad1, noGrad2]
    }[0]
    out.backward()

    let expectedGrad = 2 * x
    try await assertDataEqual(xGrad!, expectedGrad)
  }

  func testCheckpointNested() async throws {
    let xData = Tensor(data: [1.0, 2.0, 3.0])
    var xGrad: Tensor?
    let x = xData.onGradUnsafe { g in xGrad = g }

    let out = Tensor.checkpoint([x]) { xs in
      let out1 = xs[0].pow(2)
      return Tensor.checkpoint([xs[0], out1]) { ys in
        [ys[0] * ys[1] + 1]
      }
    }[0]
    out.backward()

    let expectedGrad = 3 * x.pow(2)
    try await assertDataEqual(xGrad!, expectedGrad)
  }

  func testCheckpointRandom() async throws {
    let xData = Tensor(ones: [100])
    var xGrad: Tensor?
    let x = xData.onGradUnsafe { g in xGrad = g }

    let outputs = Tensor.checkpoint([x]) { xs in
      [(xs[0] * Tensor(randnLike: xs[0]))]
    }
    outputs[0].sum().backward()

    try await assertDataEqual(xGrad!, outputs[0])
  }

  func testCheckpointTrainable() async throws {
    let layer = Linear(inCount: 4, outCount: 5)
    let xData = Tensor(randn: [8, 4])
    let target = Tensor(randn: [8, 5])
    var xGrad: Tensor?
    var x = xData.onGradUnsafe { g in xGrad = g }

    let expLoss = (layer(x) - target).pow(2).sum()
    expLoss.backward()
    let expXGrad = xGrad!
    xGrad = nil
    let expGrads = layer.parameters.map { $0.1.grad! }
    for (_, var p) in layer.parameters {
      p.grad = nil
    }

    x = xData.onGradUnsafe { g in xGrad = g }
    let actLoss = Tensor.checkpoint([x, target]) { xs in
      [(layer(xs[0]) - xs[1]).pow(2).sum()]
    }[0]
    actLoss.backward()
    try await assertClose(expLoss, actLoss)
    try await assertClose(expXGrad, xGrad!)

    for ((_, p), expGrad) in zip(layer.parameters, expGrads) {
      try await assertClose(expGrad, p.grad!)
    }
  }

  func testTriangularGrad() async throws {
    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [3, 2], dtype: .float32)
    var xGrad: Tensor?
    x.onGradUnsafe { g in xGrad = g }.tril().backward(
      Tensor(data: [-1, -2, -3, -4, -5, -6], shape: [3, 2], dtype: .float32))
    try await assertDataEqual(xGrad!, [-1, 0, -3, -4, -5, -6])
  }

  func testCacheBackendForBackward() async throws {
    // If any operations are called on this backend, it will raise an error.
    let unusedBackend = Backend()

    func testArithmetic() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [1]).onGradUnsafe { grad in xGrad = grad }
        let y1 = x / 3.0
        let y2 = 3.0 / x
        let y3 = y1 / y2
        let y4 = y1 + y3
        let y5 = 1.0 + y4
        let y6 = y5 + 1.0
        let y7 = y6 - 1.0
        let y8 = 1.0 - y7
        let y9 = y8 - x
        let y10 = y9 * x
        let y11 = y10 * 2.0
        let y12 = 2.0 * y11
        let outGrad = Tensor(onesLike: y12)
        unusedBackend.use { y12.backward(outGrad) }
        XCTAssert(xGrad != nil)
        let _ = try await xGrad!.data
      }
    }

    func testMatrixMul() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [3, 3]).onGradUnsafe { grad in xGrad = grad }
        let y = (x &* x) + (x &* x.t())
        let outGrad = Tensor(onesLike: y)
        unusedBackend.use { y.backward(outGrad) }
        XCTAssert(xGrad != nil)
        let _ = try await xGrad!.data
      }
    }

    func testTril() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [3, 3]).onGradUnsafe { grad in xGrad = grad }
        let y = x.tril()
        let outGrad = Tensor(onesLike: y)
        unusedBackend.use { y.backward(outGrad) }
        XCTAssert(xGrad != nil)
        let _ = try await xGrad!.data
      }
    }

    func testReduceRepeat() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [3, 3]).onGradUnsafe { grad in xGrad = grad }
        let y = x + x.sum(axis: 1).reshape([3, 1]).repeating(axis: 1, count: 3)
        let outGrad = Tensor(onesLike: y)
        unusedBackend.use { y.backward(outGrad) }
        XCTAssert(xGrad != nil)
        let _ = try await xGrad!.data
      }
    }

    func testConcatSplit() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [3, 6]).onGradUnsafe { grad in xGrad = grad }
        let ys = x.split(axis: 1, counts: [2, 2, 2])
        let y = Tensor(concat: ys, axis: 1)
        let outGrad = Tensor(onesLike: y)
        unusedBackend.use { y.backward(outGrad) }
        XCTAssert(xGrad != nil)
        let _ = try await xGrad!.data
      }
    }

    func testElemwise() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [3, 6]).onGradUnsafe { grad in xGrad = grad }
        let y = x.gelu().pow(2)
        let outGrad = Tensor(onesLike: y)
        unusedBackend.use { y.backward(outGrad) }
        XCTAssert(xGrad != nil)
        let _ = try await xGrad!.data
      }
    }

    func testScatterGather() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [3, 6]).onGradUnsafe { grad in xGrad = grad }
        let inds = Tensor(data: [1, 0, 5], shape: [3])
        let y = x.gather(axis: 1, indices: inds).scatter(axis: 1, count: 6, indices: inds)
        let outGrad = Tensor(onesLike: y)
        unusedBackend.use { y.backward(outGrad) }
        XCTAssert(xGrad != nil)
        let _ = try await xGrad!.data
      }
    }

    try await testArithmetic()
    try await testMatrixMul()
    try await testTril()
    try await testReduceRepeat()
    try await testConcatSplit()
    try await testElemwise()
    try await testScatterGather()
  }

  func testConcurrentBackward() async throws {
    let x = Tensor(ones: [128, 16])
    let param: Trainable.Param<Tensor> = .init(name: "foo")
    param.data = x
    let xWithGrad = x.onGrad { g in param.addGrad(g) }
    let lastResult = xWithGrad * 0

    var tasks: [Task<(), Error>] = []
    for _ in 0..<100 {
      tasks.append(
        Task.detached {
          let y = xWithGrad &* Tensor(ones: [16, 1])
          y.sum().backward()
        })
    }
    for task in tasks {
      let _ = try await task.value
    }

    // We want to make sure backward doesn't complete before all
    // the other tasks complete.
    lastResult.backward()

    try await assertDataEqual(param.grad!, Tensor(onesLike: x) * 100)
  }

}
