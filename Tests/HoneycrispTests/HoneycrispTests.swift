import XCTest

@testable import Honeycrisp

final class HoneycrispTests: XCTestCase {
  func testCast() async throws {
    let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
    var f = try await x.floats()
    XCTAssertEqual(f, [1.0, 2.0, 0.0])
    let y = x.cast(.float16)
    XCTAssertEqual(y.dtype, .float16)
    f = try await y.floats()
    XCTAssertEqual(f, [1.0, 2.0, 0.0])
    let z = x.cast(.int64)
    f = try await z.floats()
    XCTAssertEqual(f, [1.0, 2.0, 0.0])
    let w = x.cast(.bool)
    f = try await w.floats()
    XCTAssertEqual(f, [1.0, 1.0, 0.0])
  }

  func testAdd() async throws {
    let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
    let y = Tensor(data: [-1.0, 2.0, -3.0], shape: [3], dtype: .float32)
    var f = try await (x + y).floats()
    XCTAssertEqual(f, [0.0, 4.0, -3.0])

    let z = x + 3
    f = try await z.floats()
    XCTAssertEqual(f, [4.0, 5.0, 3.0])

    let w = x + 1.5
    f = try await w.floats()
    XCTAssertEqual(f, [2.5, 3.5, 1.5])
  }
  // func testMSEGrad() throws {
  //   let x = Tensor(data: [1.0, 2.0, 3.0], shape: [3])
  //   let y = Tensor(data: [2.0, 0.0, -3.0], shape: [3])
  //   var xGrad: Tensor?
  //   let diff = x.onGrad({ grad in xGrad = grad }) - y
  //   let sqDiff = diff * diff
  //   sqDiff.backward(grad: Tensor(onesLike: x))
  //   XCTAssertEqual(xGrad!.data, [-2, 4, 12])
  // }

  // func testMatrixMatrixProduct() throws {
  //   let x = Tensor(ones: [64, 128])
  //   let y = Tensor(ones: [128, 32])
  //   let z = x &* y
  //   XCTAssertEqual(z.shape, [64, 32])
  //   XCTAssert(z.data.map({ $0 == 128 }).reduce(true, { x, y in x && y }))
  // }

  // func testMatrixVectorProduct() throws {
  //   let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
  //   let y = Tensor(data: [-1, -3, 2], shape: [3, 1])
  //   var xGrad: Tensor?
  //   var yGrad: Tensor?
  //   let xParam = x.onGrad { grad in xGrad = grad }
  //   let yParam = y.onGrad { grad in yGrad = grad }
  //   let product = xParam &* yParam
  //   XCTAssertEqual(product.data, [-1, -7])
  //   let outGrad = Tensor(onesLike: product)
  //   product.backward(grad: outGrad)
  //   XCTAssertEqual(xGrad!.data, [-1, -3, 2, -1, -3, 2])
  //   XCTAssertEqual(xGrad!.shape, x.shape)
  //   XCTAssertEqual(yGrad!.data, [5, 7, 9])
  //   XCTAssertEqual(yGrad!.shape, y.shape)
  // }

  // func testSlice() throws {
  //   let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [3, 2])
  //   XCTAssertEqual(x[1].data, [3, 4])
  //   XCTAssertEqual(x[1..<3].data, [3, 4, 5, 6])
  //   XCTAssertEqual(x[1...2].data, [3, 4, 5, 6])
  //   XCTAssertEqual(x[1..<2].data, [3, 4])
  //   XCTAssertEqual(x[(-2)..<(-1)].data, [3, 4])
  //   XCTAssertEqual(x[(-2)...(-1)].data, [3, 4, 5, 6])
  //   XCTAssertEqual(x[(-2)...].data, [3, 4, 5, 6])
  //   XCTAssertEqual(x[...(-2)].data, [1, 2, 3, 4])
  //   XCTAssertEqual(x[..<(-2)].data, [1, 2])

  //   let y = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [3, 1, 4])
  //   XCTAssertEqual(y[..., 0].shape, [3, 4])
  //   XCTAssertEqual(y[..., 0].data, y.data)
  //   XCTAssertEqual(y[0, 0, 0].shape, [])
  //   XCTAssertEqual(y[0, 0, 0].data, [1])
  //   XCTAssertEqual(y[0...1, ..., 3].data, [4, 8])
  //   XCTAssertEqual(y[0..<2, ..., 3].data, [4, 8])
  //   XCTAssertEqual(y[0...2, ..., 3].data, [4, 8, 12])
  //   XCTAssertEqual(y[0..<3, ..., 3].data, [4, 8, 12])
  //   XCTAssertEqual(y[0..., ..., 3].data, [4, 8, 12])
  //   XCTAssertEqual(y[1...2, ..., 2...3].data, [7, 8, 11, 12])
  //   XCTAssertEqual(y[0...2, ..., 2...3].data, [3, 4, 7, 8, 11, 12])
  //   XCTAssertEqual(y[0...2, 0..<1, 2...3].data, [3, 4, 7, 8, 11, 12])
  //   XCTAssertEqual(y[FullRange(dims: 3), NewAxis()].data, y.data)
  //   XCTAssertEqual(y[FullRange(dims: 3), NewAxis()].shape, [3, 1, 4, 1])
  //   XCTAssertEqual(y[FullRange(dims: 2), NewAxis()].data, y.data)
  //   XCTAssertEqual(y[FullRange(dims: 2), NewAxis()].shape, [3, 1, 1, 4])
  //   XCTAssertEqual(y[FullRange(dims: 1), NewAxis()].data, y.data)
  //   XCTAssertEqual(y[FullRange(dims: 1), NewAxis()].shape, [3, 1, 1, 4])
  //   XCTAssertEqual(y[NewAxis()].data, y.data)
  //   XCTAssertEqual(y[NewAxis()].shape, [1, 3, 1, 4])

  //   var yGrad: Tensor?
  //   let yParam = y.onGrad { grad in yGrad = grad }
  //   yParam[1...2, ..., 2...3].backward(grad: Tensor(data: [1, 2, 3, 4], shape: [2, 1, 2]))
  //   XCTAssertEqual(yGrad!.shape, y.shape)
  //   XCTAssertEqual(yGrad!.data, [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4])

  //   let yParam1 = y.onGrad { grad in yGrad = grad }
  //   yParam1[..., 0, 3].backward(grad: Tensor(data: [1, 2, 3], shape: [3]))
  //   XCTAssertEqual(yGrad!.shape, y.shape)
  //   XCTAssertEqual(yGrad!.data, [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3])
  // }

  // func testEquality() throws {
  //   let t1 = Tensor(data: [1, 2, 3, 4, 3, 1, 2, 0], shape: [2, 4])
  //   let t2 = Tensor(data: [1, 0, 4, 4, 0, 0, 1, 0], shape: [2, 4])
  //   XCTAssertEqual((t1 == 2).data, [0, 1, 0, 0, 0, 0, 1, 0])
  //   XCTAssertEqual((t1 == t2).data, [1, 0, 0, 1, 0, 0, 0, 1])
  // }

  // func testElemwise() throws {
  //   func testF(input: [Float], output: [Float], grad: [Float], _ op: (Tensor) -> Tensor) throws {
  //     var actualGrad: Tensor?
  //     let tensorIn = Tensor(data: input, shape: [input.count]) { g in
  //       actualGrad = g
  //     }
  //     let actualOut = op(tensorIn)
  //     try assertClose(actualOut, Tensor(data: output, shape: [output.count]))
  //     actualOut.backward(grad: Tensor(onesLike: actualOut))
  //     try assertClose(actualGrad!, Tensor(data: grad, shape: [output.count]))
  //   }

  //   try testF(
  //     input: [-1, -2, -3, 1, 2, 3],
  //     output: [1, 4, 9, 1, 4, 9],
  //     grad: [-2, -4, -6, 2, 4, 6]
  //   ) { $0.pow(2) }
  //   try testF(
  //     input: [0.01, 1, 2, 3],
  //     output: [0.1, 1.0, sqrt(2.0), sqrt(3.0)],
  //     grad: [5.0, 0.5, 0.3535533845424652, 0.28867512941360474]
  //   ) { $0.sqrt() }
  //   try testF(
  //     input: [-100, -2, 0, 1, 2, 3, 100],
  //     output: [
  //       -0.0, -0.04540228843688965, 0.0, 0.8411920070648193, 1.9545977115631104, 2.9963626861572266,
  //       100.0,
  //     ],
  //     grad: [
  //       0.0, -0.08609922230243683, 0.5, 1.0829640626907349, 1.0860992670059204, 1.0115842819213867,
  //       1.0,
  //     ]
  //   ) { $0.gelu() }
  // }

  // func testSum() throws {
  //   let input = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3])
  //   var gradA: Tensor?
  //   let sumA = input.onGrad({ g in gradA = g }).sum(axis: 1)
  //   XCTAssertEqual(sumA.data, [6, 15, 24])
  //   sumA.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
  //   XCTAssertEqual(gradA!.data, [-1, -1, -1, -2, -2, -2, -3, -3, -3])

  //   var gradB: Tensor?
  //   let sumB = input.onGrad({ g in gradB = g }).sum(axis: 0)
  //   XCTAssertEqual(sumB.data, [12, 15, 18])
  //   sumB.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
  //   XCTAssertEqual(gradB!.data, [-1, -2, -3, -1, -2, -3, -1, -2, -3])

  //   let input1 = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8], shape: [2, 2, 2])

  //   var gradC: Tensor?
  //   let sumC = input1.onGrad({ g in gradC = g }).sum(axis: 1)
  //   XCTAssertEqual(sumC.data, Array([1 + 3, 2 + 4, 5 + 7, 6 + 8].map { Float($0) }))
  //   sumC.backward(grad: Tensor(data: [-1, -2, -3, -4], shape: [2, 2]))
  //   XCTAssertEqual(gradC!.data, [-1, -2, -1, -2, -3, -4, -3, -4])

  //   var gradD: Tensor?
  //   let sumD = input1.onGrad({ g in gradD = g }).sum()
  //   XCTAssertEqual(sumD.data, [input1.data.sum()])
  //   sumD.backward(grad: Tensor(data: [-1], shape: []))
  //   XCTAssertEqual(gradD!.data, Array(repeating: Float(-1), count: 8))

  //   XCTAssertEqual(input1.sum(keepdims: true).shape, [1, 1, 1])
  //   XCTAssertEqual(input1.sum(axis: 0, keepdims: true).shape, [1, 2, 2])
  //   XCTAssertEqual(input1.sum(axis: 1, keepdims: true).shape, [2, 1, 2])
  //   XCTAssertEqual(input1.sum(axis: 2, keepdims: true).shape, [2, 2, 1])
  // }

  // func testMinMax() throws {
  //   let input = Tensor(data: [1, 10, 2, 7, 8, 9, 6, 4, 5], shape: [3, 3])
  //   var gradA: Tensor?
  //   let maxA = input.onGrad({ g in gradA = g }).max(axis: 1)
  //   XCTAssertEqual(maxA.data, [10, 9, 6])
  //   maxA.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
  //   XCTAssertEqual(gradA!.data, [0, -1, 0, 0, 0, -2, -3, 0, 0])

  //   var gradB: Tensor?
  //   let maxB = input.onGrad({ g in gradB = g }).max(axis: 0)
  //   XCTAssertEqual(maxB.data, [7, 10, 9])
  //   maxB.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
  //   XCTAssertEqual(gradB!.data, [0, -2, 0, -1, 0, -3, 0, 0, 0])

  //   var gradC: Tensor?
  //   let minC = input.onGrad({ g in gradC = g }).min(axis: 0)
  //   XCTAssertEqual(minC.data, [1, 4, 2])
  //   minC.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
  //   XCTAssertEqual(gradC!.data, [-1, 0, -3, 0, 0, 0, 0, -2, 0])

  //   var gradD: Tensor?
  //   let maxD = input.onGrad({ g in gradD = g }).max()
  //   XCTAssertEqual(maxD.data, [10])
  //   maxD.backward(grad: Tensor(data: [-1], shape: []))
  //   XCTAssertEqual(gradD!.data, [0, -1, 0, 0, 0, 0, 0, 0, 0])

  //   XCTAssertEqual(input.max(keepdims: true).shape, [1, 1])
  //   XCTAssertEqual(input.max(axis: 0, keepdims: true).shape, [1, 3])
  //   XCTAssertEqual(input.max(axis: 1, keepdims: true).shape, [3, 1])
  // }

  // func testExpandAndRepeat() throws {
  //   let t1 = Tensor(data: [1, 2, 3, 4], shape: [2, 2])
  //   XCTAssertEqual(
  //     t1.reshape([1, 2, 1, 2]).expand(shape: [3, 7, 2, 5, 2]).shape, [3, 7, 2, 5, 2])

  //   let t2 = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [1, 2, 1, 3, 1])
  //   XCTAssertEqual(t2.repeating(axis: 0, count: 2).data, t2.data + t2.data)
  //   XCTAssertEqual(t2.repeating(axis: 1, count: 2).data, t2.data + t2.data)
  //   XCTAssertEqual(t2.repeating(axis: 2, count: 2).data, [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])
  //   XCTAssertEqual(t2.repeating(axis: 3, count: 2).data, [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])
  //   XCTAssertEqual(t2.repeating(axis: 4, count: 2).data, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

  //   var grad: Tensor?
  //   let repeated = t2.onGrad({ g in grad = g }).repeating(axis: 2, count: 2)
  //   repeated.backward(
  //     grad: Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [1, 2, 2, 3, 1]))
  //   XCTAssertEqual(grad!.data, [5, 7, 9, 17, 19, 21])
  // }

  // func testSoftmax() throws {
  //   let x = Tensor(
  //     data: [
  //       -0.10201670974493027, 0.19515414535999298, 0.5986292362213135, 1.340445637702942,
  //       -0.11801239848136902, -0.24393606185913086, -1.264183521270752, -1.2100555896759033,
  //       0.2798837423324585, -2.8798062801361084, -0.5698361992835999, 0.44209930300712585,
  //       -0.7118728160858154, -0.6576670408248901, -0.4293822646141052,
  //     ], shape: [3, 5])
  //   let outGrad: Tensor = Tensor(
  //     data: [
  //       1.2432453632354736, 1.933882474899292, -1.1054673194885254, -0.5737214684486389,
  //       1.2679708003997803, 2.3119640350341797, 0.44090110063552856, -1.0582382678985596,
  //       1.030942440032959, -0.6020350456237793, -0.5935935378074646, 0.4871285855770111,
  //       0.17254149913787842, -2.5286428928375244, -0.32286253571510315,
  //     ], shape: [3, 5])

  //   var axis0Grad: Tensor?
  //   let axis0Out = x.onGrad({ g in axis0Grad = g }).logSoftmax(axis: 0)
  //   try assertClose(
  //     axis0Out,
  //     Tensor(
  //       data: [
  //         -0.9139110445976257, -0.9212778806686401, -0.3601568043231964, -0.39329278469085693,
  //         -0.5853509902954102, -1.0558303594589233, -2.380615711212158, -2.168841600418091,
  //         -1.4538546800613403, -3.347144842147827, -1.381730556488037, -0.6743327379226685,
  //         -1.670658826828003, -2.3914055824279785, -0.8967208862304688,
  //       ], shape: [3, 5]))
  //   axis0Out.backward(grad: outGrad)
  //   try assertClose(
  //     axis0Grad!,
  //     Tensor(
  //       data: [
  //         0.05577663704752922, 0.7948125600814819, 0.2835029065608978, 0.8241385221481323,
  //         1.0769097805023193, 1.281607747077942, 0.1761925369501114, -0.8306283950805664,
  //         1.5149670839309692, -0.6141059398651123, -1.3373842239379883, -0.9710049033164978,
  //         0.5471253991127014, -2.3391058444976807, -0.46280384063720703,
  //       ], shape: [3, 5]))

  //   var axis1Grad: Tensor?
  //   let axis1Out = x.onGrad({ g in axis1Grad = g }).logSoftmax(axis: 1)
  //   try assertClose(
  //     axis1Out,
  //     Tensor(
  //       data: [
  //         -2.2592945098876953, -1.9621237516403198, -1.558648705482483, -0.8168323040008545,
  //         -2.2752904891967773, -1.2531013488769531, -2.273348808288574, -2.2192208766937256,
  //         -0.7292815446853638, -3.8889713287353516, -1.8998993635177612, -0.8879638910293579,
  //         -2.041935920715332, -1.9877302646636963, -1.7594454288482666,
  //       ], shape: [3, 5]))
  //   axis1Out.backward(grad: outGrad)
  //   try assertClose(
  //     axis1Grad!,
  //     Tensor(
  //       data: [
  //         0.9544176459312439, 1.545107364654541, -1.6874706745147705, -1.7957807779312134,
  //         0.9837263822555542, 1.7054452896118164, 0.22224761545658112, -1.289053201675415,
  //         0.006856732070446014, -0.645496129989624, -0.17693884670734406, 1.6333123445510864,
  //         0.5340267419815063, -2.147022247314453, 0.156622052192688,
  //       ], shape: [3, 5]))
  // }

  // func testConcatInner() throws {
  //   let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
  //   let y = Tensor(data: [7, 8, 9, 10], shape: [2, 2])
  //   var xGrad: Tensor?
  //   var yGrad: Tensor?
  //   let xWithGrad = x.onGrad({ g in xGrad = g })
  //   let yWithGrad = y.onGrad({ g in yGrad = g })
  //   let combined = Tensor(concat: [xWithGrad, yWithGrad], axis: 1)
  //   XCTAssertEqual(combined.shape, [2, 5])
  //   XCTAssertEqual(combined.data, [1, 2, 3, 7, 8, 4, 5, 6, 9, 10])
  //   combined.backward(grad: Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], shape: [2, 5]))
  //   XCTAssertEqual(xGrad!.data, [1, 2, 3, 6, 7, 8])
  //   XCTAssertEqual(yGrad!.data, [4, 5, 9, 10])
  // }

  // func testConcatOuter() throws {
  //   let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
  //   let y = Tensor(data: [7, 8, 9], shape: [1, 3])
  //   var xGrad: Tensor?
  //   var yGrad: Tensor?
  //   let xWithGrad = x.onGrad({ g in xGrad = g })
  //   let yWithGrad = y.onGrad({ g in yGrad = g })
  //   let combined = Tensor(concat: [xWithGrad, yWithGrad], axis: 0)
  //   XCTAssertEqual(combined.shape, [3, 3])
  //   XCTAssertEqual(combined.data, [1, 2, 3, 4, 5, 6, 7, 8, 9])
  //   combined.backward(grad: Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3]))
  //   XCTAssertEqual(xGrad!.data, [1, 2, 3, 4, 5, 6])
  //   XCTAssertEqual(yGrad!.data, [7, 8, 9])
  // }

  // func testOneHot() throws {
  //   XCTAssertEqual(Tensor(oneHot: 3, count: 5).data, [0, 0, 0, 1, 0])
  //   XCTAssertEqual(Tensor(oneHot: [3, 1], count: 5).data, [0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
  // }

  // func testTrainable() throws {
  //   class Linear: Trainable {
  //     @Parameter(name: "weight") var weight: Tensor
  //     @Parameter(name: "bias") var bias: Tensor

  //     init(inSize: Int, outSize: Int) {
  //       super.init()
  //       weight = Tensor(zeros: [inSize, outSize])
  //       bias = Tensor(zeros: [outSize])
  //     }
  //   }

  //   class Network: Trainable {
  //     @Child(name: "layer0") var layer0: Linear
  //     @Child(name: "layer1") var layer1: Linear

  //     override init() {
  //       super.init()
  //       layer0 = Linear(inSize: 3, outSize: 5)
  //       layer1 = Linear(inSize: 5, outSize: 7)
  //     }
  //   }

  //   let instance = Linear(inSize: 3, outSize: 5)
  //   let params = instance.parameters
  //   XCTAssertEqual(params.count, 2)
  //   XCTAssertEqual(params[0].0, "bias")
  //   XCTAssertEqual(params[0].1.data!.shape, [5])
  //   XCTAssertEqual(params[1].0, "weight")
  //   XCTAssertEqual(params[1].1.data!.shape, [3, 5])
  //   XCTAssert(instance.$bias.data != nil)
  //   XCTAssert(instance.$weight.data != nil)
  //   XCTAssertEqual(instance.bias.shape, [5])
  //   XCTAssertEqual(instance.weight.shape, [3, 5])

  //   let net = Network()
  //   let netParams = net.parameters
  //   XCTAssertEqual(netParams.count, 4)
  //   XCTAssertEqual(netParams[0].0, "layer0.bias")
  //   XCTAssertEqual(netParams[0].1.data!.shape, [5])
  //   XCTAssertEqual(netParams[1].0, "layer0.weight")
  //   XCTAssertEqual(netParams[1].1.data!.shape, [3, 5])
  //   XCTAssertEqual(netParams[2].0, "layer1.bias")
  //   XCTAssertEqual(netParams[2].1.data!.shape, [7])
  //   XCTAssertEqual(netParams[3].0, "layer1.weight")
  //   XCTAssertEqual(netParams[3].1.data!.shape, [5, 7])
  // }

  // func testAdam() throws {
  //   class Linear: Trainable {
  //     @Parameter(name: "weight") var weight: Tensor
  //     @Parameter(name: "bias") var bias: Tensor

  //     override init() {
  //       super.init()
  //       weight = Tensor(zeros: [])
  //       bias = Tensor(zeros: [])
  //     }

  //     func forward(_ x: Tensor) -> Tensor {
  //       return weight.expand(as: x) * x + bias.expand(as: x)
  //     }
  //   }
  //   let model = Linear()
  //   let opt = Adam(model.parameters, lr: 0.025)
  //   let inputs = Tensor(data: [-2, -1, 0, 1, 2, 3, 4, 5], shape: [2, 4])
  //   let outputs = inputs * 3.142 + 2.718
  //   for _ in 0..<1000 {
  //     let loss = (model.forward(inputs) - outputs).pow(2).sum()
  //     loss.backward()
  //     opt.step()
  //     opt.clearGrads()
  //   }
  //   XCTAssert(
  //     abs(model.weight.data[0] - 3.142) < 0.05, "model.weight.data[0]=\(model.weight.data[0])")
  //   XCTAssert(abs(model.bias.data[0] - 2.718) < 0.05, "model.bias.data[0]=\(model.bias.data[0])")
  // }
}

// func assertClose(_ x: Tensor, _ y: Tensor, atol: Float = 1e-4, rtol: Float = 1e-4) throws {
//   XCTAssertEqual(x.shape, y.shape)
//   var allGood = true
//   for (a, b) in zip(x.data, y.data) {
//     if abs(a - b) > atol && (b == 0 || abs(a / b - 1) > rtol) {
//       allGood = false
//     }
//   }
//   XCTAssert(allGood, "tensors \(x.data) and \(y.data) are not equal")
// }
