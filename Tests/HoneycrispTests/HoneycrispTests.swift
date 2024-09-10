import XCTest

@testable import Honeycrisp

final class HoneycrispTests: XCTestCase {
  func testCast() async throws {
    let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
    try await assertDataEqual(x, [1.0, 2.0, 0.0])
    let y = x.cast(.float16)
    XCTAssertEqual(y.dtype, .float16)
    try await assertDataEqual(y, [1.0, 2.0, 0.0])
    let z = x.cast(.int64)
    try await assertDataEqual(z, [1.0, 2.0, 0.0])
    let w = x.cast(.bool)
    try await assertDataEqual(w, [1.0, 1.0, 0.0])
  }

  func testAdd() async throws {
    let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
    let y = Tensor(data: [-1.0, 2.0, -3.0], shape: [3], dtype: .float32)
    try await assertDataEqual(x + y, [0.0, 4.0, -3.0])
    try await assertDataEqual(x + 3, [4.0, 5.0, 3.0])
    try await assertDataEqual(x + 1.5, [2.5, 3.5, 1.5])
    try await assertDataEqual(1.5 + x, [2.5, 3.5, 1.5])
  }

  func testMulGrad() async throws {
    let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
    let y = Tensor(data: [-1.0, 2.0, -3.0], shape: [3], dtype: .float32)

    var xGrad: Tensor?
    var yGrad: Tensor?
    let xWithGrad = x.onGrad { grad in xGrad = grad }
    let yWithGrad = y.onGrad { grad in yGrad = grad }
    let product = (xWithGrad * 2) * yWithGrad
    try await assertDataEqual(product, [-2.0, 8.0, -0.0])
    product.backward()
    try await assertDataEqual(xGrad!, [-2.0, 4.0, -6.0])
    try await assertDataEqual(yGrad!, [2.0, 4.0, 0.0])
  }

  func testMSEGrad() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0], shape: [3])
    let y = Tensor(data: [2.0, 0.0, -3.0], shape: [3])
    var xGrad: Tensor?
    let diff = x.onGrad({ grad in xGrad = grad }) - y
    let sqDiff = diff * diff
    sqDiff.backward(Tensor(onesLike: x))
    try await assertDataEqual(xGrad!, [-2, 4, 12])
  }

  func testEquals() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0, -2.0, 3.0], shape: [5])
    let y = Tensor(data: [1, 2, 2, -2, -3], shape: [5]).cast(as: x)
    try await assertDataEqual(x == y, [1, 1, 0, 1, 0])
    XCTAssertEqual((x == y).dtype, .bool)
    XCTAssertEqual((x == y).shape, x.shape)

    let t1 = Tensor(data: [1, 2, 3, 4, 3, 1, 2, 0], shape: [2, 4])
    let t2 = Tensor(data: [1, 0, 4, 4, 0, 0, 1, 0], shape: [2, 4])
    try await assertDataEqual(t1 == 2, [0, 1, 0, 0, 0, 0, 1, 0])
    try await assertDataEqual(t1 == t2, [1, 0, 0, 1, 0, 0, 0, 1])
  }

  func testSum() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0, -2.0, 3.0, 7.0], shape: [1, 2, 3, 1])
    var xGrad: Tensor?

    func useX() -> Tensor {
      x.onGrad { xGrad = $0 }
    }

    var sum = useX().sum(axis: 2)
    XCTAssertEqual(sum.shape, [1, 2, 1])
    sum.backward(Tensor(data: [-1.0, -2.0], shape: [1, 2, 1]))
    try await assertDataEqual(sum, [6.0, 8.0])
    try await assertDataEqual(xGrad!, [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0])

    sum = useX().sum(axis: 2, keepdims: true)
    XCTAssertEqual(sum.shape, [1, 2, 1, 1])
    sum.backward(Tensor(data: [-1.0, -2.0], shape: [1, 2, 1, 1]))
    try await assertDataEqual(sum, [6.0, 8.0])
    try await assertDataEqual(xGrad!, [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0])

    sum = useX().sum(axis: 1)
    XCTAssertEqual(sum.shape, [1, 3, 1])
    sum.backward(Tensor(data: [-1.0, -2.0, -3.0], shape: [1, 3, 1]))
    try await assertDataEqual(sum, [-1.0, 5.0, 10.0])
    try await assertDataEqual(xGrad!, [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0])

    sum = useX().sum(axis: 1, keepdims: true)
    XCTAssertEqual(sum.shape, [1, 1, 3, 1])
    sum.backward(Tensor(data: [-1.0, -2.0, -3.0], shape: [1, 1, 3, 1]))
    try await assertDataEqual(sum, [-1.0, 5.0, 10.0])
    try await assertDataEqual(xGrad!, [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0])

    for axis in [0, -1] {
      for keepdims in [false, true] {
        sum = useX().sum(axis: axis, keepdims: keepdims)
        if keepdims {
          XCTAssertEqual(sum.shape, [1, 2, 3, 1])
        } else if axis == 0 {
          XCTAssertEqual(sum.shape, [2, 3, 1])
        } else {
          XCTAssertEqual(sum.shape, [1, 2, 3])
        }
        sum.backward(Tensor(data: [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], shape: sum.shape))
        try await assertDataEqual(sum, x)
        try await assertDataEqual(xGrad!, [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0])
      }
    }

    sum = useX().sum()
    XCTAssertEqual(sum.shape, [])
    sum.backward()
    try await assertDataEqual(sum, [14.0])
    try await assertDataEqual(xGrad!, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    sum = useX().sum(keepdims: true)
    XCTAssertEqual(sum.shape, [1, 1, 1, 1])
    sum.backward()
    try await assertDataEqual(sum, [14.0])
    try await assertDataEqual(xGrad!, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    // Older tests are below

    let input = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3])
    var gradA: Tensor?
    let sumA = input.onGrad({ g in gradA = g }).sum(axis: 1)
    try await assertDataEqual(sumA, [6, 15, 24])
    sumA.backward(Tensor(data: [-1, -2, -3], shape: [3]))
    try await assertDataEqual(gradA!, [-1, -1, -1, -2, -2, -2, -3, -3, -3])

    var gradB: Tensor?
    let sumB = input.onGrad({ g in gradB = g }).sum(axis: 0)
    try await assertDataEqual(sumB, [12, 15, 18])
    sumB.backward(Tensor(data: [-1, -2, -3], shape: [3]))
    try await assertDataEqual(gradB!, [-1, -2, -3, -1, -2, -3, -1, -2, -3])

    let input1 = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8], shape: [2, 2, 2])

    var gradC: Tensor?
    let sumC = input1.onGrad({ g in gradC = g }).sum(axis: 1)
    try await assertDataEqual(sumC, Array([1 + 3, 2 + 4, 5 + 7, 6 + 8].map { Float($0) }))
    sumC.backward(Tensor(data: [-1, -2, -3, -4], shape: [2, 2]))
    try await assertDataEqual(gradC!, [-1, -2, -1, -2, -3, -4, -3, -4])

    var gradD: Tensor?
    let sumD = input1.onGrad({ g in gradD = g }).sum()
    try await assertDataEqual(sumD, [[1, 2, 3, 4, 5, 6, 7, 8].sum()])
    sumD.backward(Tensor(data: [-1], shape: []))
    try await assertDataEqual(gradD!, Array(repeating: Float(-1), count: 8))

    XCTAssertEqual(input1.sum(keepdims: true).shape, [1, 1, 1])
    XCTAssertEqual(input1.sum(axis: 0, keepdims: true).shape, [1, 2, 2])
    XCTAssertEqual(input1.sum(axis: 1, keepdims: true).shape, [2, 1, 2])
    XCTAssertEqual(input1.sum(axis: 2, keepdims: true).shape, [2, 2, 1])
  }

  func testRepeat() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0, -2.0, 3.0, 7.0], shape: [1, 2, 3, 1])
    var xGrad: Tensor?

    func useX() -> Tensor {
      x.onGrad { xGrad = $0 }
    }

    let repeated = useX().repeating(axis: 2, count: 2)
    XCTAssertEqual(repeated.shape, [1, 2, 6, 1])
    repeated.backward(
      Tensor(
        data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape: [1, 2, 6, 1]))
    try await assertDataEqual(
      repeated, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, -2.0, 3.0, 7.0, -2.0, 3.0, 7.0])
    try await assertDataEqual(xGrad!, [5.0, 7.0, 9.0, 17.0, 19.0, 21.0])
  }

  func testGather() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0, -2.0, 3.0, 7.0], shape: [1, 2, 3, 1])
    var xGrad: Tensor?

    func useX() -> Tensor {
      x.onGrad { xGrad = $0 }
    }

    // Unbroadcasted gather along inner axis
    var out = useX().gather(
      axis: 2, indices: Tensor(data: [2, 0, 1, 2], shape: [1, 2, 2, 1], dtype: .int64))
    XCTAssertEqual(out.shape, [1, 2, 2, 1])
    out.backward(Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 2, 2, 1]))
    try await assertDataEqual(out, [3.0, 1.0, 3.0, 7.0])
    try await assertDataEqual(xGrad!, [2.0, 0.0, 1.0, 0.0, 3.0, 4.0])

    // Broadcasted gather along inner axis
    out = useX().gather(
      axis: 2, indices: Tensor(data: [2, 0], shape: [2], dtype: .int64))
    XCTAssertEqual(out.shape, [1, 2, 2, 1])
    out.backward(Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 2, 2, 1]))
    try await assertDataEqual(out, [3.0, 1.0, 7.0, -2.0])
    try await assertDataEqual(xGrad!, [2.0, 0.0, 1.0, 4.0, 0.0, 3.0])

    // Unbroadcasted gather along outer axis
    out = useX().gather(
      axis: 1, indices: Tensor(data: [0, 1, 0, 1, 0, 0], shape: [1, 2, 3, 1], dtype: .int64))
    XCTAssertEqual(out.shape, [1, 2, 3, 1])
    out.backward(Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [1, 2, 3, 1]))
    try await assertDataEqual(out, [1.0, 3.0, 3.0, -2.0, 2.0, 3.0])
    try await assertDataEqual(xGrad!, [1.0, 5.0, 9.0, 4.0, 2.0, 0.0])

    // Broadcasted scatter along outer axis
    let permuteMe = Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [1, 2, 3, 1])
    let permuted = try await permuteMe.scatter(
      axis: 1, count: 2, indices: Tensor(data: [1, 0], shape: [2])
    ).floats()
    XCTAssertEqual(permuted, [4.0, 5.0, 6.0, 1.0, 2.0, 3.0])

    // Broadcasted gather along outer axis
    out = useX().gather(
      axis: 1, indices: Tensor(data: [1, 0], shape: [2], dtype: .int64))
    XCTAssertEqual(out.shape, [1, 2, 3, 1])
    out.backward(Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [1, 2, 3, 1]))
    try await assertDataEqual(out, [-2.0, 3.0, 7.0, 1.0, 2.0, 3.0])
    try await assertDataEqual(xGrad!, [4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
  }

  func testMatrixMatrixProduct() async throws {
    // Sanity check for transposes.
    try await {
      let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
      let y = Tensor(data: [-1, -2, -3, -4, -5, -6], shape: [3, 2])
      let z1 = Tensor.matmul(a: x, transA: false, b: y, transB: false, transOut: false)
      let z2 = Tensor.matmul(a: y, transA: true, b: x, transB: true, transOut: true)
      XCTAssertEqual(z1.shape, [2, 2])
      XCTAssertEqual(z2.shape, [2, 2])
      try await assertDataEqual(z1, [-22, -28, -49, -64])
      try await assertDataEqual(z2, [-22, -28, -49, -64])
    }()

    try await {
      let x = Tensor(ones: [64, 128])
      let y = Tensor(ones: [128, 32])
      let z = Tensor.matmul(a: x, transA: false, b: y, transB: false, transOut: false)
      XCTAssertEqual(z.shape, [64, 32])
      try await assertDataEqual(z, [Float](repeating: 128, count: 64 * 32))
    }()
  }

  func testMatrixVectorProduct() async throws {
    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
    let y = Tensor(data: [-1, -3, 2], shape: [3, 1])
    var xGrad: Tensor?
    var yGrad: Tensor?
    let xParam = x.onGrad { grad in xGrad = grad }
    let yParam = y.onGrad { grad in yGrad = grad }
    let product = xParam &* yParam
    try await assertDataEqual(product, [-1, -7])
    product.backward()
    try await assertDataEqual(xGrad!, [-1, -3, 2, -1, -3, 2])
    XCTAssertEqual(xGrad!.shape, x.shape)
    try await assertDataEqual(yGrad!, [5, 7, 9])
    XCTAssertEqual(yGrad!.shape, y.shape)
  }

  func testIndexing() async throws {
    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [3, 2])
    XCTAssertEqual(x[0].shape, [2])
    XCTAssertEqual(x[..., 0].shape, [3])
    try await assertDataEqual(x[1], [3, 4])
    try await assertDataEqual(x[1..<3], [3, 4, 5, 6])
    try await assertDataEqual(x[1...2], [3, 4, 5, 6])
    try await assertDataEqual(x[1..<2], [3, 4])
    try await assertDataEqual(x[(-2)..<(-1)], [3, 4])
    try await assertDataEqual(x[(-2)...(-1)], [3, 4, 5, 6])
    try await assertDataEqual(x[(-2)...], [3, 4, 5, 6])
    try await assertDataEqual(x[...(-2)], [1, 2, 3, 4])
    try await assertDataEqual(x[..<(-2)], [1, 2])

    let y = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [3, 1, 4])
    XCTAssertEqual(y[..., 0].shape, [3, 4])
    try await assertDataEqual(y[..., 0], y)
    XCTAssertEqual(y[0, ..., 0].shape, [1])
    try await assertDataEqual(y[0, ..., 0], [1])
    XCTAssertEqual(y[..., 0, ...].shape, [3, 4])
    try await assertDataEqual(y[0, FullRange(dims: 2)], [1, 2, 3, 4])
    XCTAssertEqual(y[0, FullRange(dims: 2)].shape, [1, 4])
    try await assertDataEqual(y[..., 0, ...], y)
    XCTAssertEqual(y[0, 0, 0].shape, [])
    try await assertDataEqual(y[0, 0, 0], [1])
    try await assertDataEqual(y[0...1, ..., 3], [4, 8])
    try await assertDataEqual(y[0..<2, ..., 3], [4, 8])
    try await assertDataEqual(y[0...2, ..., 3], [4, 8, 12])
    try await assertDataEqual(y[0..<3, ..., 3], [4, 8, 12])
    try await assertDataEqual(y[0..., ..., 3], [4, 8, 12])
    try await assertDataEqual(y[1...2, ..., 2...3], [7, 8, 11, 12])
    try await assertDataEqual(y[0...2, ..., 2...3], [3, 4, 7, 8, 11, 12])
    try await assertDataEqual(y[0...2, 0..<1, 2...3], [3, 4, 7, 8, 11, 12])
    try await assertDataEqual(y[FullRange(dims: 3), NewAxis()], y)
    XCTAssertEqual(y[FullRange(dims: 3), NewAxis()].shape, [3, 1, 4, 1])
    try await assertDataEqual(y[FullRange(dims: 2), NewAxis()], y)
    XCTAssertEqual(y[FullRange(dims: 2), NewAxis()].shape, [3, 1, 1, 4])
    try await assertDataEqual(y[FullRange(dims: 1), NewAxis()], y)
    XCTAssertEqual(y[FullRange(dims: 1), NewAxis()].shape, [3, 1, 1, 4])
    try await assertDataEqual(y[NewAxis()], y)
    XCTAssertEqual(y[NewAxis()].shape, [1, 3, 1, 4])

    XCTAssertEqual(y[..., 0, 3].shape, [3])

    var yGrad: Tensor?
    let yParam = y.onGrad { grad in yGrad = grad }
    yParam[1...2, ..., 2...3].backward(Tensor(data: [1, 2, 3, 4], shape: [2, 1, 2]))
    XCTAssertEqual(yGrad!.shape, y.shape)
    try await assertDataEqual(yGrad!, [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4])

    let yParam1 = y.onGrad { grad in yGrad = grad }
    yParam1[..., 0, 3].backward(Tensor(data: [1, 2, 3], shape: [3]))
    XCTAssertEqual(yGrad!.shape, y.shape)
    try await assertDataEqual(yGrad!, [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3])
  }

  func testElemwise() async throws {
    func testF(input: [Float], output: [Float], grad: [Float], _ op: (Tensor) -> Tensor)
      async throws
    {
      var actualGrad: Tensor?
      let tensorIn = Tensor(data: input, shape: [input.count]) { g in
        actualGrad = g
      }
      assert(tensorIn.needsGrad, "\(tensorIn.dtype) \(tensorIn.needsGrad)")
      let actualOut = op(tensorIn)
      try await assertClose(actualOut, Tensor(data: output, shape: [output.count]))
      actualOut.backward(Tensor(onesLike: actualOut))
      try await assertClose(actualGrad!, Tensor(data: grad, shape: [output.count]))
    }

    try await testF(
      input: [-1, -2, -3, 1, 2, 3],
      output: [1, 4, 9, 1, 4, 9],
      grad: [-2, -4, -6, 2, 4, 6]
    ) { $0.pow(2) }
    try await testF(
      input: [1, 2],
      output: [2.718281828459045, 7.38905609893065],
      grad: [2.718281828459045, 7.38905609893065]
    ) { $0.exp() }
    try await testF(
      input: [0.01, 1, 2, 3],
      output: [0.1, 1.0, sqrt(2.0), sqrt(3.0)],
      grad: [5.0, 0.5, 0.3535533845424652, 0.28867512941360474]
    ) { $0.sqrt() }
    try await testF(
      input: [-100, -2, 0, 1, 2, 3, 100],
      output: [
        -0.0, -0.04540228843688965, 0.0, 0.8411920070648193, 1.9545977115631104, 2.9963626861572266,
        100.0,
      ],
      grad: [
        0.0, -0.08609922230243683, 0.5, 1.0829640626907349, 1.0860992670059204, 1.0115842819213867,
        1.0,
      ]
    ) { $0.gelu() }
  }

  func testMinMax() async throws {
    let input = Tensor(data: [1, 10, 2, 7, 8, 9, 6, 4, 5], shape: [3, 3])
    var gradA: Tensor?
    let maxA = input.onGrad({ g in gradA = g }).max(axis: 1)
    try await assertDataEqual(maxA, [10, 9, 6])
    maxA.backward(Tensor(data: [-1, -2, -3], shape: [3]))
    try await assertDataEqual(gradA!, [0, -1, 0, 0, 0, -2, -3, 0, 0])

    var gradB: Tensor?
    let maxB = input.onGrad({ g in gradB = g }).max(axis: 0)
    try await assertDataEqual(maxB, [7, 10, 9])
    maxB.backward(Tensor(data: [-1, -2, -3], shape: [3]))
    try await assertDataEqual(gradB!, [0, -2, 0, -1, 0, -3, 0, 0, 0])

    var gradC: Tensor?
    let minC = input.onGrad({ g in gradC = g }).min(axis: 0)
    try await assertDataEqual(minC, [1, 4, 2])
    minC.backward(Tensor(data: [-1, -2, -3], shape: [3]))
    try await assertDataEqual(gradC!, [-1, 0, -3, 0, 0, 0, 0, -2, 0])

    var gradD: Tensor?
    let maxD = input.onGrad({ g in gradD = g }).max()
    XCTAssertEqual(maxD.shape, [])
    try await assertDataEqual(maxD, [10])
    maxD.backward(Tensor(data: [-1], shape: []))
    try await assertDataEqual(gradD!, [0, -1, 0, 0, 0, 0, 0, 0, 0])

    var gradE: Tensor?
    let maxE = input.onGrad({ g in gradE = g }).min()
    XCTAssertEqual(maxE.shape, [])
    try await assertDataEqual(maxE, [1])
    maxE.backward(Tensor(data: [-1], shape: []))
    try await assertDataEqual(gradE!, [-1.0, 0, 0, 0, 0, 0, 0, 0, 0])

    XCTAssertEqual(input.max(keepdims: true).shape, [1, 1])
    XCTAssertEqual(input.max(axis: 0, keepdims: true).shape, [1, 3])
    XCTAssertEqual(input.max(axis: 1, keepdims: true).shape, [3, 1])
  }

  func testExpandAndRepeat() async throws {
    let t1Arr: [Float] = [1, 2, 3, 4]
    let t2Arr: [Float] = [1, 2, 3, 4, 5, 6]
    let t1 = Tensor(data: t1Arr, shape: [2, 2])
    XCTAssertEqual(
      t1.reshape([1, 2, 1, 2]).expand(shape: [3, 7, 2, 5, 2]).shape, [3, 7, 2, 5, 2])

    let t2 = Tensor(data: t2Arr, shape: [1, 2, 1, 3, 1])
    try await assertDataEqual(t2.repeating(axis: 0, count: 2), t2Arr + t2Arr)
    try await assertDataEqual(t2.repeating(axis: 1, count: 2), t2Arr + t2Arr)
    try await assertDataEqual(
      t2.repeating(axis: 2, count: 2), [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])
    try await assertDataEqual(
      t2.repeating(axis: 3, count: 2), [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])
    try await assertDataEqual(
      t2.repeating(axis: 4, count: 2), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

    var grad: Tensor?
    let repeated = t2.onGrad({ g in grad = g }).repeating(axis: 2, count: 2)
    repeated.backward(
      Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [1, 2, 2, 3, 1]).cast(as: t1))
    try await assertDataEqual(grad!, [5, 7, 9, 17, 19, 21])

    // Repeating past last axis should introduce a new dimension
    XCTAssertEqual(t1.repeating(axis: 2, count: 3).shape, [2, 2, 3])
    try await assertDataEqual(
      t1.repeating(axis: 2, count: 3), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
  }

  func testSoftmax() async throws {
    let x = Tensor(
      data: [
        -0.10201670974493027, 0.19515414535999298, 0.5986292362213135, 1.340445637702942,
        -0.11801239848136902, -0.24393606185913086, -1.264183521270752, -1.2100555896759033,
        0.2798837423324585, -2.8798062801361084, -0.5698361992835999, 0.44209930300712585,
        -0.7118728160858154, -0.6576670408248901, -0.4293822646141052,
      ], shape: [3, 5])
    let outGrad: Tensor = Tensor(
      data: [
        1.2432453632354736, 1.933882474899292, -1.1054673194885254, -0.5737214684486389,
        1.2679708003997803, 2.3119640350341797, 0.44090110063552856, -1.0582382678985596,
        1.030942440032959, -0.6020350456237793, -0.5935935378074646, 0.4871285855770111,
        0.17254149913787842, -2.5286428928375244, -0.32286253571510315,
      ], shape: [3, 5])

    var axis0Grad: Tensor?
    let axis0Out = x.onGrad({ g in axis0Grad = g }).logSoftmax(axis: 0)
    try await assertClose(
      axis0Out,
      Tensor(
        data: [
          -0.9139110445976257, -0.9212778806686401, -0.3601568043231964, -0.39329278469085693,
          -0.5853509902954102, -1.0558303594589233, -2.380615711212158, -2.168841600418091,
          -1.4538546800613403, -3.347144842147827, -1.381730556488037, -0.6743327379226685,
          -1.670658826828003, -2.3914055824279785, -0.8967208862304688,
        ], shape: [3, 5]))
    axis0Out.backward(outGrad)
    try await assertClose(
      axis0Grad!,
      Tensor(
        data: [
          0.05577663704752922, 0.7948125600814819, 0.2835029065608978, 0.8241385221481323,
          1.0769097805023193, 1.281607747077942, 0.1761925369501114, -0.8306283950805664,
          1.5149670839309692, -0.6141059398651123, -1.3373842239379883, -0.9710049033164978,
          0.5471253991127014, -2.3391058444976807, -0.46280384063720703,
        ], shape: [3, 5]))

    var axis1Grad: Tensor?
    let axis1Out = x.onGrad({ g in axis1Grad = g }).logSoftmax(axis: 1)
    try await assertClose(
      axis1Out,
      Tensor(
        data: [
          -2.2592945098876953, -1.9621237516403198, -1.558648705482483, -0.8168323040008545,
          -2.2752904891967773, -1.2531013488769531, -2.273348808288574, -2.2192208766937256,
          -0.7292815446853638, -3.8889713287353516, -1.8998993635177612, -0.8879638910293579,
          -2.041935920715332, -1.9877302646636963, -1.7594454288482666,
        ], shape: [3, 5]))
    axis1Out.backward(outGrad)
    try await assertClose(
      axis1Grad!,
      Tensor(
        data: [
          0.9544176459312439, 1.545107364654541, -1.6874706745147705, -1.7957807779312134,
          0.9837263822555542, 1.7054452896118164, 0.22224761545658112, -1.289053201675415,
          0.006856732070446014, -0.645496129989624, -0.17693884670734406, 1.6333123445510864,
          0.5340267419815063, -2.147022247314453, 0.156622052192688,
        ], shape: [3, 5]))
  }

  func testConcatInner() async throws {
    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3]).cast(.float32)
    let y = Tensor(data: [7, 8, 9, 10], shape: [2, 2]).cast(.float32)
    var xGrad: Tensor?
    var yGrad: Tensor?
    let xWithGrad = x.onGrad({ g in xGrad = g })
    let yWithGrad = y.onGrad({ g in yGrad = g })
    let combined = Tensor(concat: [xWithGrad, yWithGrad], axis: 1)
    XCTAssertEqual(combined.shape, [2, 5])
    try await assertDataEqual(combined, [1, 2, 3, 7, 8, 4, 5, 6, 9, 10])
    combined.backward(Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], shape: [2, 5]).cast(as: x))
    try await assertDataEqual(xGrad!, [1, 2, 3, 6, 7, 8])
    try await assertDataEqual(yGrad!, [4, 5, 9, 10])
  }

  func testConcatOuter() async throws {
    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3]).cast(.float32)
    let y = Tensor(data: [7, 8, 9], shape: [1, 3]).cast(.float32)
    var xGrad: Tensor?
    var yGrad: Tensor?
    let xWithGrad = x.onGrad({ g in xGrad = g })
    let yWithGrad = y.onGrad({ g in yGrad = g })
    let combined = Tensor(concat: [xWithGrad, yWithGrad], axis: 0)
    XCTAssertEqual(combined.shape, [3, 3])
    try await assertDataEqual(combined, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    combined.backward(Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3]).cast(as: x))
    try await assertDataEqual(xGrad!, [1, 2, 3, 4, 5, 6])
    try await assertDataEqual(yGrad!, [7, 8, 9])
  }

  func testOneHot() async throws {
    try await assertDataEqual(Tensor(oneHot: 3, count: 5), [0, 0, 0, 1, 0])
    try await assertDataEqual(Tensor(oneHot: [3, 1], count: 5), [0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
  }

  func testTrainable() throws {
    class Linear: Trainable {
      @Parameter(name: "weight") var weight: Tensor
      @Parameter(name: "bias") var bias: Tensor

      init(inSize: Int, outSize: Int) {
        super.init()
        weight = Tensor(zeros: [inSize, outSize])
        bias = Tensor(zeros: [outSize])
      }
    }

    class Network: Trainable {
      @Child(name: "layer0") var layer0: Linear
      @Child(name: "layer1") var layer1: Linear

      override init() {
        super.init()
        layer0 = Linear(inSize: 3, outSize: 5)
        layer1 = Linear(inSize: 5, outSize: 7)
      }
    }

    let instance = Linear(inSize: 3, outSize: 5)
    let params = instance.parameters
    XCTAssertEqual(params.count, 2)
    XCTAssertEqual(params[0].0, "bias")
    XCTAssertEqual(params[0].1.data!.shape, [5])
    XCTAssertEqual(params[1].0, "weight")
    XCTAssertEqual(params[1].1.data!.shape, [3, 5])
    XCTAssert(instance.$bias.data != nil)
    XCTAssert(instance.$weight.data != nil)
    XCTAssertEqual(instance.bias.shape, [5])
    XCTAssertEqual(instance.weight.shape, [3, 5])

    let net = Network()
    let netParams = net.parameters
    XCTAssertEqual(netParams.count, 4)
    XCTAssertEqual(netParams[0].0, "layer0.bias")
    XCTAssertEqual(netParams[0].1.data!.shape, [5])
    XCTAssertEqual(netParams[1].0, "layer0.weight")
    XCTAssertEqual(netParams[1].1.data!.shape, [3, 5])
    XCTAssertEqual(netParams[2].0, "layer1.bias")
    XCTAssertEqual(netParams[2].1.data!.shape, [7])
    XCTAssertEqual(netParams[3].0, "layer1.weight")
    XCTAssertEqual(netParams[3].1.data!.shape, [5, 7])
  }

  func testAdam() async throws {
    class Linear: Trainable {
      @Parameter(name: "weight") var weight: Tensor
      @Parameter(name: "bias") var bias: Tensor

      override init() {
        super.init()
        weight = Tensor(zeros: [])
        bias = Tensor(zeros: [])
      }

      func forward(_ x: Tensor) -> Tensor {
        return weight.expand(as: x) * x + bias.expand(as: x)
      }
    }
    let model = Linear()
    let opt = Adam(model.parameters, lr: 0.025)
    let inputs = Tensor(data: [-2, -1, 0, 1, 2, 3, 4, 5], shape: [2, 4]).cast(.float32)
    let outputs = inputs * 3.142 + 2.718
    for _ in 0..<1000 {
      let loss = (model.forward(inputs) - outputs).pow(2).sum()
      loss.backward()
      opt.step()
      opt.clearGrads()
    }
    let wData = try await model.weight.item()
    let bData = try await model.bias.item()
    let wErr = abs(wData - 3.142)
    let bErr = abs(bData - 2.718)
    XCTAssert(wErr < 0.05, "model.weight.data[0]=\(wData)")
    XCTAssert(bErr < 0.05, "model.bias.data[0]=\(bData)")
  }
}

func assertClose(
  _ x: Tensor, _ y: Tensor, atol: Float = 1e-4, rtol: Float = 1e-4, file: StaticString = #filePath,
  line: UInt = #line
) async throws {
  XCTAssertEqual(x.shape, y.shape)
  var allGood = true
  let xData = try await x.floats()
  let yData = try await y.floats()
  for (a, b) in zip(xData, yData) {
    if abs(a - b) > atol && (b == 0 || abs(a / b - 1) > rtol) {
      allGood = false
    }
  }
  XCTAssert(allGood, "tensors \(xData) and \(yData) are not equal", file: file, line: line)
}

func assertDataEqual(
  _ x: Tensor, _ y: [Float], file: StaticString = #filePath, line: UInt = #line
) async throws {
  let data = try await x.floats()
  XCTAssertEqual(data, y, file: file, line: line)
}

func assertDataEqual(
  _ x: Tensor, _ y: Tensor, file: StaticString = #filePath, line: UInt = #line
) async throws {
  let data = try await x.floats()
  let data1 = try await y.floats()
  XCTAssertEqual(data, data1, file: file, line: line)
}
