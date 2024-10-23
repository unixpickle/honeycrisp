import Gzip
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
      let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
      try await assertDataEqual(x, [1.0, 2.0, 0.0])
      let y = x.cast(.float16)
      XCTAssertEqual(y.dtype, .float16)
      try await assertDataEqual(y, [1.0, 2.0, 0.0])
      let z = x.cast(.int64)
      try await assertDataEqual(z, [1.0, 2.0, 0.0])
      let w = x.cast(.bool)
      try await assertDataEqual(w, [1.0, 1.0, 0.0])

      let big = Tensor(data: [1_000_000_000], shape: [1], dtype: .float32)
      let bigFloat = try await big.cast(.float16).item()
      XCTAssert(!bigFloat.isFinite)
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

  func testAdd() async throws {
    try await runInBackends {
      let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
      let y = Tensor(data: [-1.0, 2.0, -3.0], shape: [3], dtype: .float32)
      try await assertDataEqual(x + y, [0.0, 4.0, -3.0])
      try await assertDataEqual(x.cast(.int64) + y.cast(.int64), [0.0, 4.0, -3.0])
      try await assertDataEqual(x.cast(.float16) + y.cast(.float16), [0.0, 4.0, -3.0])
      try await assertDataEqual(x + 3, [4.0, 5.0, 3.0])
      try await assertDataEqual(x.cast(.int64) + 3, [4.0, 5.0, 3.0])
      try await assertDataEqual(x.cast(.float16) + 3, [4.0, 5.0, 3.0])
      try await assertDataEqual(3 + x, [4.0, 5.0, 3.0])
      try await assertDataEqual(3 + x.cast(.int64), [4.0, 5.0, 3.0])
      try await assertDataEqual(3 + x.cast(.float16), [4.0, 5.0, 3.0])
      try await assertDataEqual(x + 1.5, [2.5, 3.5, 1.5])
      try await assertDataEqual(1.5 + x, [2.5, 3.5, 1.5])
    }
  }

  func testSub() async throws {
    try await runInBackends {
      let x = Tensor(data: [1.0, 2.0, 0.0], shape: [3], dtype: .float32)
      let y = Tensor(data: [-1.0, 2.0, -3.0], shape: [3], dtype: .float32)
      try await assertDataEqual(x - y, [2.0, 0.0, 3.0])
      try await assertDataEqual(x.cast(.int64) - y.cast(.int64), [2.0, 0.0, 3.0])
      try await assertDataEqual(x.cast(.float16) - y.cast(.float16), [2.0, 0.0, 3.0])
      try await assertDataEqual(x - (-3), [4.0, 5.0, 3.0])
      try await assertDataEqual(x.cast(.int64) - (-3), [4.0, 5.0, 3.0])
      try await assertDataEqual(x.cast(.float16) - (-3), [4.0, 5.0, 3.0])
      try await assertDataEqual(3 - x, [2.0, 1.0, 3.0])
      try await assertDataEqual(3 - x.cast(.int64), [2.0, 1.0, 3.0])
      try await assertDataEqual(3 - x.cast(.float16), [2.0, 1.0, 3.0])
      try await assertDataEqual(x - 1.5, [-0.5, 0.5, -1.5])
      try await assertDataEqual(1.5 - x, [0.5, -0.5, 1.5])
    }
  }

  func testMul() async throws {
    try await runInBackends {
      let x = Tensor(data: [1.0, 2.0, 3.0, 0.0], shape: [4], dtype: .float32)
      let y = Tensor(data: [-1.0, 2.0, 2.0, -3.0], shape: [4], dtype: .float32)
      try await assertDataEqual(x * y, [-1.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(x.cast(.float16) * y.cast(.float16), [-1.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(x.cast(.int64) * y.cast(.int64), [-1.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(x * 2, [2.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(x.cast(.float16) * 2, [2.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(x.cast(.int64) * 2, [2.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(2 * x, [2.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(2 * x.cast(.float16), [2.0, 4.0, 6.0, 0.0])
      try await assertDataEqual(2 * x.cast(.int64), [2.0, 4.0, 6.0, 0.0])
    }
  }

  func testDiv() async throws {
    try await runInBackends {
      let x = Tensor(data: [1.0, 2.0, 6.0, 0.0], shape: [4], dtype: .float32)
      let y = Tensor(data: [-1.0, 2.0, 2.0, 3.0], shape: [4], dtype: .float32)
      try await assertDataEqual(x / y, [-1.0, 1.0, 3.0, 0.0])
      try await assertDataEqual(x.cast(.float16) / y.cast(.float16), [-1.0, 1.0, 3.0, 0.0])
      try await assertDataEqual(x.cast(.int64) / y.cast(.int64), [-1.0, 1.0, 3.0, 0.0])
      try await assertDataEqual(x / 2, [0.5, 1.0, 3.0, 0.0])
      try await assertDataEqual(x.cast(.float16) / 2, [0.5, 1.0, 3.0, 0.0])
      try await assertDataEqual(x.cast(.int64) / 2, [0.0, 1.0, 3.0, 0.0])
      try await assertDataEqual(6 / y, [-6.0, 3.0, 3.0, 2.0])
      try await assertDataEqual(6 / y.cast(.float16), [-6.0, 3.0, 3.0, 2.0])
      try await assertDataEqual(6 / y.cast(.int64), [-6.0, 3.0, 3.0, 2.0])
      try await assertDataEqual(2 / y.cast(.int64), [-2.0, 1.0, 1.0, 0.0])

      var xGrad: Tensor?
      var yGrad: Tensor?
      let divided = (x.onGrad { g in xGrad = g }) / (y.onGrad { g in yGrad = g })
      divided.backward(Tensor(data: [-1.0, 2.0, 3.0, 4.0], shape: [4]))
      try await assertClose(xGrad!, [1.0, 1.0, 1.5, 1.3333333333])
      try await assertClose(yGrad!, [1.0, -1.0, -4.5, -0.0])
    }
  }

  func testMod() async throws {
    try await runInBackends {
      func testSimple(
        _ x: Int, _ y: Int, _ out: Int, file: StaticString = #filePath,
        line: UInt = #line
      ) async throws {
        for dtype: Tensor.DType in [.float32, .float16, .int64] {
          try await assertDataEqual(
            Tensor(data: [x], shape: [1], dtype: dtype)
              % Tensor(data: [y], shape: [1], dtype: dtype), [Float(out)], file: file,
            line: line)
        }
      }
      try await testSimple(15, 4, 3)
      try await testSimple(15, -4, -1)
      try await testSimple(-15, -4, -3)
      try await testSimple(-15, 4, 1)

      let x = Tensor(data: [0.5, 2.0, 5.0, 5.0, 0.0, -0.5, -0.5], shape: [7], dtype: .float32)
      let y = Tensor(data: [-1.0, -2.0, 1.5, 2.0, 3.0, 2.0, -2.0], shape: [7], dtype: .float32)
      try await assertDataEqual(Tensor(data: [3], shape: [1]) % Tensor(data: [2], shape: [1]), [1])

      try await assertDataEqual(x % y, [-0.5, 0.0, 0.5, 1.0, 0.0, 1.5, -0.5])
      try await assertDataEqual(
        x.cast(.float16) % y.cast(.float16), [-0.5, 0.0, 0.5, 1.0, 0.0, 1.5, -0.5])
      try await assertDataEqual(
        x.cast(.int64) % y.cast(.int64), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
      try await assertDataEqual(x % 2, [0.5, 0.0, 1.0, 1.0, 0.0, 1.5, 1.5])
      try await assertDataEqual(x.cast(.float16) % 2, [0.5, 0.0, 1.0, 1.0, 0.0, 1.5, 1.5])
      try await assertDataEqual(x.cast(.int64) % 2, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
      try await assertDataEqual(7.5 % y, [-0.5, -0.5, 0.0, 1.5, 1.5, 1.5, -0.5])
      try await assertDataEqual(7.5 % y.cast(.float16), [-0.5, -0.5, 0.0, 1.5, 1.5, 1.5, -0.5])
      try await assertDataEqual(7 % y.cast(.int64), [0, -1, 0, 1, 1, 1, -1])

      var xGrad: Tensor?
      var yGrad: Tensor?
      let modded = (x.onGrad { g in xGrad = g }) % (y.onGrad { g in yGrad = g })
      modded.backward(Tensor(data: [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], shape: [7]))
      try await assertClose(xGrad!, [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
      try await assertClose(yGrad!, [0.0, -2.0, -0, -0, -5.0, -0, -0])
    }
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

  func testComparison() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0, -2.0, 3.0], shape: [5])
    let y = Tensor(data: [1.5, 2, 2, -2, -3], shape: [5])
    try await assertDataEqual(x <= y, [1, 1, 0, 1, 0])
    try await assertDataEqual(x < y, [1, 0, 0, 0, 0])
    try await assertDataEqual(x >= y, [0, 1, 1, 1, 1])
    try await assertDataEqual(x > y, [0, 0, 1, 0, 1])
    XCTAssertEqual((x >= y).dtype, .bool)
    XCTAssertEqual((x >= y).shape, x.shape)
  }

  func testSum() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0, -2.0, 3.0, 7.0], shape: [1, 2, 3, 1], dtype: .float32)
    var xGrad: Tensor?

    func useX() -> Tensor {
      x.onGrad { xGrad = $0 }
    }

    var sum = useX().sum(axis: 2)
    XCTAssertEqual(sum.shape, [1, 2, 1])
    sum.backward(Tensor(data: [-1.0, -2.0], shape: [1, 2, 1], dtype: .float32))
    try await assertDataEqual(sum, [6.0, 8.0])
    try await assertDataEqual(xGrad!, [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0])

    sum = useX().sum(axis: 2, keepdims: true)
    XCTAssertEqual(sum.shape, [1, 2, 1, 1])
    sum.backward(Tensor(data: [-1.0, -2.0], shape: [1, 2, 1, 1], dtype: .float32))
    try await assertDataEqual(sum, [6.0, 8.0])
    try await assertDataEqual(xGrad!, [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0])

    sum = useX().sum(axis: 1)
    XCTAssertEqual(sum.shape, [1, 3, 1])
    sum.backward(Tensor(data: [-1.0, -2.0, -3.0], shape: [1, 3, 1], dtype: .float32))
    try await assertDataEqual(sum, [-1.0, 5.0, 10.0])
    try await assertDataEqual(xGrad!, [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0])

    sum = useX().sum(axis: 1, keepdims: true)
    XCTAssertEqual(sum.shape, [1, 1, 3, 1])
    sum.backward(Tensor(data: [-1.0, -2.0, -3.0], shape: [1, 1, 3, 1], dtype: .float32))
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
        sum.backward(
          Tensor(data: [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], shape: sum.shape, dtype: .float32))
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

    let ndData = Tensor(
      data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [2, 2, 3], dtype: .float32)
    let ndSum = ndData.sum(axis: 1)
    try await assertDataEqual(ndSum, [5, 7, 9, 17, 19, 21])

    // Older tests are below

    let input = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3], dtype: .float32)
    var gradA: Tensor?
    let sumA = input.onGrad({ g in gradA = g }).sum(axis: 1)
    try await assertDataEqual(sumA, [6, 15, 24])
    sumA.backward(Tensor(data: [-1, -2, -3], shape: [3], dtype: .float32))
    try await assertDataEqual(gradA!, [-1, -1, -1, -2, -2, -2, -3, -3, -3])

    var gradB: Tensor?
    let sumB = input.onGrad({ g in gradB = g }).sum(axis: 0)
    try await assertDataEqual(sumB, [12, 15, 18])
    sumB.backward(Tensor(data: [-1, -2, -3], shape: [3], dtype: .float32))
    try await assertDataEqual(gradB!, [-1, -2, -3, -1, -2, -3, -1, -2, -3])

    let input1 = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8], shape: [2, 2, 2], dtype: .float32)

    var gradC: Tensor?
    let sumC = input1.onGrad({ g in gradC = g }).sum(axis: 1)
    try await assertDataEqual(sumC, Array([1 + 3, 2 + 4, 5 + 7, 6 + 8].map { Float($0) }))
    sumC.backward(Tensor(data: [-1, -2, -3, -4], shape: [2, 2], dtype: .float32))
    try await assertDataEqual(gradC!, [-1, -2, -1, -2, -3, -4, -3, -4])

    var gradD: Tensor?
    let sumD = input1.onGrad({ g in gradD = g }).sum()
    try await assertDataEqual(sumD, [[1, 2, 3, 4, 5, 6, 7, 8].sum()])
    sumD.backward(Tensor(data: [-1], shape: [], dtype: .float32))
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
    try await runInBackends {
      for dtype: Tensor.DType in [.float16, .float32] {
        let x = Tensor(data: [1.0, 2.0, 3.0, -2.0, 3.0, 7.0], shape: [1, 2, 3, 1], dtype: dtype)
        var xGrad: Tensor?

        func useX() -> Tensor {
          x.onGrad { xGrad = $0 }
        }

        // Unbroadcasted gather along inner axis
        var out = useX().gather(
          axis: 2, indices: Tensor(data: [2, 0, 1, 2], shape: [1, 2, 2, 1], dtype: .int64))
        XCTAssertEqual(out.shape, [1, 2, 2, 1])
        out.backward(Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 2, 2, 1], dtype: dtype))
        try await assertDataEqual(out, [3.0, 1.0, 3.0, 7.0])
        try await assertDataEqual(xGrad!, [2.0, 0.0, 1.0, 0.0, 3.0, 4.0])

        // Broadcasted gather along inner axis
        out = useX().gather(
          axis: 2, indices: Tensor(data: [2, 0], shape: [2], dtype: .int64))
        XCTAssertEqual(out.shape, [1, 2, 2, 1])
        out.backward(Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 2, 2, 1], dtype: dtype))
        try await assertDataEqual(out, [3.0, 1.0, 7.0, -2.0])
        try await assertDataEqual(xGrad!, [2.0, 0.0, 1.0, 4.0, 0.0, 3.0])

        // Unbroadcasted gather along outer axis
        out = useX().gather(
          axis: 1, indices: Tensor(data: [0, 1, 0, 1, 0, 0], shape: [1, 2, 3, 1], dtype: .int64))
        XCTAssertEqual(out.shape, [1, 2, 3, 1])
        out.backward(
          Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [1, 2, 3, 1], dtype: dtype))
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
        out.backward(
          Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [1, 2, 3, 1], dtype: dtype))
        try await assertDataEqual(out, [-2.0, 3.0, 7.0, 1.0, 2.0, 3.0])
        try await assertDataEqual(xGrad!, [4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
      }
      for dtype: Tensor.DType in [.float16, .float32, .int64] {
        let transposeMe = Tensor(
          data: [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11,
            -12,
          ], shape: [2, 3, 4], dtype: dtype)
        try await assertDataEqual(
          transposeMe.t(),
          [
            1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, -1, -5, -9, -2, -6, -10, -3, -7, -11, -4, -8,
            -12,
          ])
        let buf = try await transposeMe.t().data
        XCTAssert(
          buf.buffer.allocatedSize >= transposeMe.dtype.byteSize * transposeMe.shape.product())

        // Subtle test for boolean gather
        try await assertDataEqual(transposeMe.t() == 0, (transposeMe == 0).t())
      }
    }
  }

  func testMatrixMatrixProduct() async throws {
    try await runInBackends(coreML: true) {
      // Sanity check for transposes.
      for transA in [false, true] {
        for transB in [false, true] {
          for transOut in [false, true] {
            for dtype in [Tensor.DType.float32, Tensor.DType.float16] {
              let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3], dtype: dtype)
              let y = Tensor(
                data: [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12], shape: [4, 3],
                dtype: dtype
              ).t()
              var aGrad: Tensor?
              var bGrad: Tensor?
              let a = x.onGrad { grad in aGrad = grad }
              let b = y.onGrad { grad in bGrad = grad }
              let subOut = Tensor.matmul(
                a: transA ? a.t() : a, transA: transA,
                b: transB ? b.t() : b, transB: transB, transOut: transOut)
              let out = transOut ? subOut.t() : subOut
              XCTAssertEqual(out.shape, [2, 4])
              out.backward(Tensor(data: [8, 7, 6, 5, 4, 3, 2, 1], shape: [2, 4], dtype: dtype))
              try await assertDataEqual(out, [-14, -32, -50, -68, -32, -77, -122, -167])
              try await assertDataEqual(aGrad!, [-128.0, -154.0, -180.0, -40.0, -50.0, -60.0])
              try await assertDataEqual(
                bGrad!, [24.0, 19.0, 14.0, 9.0, 36.0, 29.0, 22.0, 15.0, 48.0, 39.0, 30.0, 21.0])
            }
          }
        }
      }
      let x = Tensor(ones: [64, 128])
      let y = Tensor(ones: [128, 32])
      let z = Tensor.matmul(a: x, transA: false, b: y, transB: false, transOut: false)
      XCTAssertEqual(z.shape, [64, 32])
      try await assertDataEqual(z, [Float](repeating: 128, count: 64 * 32))

      let small = x.cast(.float16) &* (y.cast(.float16))
      let smallFloat = try await small.cast(.float32).sum().item()
      XCTAssert(smallFloat.isFinite)

      let big = (x.cast(.float16) * 1000) &* (y.cast(.float16) * 1000)
      let bigFloat = try await big.cast(.float32).sum().item()
      XCTAssert(!bigFloat.isFinite)
    }
  }

  func testBatchedMatmul() async throws {
    try await runInBackends(coreML: true) {
      for transA in [false, true] {
        for transB in [false, true] {
          for transOut in [false, true] {
            for dtype in [Tensor.DType.float32, Tensor.DType.float16] {
              let x = Tensor(rand: [3, 2, 3], dtype: dtype)
              let y = Tensor(rand: [3, 3, 4], dtype: dtype)
              var xGrad: Tensor?
              var yGrad: Tensor?
              let a = x.onGrad { grad in xGrad = grad }
              let b = y.onGrad { grad in yGrad = grad }
              let preOut = Tensor.batchedMatmul(
                a: transA ? a.t() : a, transA: transA,
                b: transB ? b.t() : b, transB: transB, transOut: transOut)
              let out = transOut ? preOut.t() : preOut
              XCTAssertEqual(out.shape, [3, 2, 4])
              let outGrad = Tensor(randLike: out)
              out.backward(outGrad)

              for i in 0..<x.shape[0] {
                let eps = Float(dtype == .float32 ? 1e-4 : 1e-2)
                var subXGrad: Tensor?
                var subYGrad: Tensor?
                let a = x[i].onGrad { grad in subXGrad = grad }
                let b = y[i].onGrad { grad in subYGrad = grad }
                let singleOut = Tensor.matmul(
                  a: transA ? a.t() : a, transA: transA,
                  b: transB ? b.t() : b, transB: transB, transOut: transOut)
                let subOut = transOut ? singleOut.t() : singleOut
                subOut.backward(outGrad[i])
                XCTAssertEqual(subOut.shape, out[i].shape)
                try await assertClose(subOut, out[i], atol: eps, rtol: eps)
                try await assertClose(subXGrad!, xGrad![i], atol: eps, rtol: eps)
                try await assertClose(subYGrad!, yGrad![i], atol: eps, rtol: eps)
              }
            }
          }
        }
      }
    }
  }

  func testMatrixVectorProduct() async throws {
    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3], dtype: .float32)
    let y = Tensor(data: [-1, -3, 2], shape: [3, 1], dtype: .float32)
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

  func testTril() async throws {
    try await assertDataEqual(
      Tensor(data: [1, 2, 3, 4, 5, 6], shape: [3, 2]).tril(), [1, 0, 3, 4, 5, 6])
    try await assertDataEqual(
      Tensor(data: [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], shape: [2, 3, 2]).tril(),
      [1, 0, 3, 4, 5, 6, -1, 0, -3, -4, -5, -6])
    try await assertDataEqual(
      Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3]).tril(), [1, 0, 0, 4, 5, 0])
    try await assertDataEqual(
      Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3]).tril(), [1, 0, 0, 4, 5, 0, 7, 8, 9])

    let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [3, 2], dtype: .float32)
    var xGrad: Tensor?
    x.onGrad { g in xGrad = g }.tril().backward(
      Tensor(data: [-1, -2, -3, -4, -5, -6], shape: [3, 2], dtype: .float32))
    try await assertDataEqual(xGrad!, [-1, 0, -3, -4, -5, -6])
  }

  func testIndexing() async throws {
    try await runInBackends {
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

      let y = Tensor(
        data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [3, 1, 4], dtype: .float32)
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
      yParam[1...2, ..., 2...3].backward(
        Tensor(data: [1, 2, 3, 4], shape: [2, 1, 2], dtype: .float32))
      XCTAssertEqual(yGrad!.shape, y.shape)
      try await assertDataEqual(yGrad!, [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4])

      let yParam1 = y.onGrad { grad in yGrad = grad }
      yParam1[..., 0, 3].backward(Tensor(data: [1, 2, 3], shape: [3], dtype: .float32))
      XCTAssertEqual(yGrad!.shape, y.shape)
      try await assertDataEqual(yGrad!, [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3])

      // High dimensional permutations.
      let bigArr = Tensor(
        data: [
          4, 8, 7, 2, 9, 6, 6, 7, 6, 8, 4, 9, 1, 9, 9, 7, 2, 3, 0, 9, 7, 6, 9, 8, 1, 3, 5, 0, 1, 5,
          9, 5, 0, 5, 4, 0, 2, 9, 8, 7, 5, 5, 7, 3, 0, 1, 5, 1, 9, 6, 3, 3, 6, 7, 1, 4, 6, 1, 6, 3,
          2, 8, 1, 2, 1, 2, 1, 8, 0, 4, 2, 7, 1, 0, 0, 7, 3, 9, 5, 5, 6, 5, 7, 3, 9, 3, 9, 0, 9, 8,
          9, 9, 0, 7, 0, 6, 2, 8, 7, 6, 8, 8, 3, 7, 8, 3, 8, 9, 4, 0, 7, 7, 0, 4, 3, 3, 9, 4, 3, 5,
        ], shape: [2, 1, 3, 5, 4], dtype: .float32)
      let perm1 = bigArr[PermuteAxes(1, 2, 3, 4, 0)]
      XCTAssertEqual(perm1.shape, [1, 3, 5, 4, 2])
      try await assertDataEqual(
        perm1,
        [
          4, 2, 8, 8, 7, 1, 2, 2, 9, 1, 6, 2, 6, 1, 7, 8, 6, 0, 8, 4, 4, 2, 9, 7, 1, 1, 9, 0, 9, 0,
          7, 7, 2, 3, 3, 9, 0, 5, 9, 5, 7, 6, 6, 5, 9, 7, 8, 3, 1, 9, 3, 3, 5, 9, 0, 0, 1, 9, 5, 8,
          9, 9, 5, 9, 0, 0, 5, 7, 4, 0, 0, 6, 2, 2, 9, 8, 8, 7, 7, 6, 5, 8, 5, 8, 7, 3, 3, 7, 0, 8,
          1, 3, 5, 8, 1, 9, 9, 4, 6, 0, 3, 7, 3, 7, 6, 0, 7, 4, 1, 3, 4, 3, 6, 9, 1, 4, 6, 3, 3, 5,
        ])
      let perm2 = bigArr[FullRange(dims: 2), PermuteAxes(2, 1, 0)]
      XCTAssertEqual(perm2.shape, [2, 1, 4, 5, 3])
      try await assertDataEqual(
        perm2,
        [
          4, 7, 5, 9, 1, 0, 6, 1, 9, 1, 0, 6, 2, 2, 6, 8, 6, 5, 6, 3, 1, 8, 5, 6, 9, 5, 7, 3, 9, 1,
          7, 9, 7, 6, 5, 5, 4, 9, 3, 9, 4, 1, 0, 8, 6, 2, 8, 3, 7, 0, 1, 9, 5, 3, 7, 0, 4, 9, 7, 3,
          2, 6, 8, 1, 9, 8, 0, 9, 4, 1, 0, 0, 3, 2, 9, 8, 5, 8, 2, 3, 3, 4, 8, 0, 0, 7, 4, 9, 8, 4,
          1, 7, 3, 1, 9, 8, 2, 9, 7, 0, 0, 3, 5, 7, 3, 2, 3, 7, 8, 0, 9, 7, 9, 7, 7, 6, 3, 5, 6, 5,
        ])
      let perm3 = bigArr[PermuteAxes(1, 0)]
      XCTAssertEqual(perm3.shape, [1, 2, 3, 5, 4])
      try await assertDataEqual(
        perm3,
        [
          4, 8, 7, 2, 9, 6, 6, 7, 6, 8, 4, 9, 1, 9, 9, 7, 2, 3, 0, 9, 7, 6, 9, 8, 1, 3, 5, 0, 1, 5,
          9, 5, 0, 5, 4, 0, 2, 9, 8, 7, 5, 5, 7, 3, 0, 1, 5, 1, 9, 6, 3, 3, 6, 7, 1, 4, 6, 1, 6, 3,
          2, 8, 1, 2, 1, 2, 1, 8, 0, 4, 2, 7, 1, 0, 0, 7, 3, 9, 5, 5, 6, 5, 7, 3, 9, 3, 9, 0, 9, 8,
          9, 9, 0, 7, 0, 6, 2, 8, 7, 6, 8, 8, 3, 7, 8, 3, 8, 9, 4, 0, 7, 7, 0, 4, 3, 3, 9, 4, 3, 5,
        ])
    }
  }

  func testChunk() async throws {
    for i in 1..<10 {
      for count in 1...i {
        if i % count != 0 {
          continue
        }
        let input = Tensor(range: 0..<i)
        let split = input.chunk(axis: 0, count: count)
        XCTAssertEqual(split.count, count)
        if i % count == 0 {
          XCTAssert(split.allSatisfy { $0.shape[0] == split[0].shape[0] })
        } else {
          XCTAssert(split[..<(split.count - 1)].allSatisfy { $0.shape[0] == split[0].shape[0] })
          XCTAssert(split.map { $0.shape[0] }.sum() == i)
        }
      }
    }
  }

  func testElemwise() async throws {
    func testF(
      input: [Float], output: [Float], grad: [Float], tol: Float = 1e-4, _ op: (Tensor) -> Tensor
    )
      async throws
    {
      try await runInBackends {
        for dtype in [Tensor.DType.float32, Tensor.DType.float16] {
          var actualGrad: Tensor?
          let tensorIn = Tensor(data: input, shape: [input.count], dtype: dtype) { g in
            actualGrad = g
          }
          assert(tensorIn.needsGrad, "\(tensorIn.dtype) \(tensorIn.needsGrad)")
          let actualOut = op(tensorIn)
          let thisTol = tol * Float(dtype == .float32 ? 1.0 : 100.0)
          try await assertClose(
            actualOut, Tensor(data: output, shape: [output.count]), atol: thisTol, rtol: thisTol)
          actualOut.backward(Tensor(onesLike: actualOut))
          try await assertClose(
            actualGrad!, Tensor(data: grad, shape: [output.count]), atol: thisTol, rtol: thisTol)
        }
      }
    }

    try await testF(
      input: [-1, -2, -3, 1, 2, 3],
      output: [1, 4, 9, 1, 4, 9],
      grad: [-2, -4, -6, 2, 4, 6]
    ) { $0.pow(2) }
    try await testF(
      input: [-1, -2, -3, 1, 2, 3],
      output: [1.0 / -1.0, 1.0 / -2.0, 1.0 / -3.0, 1.0, 1.0 / 2.0, 1.0 / 3.0].map({ Float($0) }),
      grad: [-1.0, -0.25, -0.11111111, -1.0, -0.25, -0.11111111]
    ) { $0.pow(-1) }
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
      ],
      tol: 1e-3
    ) { $0.gelu() }
    try await testF(
      input: [-100, -2, 2, 100],
      output: [100, 2, 2, 100],
      grad: [-1, -1, 1, 1]
    ) { $0.abs() }
    try await testF(
      input: [-100, -2, -0.1, 0.1, 1, 2, 3, 100],
      output: [0, 0, 0, 0.1, 1, 2, 3, 100],
      grad: [0, 0, 0, 1, 1, 1, 1, 1]
    ) { $0.relu() }
    try await testF(
      input: [-100, 100, 0, 3.0],
      output: [0, 1, 0.5, 0.9525741338729858],
      grad: [0, 0, 0.25, Float(0.9525741338729858 * (1 - 0.9525741338729858))]
    ) { $0.sigmoid() }
    try await testF(
      input: [1, Float(M_E), Float(M_E * M_E), 10000.0],
      output: [0, 1, 2, log(10000.0)],
      grad: [1, 1 / Float(M_E), 1 / Float(M_E * M_E), Float(1 / 10000.0)]
    ) { $0.log() }
    try await testF(
      input: [Float](repeating: 9.0, count: 1024),
      output: [Float](repeating: 9.0, count: 1024),
      grad: [Float](repeating: 1.0, count: 1024)
    ) { $0.exp().log() }
    try await testF(
      input: [-1, -2, -3, 1.1, 2, 3],
      output: [1, 1, 1, 1.1, 2, 3],
      grad: [0, 0, 0, 1, 1, 1]
    ) { $0.clamp(min: 1) }
    try await testF(
      input: [-1, -2, -3, 0.9, 1.1, 2, 3],
      output: [-1, -2, -3, 0.9, 1, 1, 1],
      grad: [1, 1, 1, 1, 0, 0, 0]
    ) { $0.clamp(max: 1) }
    try await testF(
      input: [-1, -2, -3, 0.9, 1.1, 2, 3],
      output: [-1, -1.5, -1.5, 0.9, 1, 1, 1],
      grad: [1, 0, 0, 1, 0, 0, 0]
    ) { $0.clamp(min: -1.5, max: 1) }
  }

  func testMinMax() async throws {
    let input = Tensor(data: [1, 10, 2, 7, 8, 9, 6, 4, 5], shape: [3, 3], dtype: .float32)
    var gradA: Tensor?
    let maxA = input.onGrad({ g in gradA = g }).max(axis: 1)
    try await assertDataEqual(maxA, [10, 9, 6])
    maxA.backward(Tensor(data: [-1, -2, -3], shape: [3], dtype: .float32))
    try await assertDataEqual(gradA!, [0, -1, 0, 0, 0, -2, -3, 0, 0])

    var gradB: Tensor?
    let maxB = input.onGrad({ g in gradB = g }).max(axis: 0)
    try await assertDataEqual(maxB, [7, 10, 9])
    maxB.backward(Tensor(data: [-1, -2, -3], shape: [3], dtype: .float32))
    try await assertDataEqual(gradB!, [0, -2, 0, -1, 0, -3, 0, 0, 0])

    var gradC: Tensor?
    let minC = input.onGrad({ g in gradC = g }).min(axis: 0)
    try await assertDataEqual(minC, [1, 4, 2])
    minC.backward(Tensor(data: [-1, -2, -3], shape: [3], dtype: .float32))
    try await assertDataEqual(gradC!, [-1, 0, -3, 0, 0, 0, 0, -2, 0])

    var gradD: Tensor?
    let maxD = input.onGrad({ g in gradD = g }).max()
    XCTAssertEqual(maxD.shape, [])
    try await assertDataEqual(maxD, [10])
    maxD.backward(Tensor(data: [-1], shape: [], dtype: .float32))
    try await assertDataEqual(gradD!, [0, -1, 0, 0, 0, 0, 0, 0, 0])

    var gradE: Tensor?
    let maxE = input.onGrad({ g in gradE = g }).min()
    XCTAssertEqual(maxE.shape, [])
    try await assertDataEqual(maxE, [1])
    maxE.backward(Tensor(data: [-1], shape: [], dtype: .float32))
    try await assertDataEqual(gradE!, [-1.0, 0, 0, 0, 0, 0, 0, 0, 0])

    XCTAssertEqual(input.max(keepdims: true).shape, [1, 1])
    XCTAssertEqual(input.max(axis: 0, keepdims: true).shape, [1, 3])
    XCTAssertEqual(input.max(axis: 1, keepdims: true).shape, [3, 1])
  }

  func testExpandAndRepeat() async throws {
    try await runInBackends {
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

  func testWhen() async throws {
    let x = Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [4])
    let y = Tensor(data: [-1.0, -2.0, -3.0, -4.0], shape: [4])
    let mask = Tensor(data: [false, true, true, false], shape: [4])
    var xGrad: Tensor?
    var yGrad: Tensor?
    let out = mask.when(isTrue: x.onGrad { g in xGrad = g }, isFalse: y.onGrad { g in yGrad = g })
    try await assertDataEqual(out, [-1.0, 2.0, 3.0, -4.0])
    out.backward(Tensor(data: [5.0, 6.0, 7.0, 8.0], shape: [4]))
    try await assertDataEqual(xGrad!, [0, 6.0, 7.0, 0.0])
    try await assertDataEqual(yGrad!, [5.0, 0.0, 0.0, 8.0])
  }

  func testOneHot() async throws {
    try await assertDataEqual(Tensor(oneHot: 3, count: 5), [0, 0, 0, 1, 0])
    try await assertDataEqual(Tensor(oneHot: [3, 1], count: 5), [0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
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

      init(inSize: Int, outSize: Int) {
        super.init()
        weight = Tensor(rand: [inSize, outSize])
        bias = Tensor(rand: [outSize])
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

      instance.optional = Tensor(zeros: [7])
      params = instance.parameters
      XCTAssertEqual(params.count, 3)
      XCTAssertEqual(params[1].0, "optional")
      XCTAssertEqual(params[1].1.data!.shape, [7])
      instance.optional = nil
    }

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

    let net1 = Network()
    try net1.loadState(try await net.state())
    for ((name1, param1), (name2, param2)) in zip(net.parameters, net1.parameters) {
      XCTAssertEqual(param1.data!.dtype, param2.data!.dtype)
      XCTAssertEqual(param1.data!.shape, param2.data!.shape)
      try await assertClose(param1.data!, param2.data!, "params differ for names \(name1)/\(name2)")
    }
  }

  func testAdam() async throws {
    class Linear: Trainable {
      @Param(name: "weight") var weight: Tensor
      @Param(name: "bias") var bias: Tensor

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

  func testRandom() async throws {
    try await runInBackends {
      let x = Tensor(randn: [1000, 100])
      let xMean = try await x.mean().item()
      let xStd = try await x.variance().sqrt().item()
      XCTAssert(abs(xMean) < 0.03, "unexpected mean of normal: \(xMean)")
      XCTAssert(abs(xStd - 1) < 0.03, "unexpected standard deviation of normal: \(xStd)")

      let z = Tensor(rand: [1000, 100])
      let zMean = try await z.mean().item()
      let zStd = try await z.variance().sqrt().item()
      XCTAssert(abs(zMean - 0.5) < 0.03, "unexpected mean of uniform: \(zMean)")
      XCTAssert(abs(zStd - 0.288675135) < 0.03, "unexpected standard deviation of uniform: \(zStd)")

      do {
        let rng = try await Backend.current.defaultRandom()
        let state = try await rng.save()
        let a = Tensor(randn: [1234])
        let _ = try await a.data  // avoid a race condition
        try await rng.restore(state)
        let b = Tensor(randn: [1234])
        try await assertDataEqual(a, b)
      } catch BackendError.notImplemented(_) {
      }
    }
  }

  func testNoGrad() async throws {
    let x = Tensor(ones: [3]).onGrad { grad in assert(false) }
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

  func testCacheBackendForBackward() async throws {
    // If any operations are called on this backend, it will raise an error.
    let unusedBackend = Backend()

    func testArithmetic() async throws {
      return try await runInBackends {
        var xGrad: Tensor?
        let x = Tensor(ones: [1]).onGrad { grad in xGrad = grad }
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
        let x = Tensor(ones: [3, 3]).onGrad { grad in xGrad = grad }
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
        let x = Tensor(ones: [3, 3]).onGrad { grad in xGrad = grad }
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
        let x = Tensor(ones: [3, 3]).onGrad { grad in xGrad = grad }
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
        let x = Tensor(ones: [3, 6]).onGrad { grad in xGrad = grad }
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
        let x = Tensor(ones: [3, 6]).onGrad { grad in xGrad = grad }
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
        let x = Tensor(ones: [3, 6]).onGrad { grad in xGrad = grad }
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

  func testConv2D() async throws {
    try await runInBackends {
      guard let fileURL = Bundle.module.url(forResource: "conv2d", withExtension: "json.gz") else {
        XCTFail("Missing file: conv2d.json.gz")
        return
      }
      let data = try (try Data(contentsOf: fileURL)).gunzipped()

      struct ConvTestCaseConfig: Codable {
        let kernelSize: [Int]
        let imageSize: [Int]
        let stride: [Int]
        let dilation: [Int]
        let padding: [Int]
        let groups: Int
      }
      struct ConvTestCase: Codable {
        let conv: ConvTestCaseConfig
        let outShape: [Int]
        let output: [Float]
        let imageGrad: [Float]
        let kernelGrad: [Float]
      }
      typealias JSONData = [ConvTestCase]
      let decodedData = try JSONDecoder().decode(JSONData.self, from: data)

      for testCase in decodedData {
        let c = testCase.conv
        func createConv(_ channelsLast: Bool) throws -> Conv2DConfig {
          try Conv2DConfig(
            inChannels: c.imageSize[2],
            outChannels: c.kernelSize[2],
            kernelSize: SpatialDim2D(x: c.kernelSize[1], y: c.kernelSize[0]),
            imageSize: SpatialDim2D(x: c.imageSize[1], y: c.imageSize[0]),
            stride: SpatialDim2D(x: c.stride[1], y: c.stride[0]),
            dilation: SpatialDim2D(x: c.dilation[1], y: c.dilation[0]),
            padding: Conv2DConfig.Padding(
              before: SpatialDim2D(x: c.padding[1], y: c.padding[0]),
              after: SpatialDim2D(x: c.padding[1], y: c.padding[0])),
            groups: c.groups,
            channelsLast: channelsLast)
        }
        let conv = try createConv(false)
        let kernelShape = conv.kernelTensorShape()
        let imageShape = conv.imageTensorShape(batch: 1)
        let kernel = Tensor(range: 0..<kernelShape.product(), dtype: .float32).reshape(kernelShape)
        let image = -Tensor(range: 0..<imageShape.product(), dtype: .float32).reshape(imageShape)
        var imageGrad: Tensor?
        var kernelGrad: Tensor?
        let output = Tensor.conv2D(
          conv, image: image.onGrad { g in imageGrad = g },
          kernel: kernel.onGrad { g in kernelGrad = g })
        XCTAssertEqual(output.shape, testCase.outShape, "\(conv)")
        try await assertDataEqual(output, testCase.output, "\(conv)")

        let outGrad = (Tensor(range: 0..<output.shape.product()) * 4).reshape(output.shape).cast(
          .float32)
        output.backward(outGrad)
        try await assertDataEqual(imageGrad!, testCase.imageGrad, "\(conv)")
        try await assertClose(kernelGrad!, testCase.kernelGrad, "\(conv)")

        let convCLast = try createConv(true)
        var transImageGrad: Tensor?
        var transKernelGrad: Tensor?
        let outputCLast = Tensor.conv2D(
          convCLast, image: image.onGrad({ g in transImageGrad = g })[PermuteAxes(0, 2, 3, 1)],
          kernel: kernel.onGrad { g in transKernelGrad = g })[PermuteAxes(0, 3, 1, 2)]
        try await assertDataEqual(output, outputCLast)
        outputCLast.backward(outGrad)
        try await assertDataEqual(transImageGrad!, imageGrad!)
        try await assertClose(transKernelGrad!, kernelGrad!)

        var batchedImageGrad: Tensor?
        var batchedKernelGrad: Tensor?
        let batchedInput = image.onGrad({ g in batchedImageGrad = g })
        let batchedOutput = Tensor.conv2D(
          conv, image: Tensor(concat: [Tensor(zerosLike: batchedInput), batchedInput], axis: 0),
          kernel: kernel.onGrad { g in batchedKernelGrad = g })
        batchedOutput.backward(outGrad.repeating(axis: 0, count: 2))
        try await assertClose(batchedImageGrad!, imageGrad!)
        try await assertClose(batchedKernelGrad!, kernelGrad!)
      }
    }
  }

  func testConv2DTransposeGrads() async throws {
    try await runInBackends {
      let conv = try Conv2DConfig(
        inChannels: 6, outChannels: 4, kernelSize: .init(x: 3, y: 2), imageSize: .init(x: 8, y: 9),
        stride: .init(x: 2, y: 1), dilation: .init(x: 2, y: 2),
        padding: .init(before: .init(x: 1, y: 0), after: .init(x: 2, y: 1)), groups: 2,
        channelsLast: false)
      let input = Tensor(rand: conv.outputTensorShape(batch: 2))
      let kernel = Tensor(rand: conv.kernelTensorShape())
      var inputGrad: Tensor?
      var kernelGrad: Tensor?
      let output = Tensor.conv2DTranspose(
        conv, image: input.onGrad { g in inputGrad = g },
        kernel: kernel.onGrad { g in kernelGrad = g })
      let outGrad = Tensor(randLike: output)
      output.backward(outGrad)

      let approxKernelGrad = try await estimateGradient(delta: 0.5, input: kernel, outGrad: outGrad)
      { x in
        Tensor.conv2DTranspose(conv, image: input, kernel: x)
      }
      try await assertClose(approxKernelGrad, kernelGrad!)

      let approxInputGrad = try await estimateGradient(delta: 0.5, input: input, outGrad: outGrad) {
        x in
        Tensor.conv2DTranspose(conv, image: x, kernel: kernel)
      }
      try await assertClose(approxInputGrad, inputGrad!)
    }
  }

  func testConv1D() async throws {
    try await runInBackends {
      let conv1d = try Conv1DConfig(
        inChannels: 6, outChannels: 4, kernelSize: .init(x: 2), imageSize: .init(x: 8),
        stride: .init(x: 2), dilation: .init(x: 3),
        padding: .init(before: .init(x: 1), after: .init(x: 2)), groups: 2,
        channelsLast: false)
      let conv2d = try Conv2DConfig(
        inChannels: 6, outChannels: 4, kernelSize: .init(x: 2, y: 1), imageSize: .init(x: 8, y: 1),
        stride: .init(x: 2, y: 1), dilation: .init(x: 3, y: 1),
        padding: .init(before: .init(x: 1, y: 0), after: .init(x: 2, y: 0)), groups: 2,
        channelsLast: false)
      let input = Tensor(rand: conv1d.imageTensorShape(batch: 2))
      let kernel = Tensor(rand: conv1d.kernelTensorShape())
      var inputGrad: Tensor?
      var kernelGrad: Tensor?
      let output = Tensor.conv1D(
        conv1d, image: input.onGrad { g in inputGrad = g },
        kernel: kernel.onGrad { g in kernelGrad = g })
      let outGrad = Tensor(randLike: output)
      output.backward(outGrad)

      var inputGrad2D: Tensor?
      var kernelGrad2D: Tensor?
      let image2D = input.onGrad { g in inputGrad2D = g }.unsqueeze(axis: -2)
      let kernel2D = kernel.onGrad { g in kernelGrad2D = g }.unsqueeze(axis: -2)
      let output2DRaw = Tensor.conv2D(conv2d, image: image2D, kernel: kernel2D)
      let output2D = output2DRaw.squeeze(axis: -2)
      output2D.backward(outGrad)

      try await assertClose(output2D, output)
      try await assertClose(inputGrad2D!, inputGrad!)
      try await assertClose(kernelGrad2D!, kernelGrad!)
    }
  }

  func testGroupNorm() async throws {
    let gnCfirst = GroupNorm(groupCount: 4, channelCount: 16, channelsLast: false)
    let gnClast = GroupNorm(groupCount: 4, channelCount: 16, channelsLast: true)
    let inputCfirst = Tensor(rand: [2, 16, 10, 10])
    let inputClast = inputCfirst.move(axis: 1, to: -1)
    let outputCfirst = gnCfirst(inputCfirst)
    let outputClast = gnClast(inputClast).move(axis: -1, to: 1)
    try await assertClose(outputCfirst, outputClast)
  }
}

func estimateGradient(delta: Float = 1e-4, input: Tensor, outGrad: Tensor, fn: (Tensor) -> Tensor)
  async throws -> Tensor
{
  var result = [Float](repeating: 0, count: input.shape.product())
  for i in 0..<result.count {
    let bias = Tensor(oneHot: i, count: result.count).reshape(as: input).cast(as: input) * delta
    let o1: Float = try await (fn(input - bias) * outGrad).sum().item()
    let o2: Float = try await (fn(input + bias) * outGrad).sum().item()
    result[i] = (o2 - o1) / (2 * delta)
  }
  return Tensor(data: result, shape: input.shape, dtype: input.dtype)
}

func assertClose(
  _ x: Tensor, _ y: Tensor, _ msg: String? = nil, atol: Float = 1e-4, rtol: Float = 1e-4,
  file: StaticString = #filePath,
  line: UInt = #line
) async throws {
  XCTAssertEqual(x.shape, y.shape)
  var allGood = true
  let xData = try await x.floats()
  let yData = try await y.floats()
  for (a, b) in zip(xData, yData) {
    if a.isNaN != b.isNaN || (abs(a - b) > atol && (b == 0 || abs(a / b - 1) > rtol)) {
      allGood = false
    }
  }
  if let msg = msg {
    XCTAssert(
      allGood, "tensors \(xData) and \(yData) are not equal: \(msg)", file: file, line: line)
  } else {
    XCTAssert(allGood, "tensors \(xData) and \(yData) are not equal", file: file, line: line)
  }
}

func assertClose(
  _ x: Tensor, _ yData: [Float], _ msg: String? = nil, atol: Float = 1e-4, rtol: Float = 1e-4,
  file: StaticString = #filePath,
  line: UInt = #line
) async throws {
  XCTAssertEqual(x.shape.product(), yData.count)
  var allGood = true
  let xData = try await x.floats()
  for (a, b) in zip(xData, yData) {
    if a.isNaN != b.isNaN || (abs(a - b) > atol && (b == 0 || abs(a / b - 1) > rtol)) {
      allGood = false
    }
  }
  if let msg = msg {
    XCTAssert(
      allGood, "tensors \(xData) and \(yData) are not equal: \(msg)", file: file, line: line)
  } else {
    XCTAssert(allGood, "tensors \(xData) and \(yData) are not equal", file: file, line: line)
  }
}

func assertDataEqual(
  _ x: Tensor, _ y: [Float], _ msg: String? = nil, file: StaticString = #filePath,
  line: UInt = #line
) async throws {
  let data = try await x.floats()
  if let msg = msg {
    XCTAssertEqual(data, y, msg, file: file, line: line)
  } else {
    XCTAssertEqual(data, y, file: file, line: line)
  }
}

func assertDataEqual(
  _ x: Tensor, _ y: Tensor, file: StaticString = #filePath, line: UInt = #line
) async throws {
  let data = try await x.floats()
  let data1 = try await y.floats()
  XCTAssertEqual(data, data1, file: file, line: line)
}
