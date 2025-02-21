import Gzip
import XCTest

@testable import Honeycrisp

public func assertClose(
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

public func assertClose(
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
    if let msg = msg {
      XCTAssert(
        allGood, "tensors \(xData) and \(yData) are not equal: \(msg)", file: file, line: line)
    } else {
      XCTAssert(allGood, "tensors \(xData) and \(yData) are not equal", file: file, line: line)
    }
  }
}

public func assertCloseToIdentity(
  _ x: Tensor, _ msg: String? = nil, atol: Float = 1e-4, rtol: Float = 1e-4,
  file: StaticString = #filePath,
  line: UInt = #line
) async throws {
  XCTAssert(x.shape.count >= 2)
  XCTAssertEqual(x.shape[x.shape.count - 2], x.shape[x.shape.count - 1])
  let eye = Tensor(identity: x.shape.last!, dtype: x.dtype).expand(as: x)
  try await assertClose(x, eye, msg, atol: atol, rtol: rtol)
}

public func assertDataEqual(
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

public func assertDataEqual64(
  _ x: Tensor, _ y: [Int64], _ msg: String? = nil, file: StaticString = #filePath,
  line: UInt = #line
) async throws {
  let data = try await x.int64s()
  if let msg = msg {
    XCTAssertEqual(data, y, msg, file: file, line: line)
  } else {
    XCTAssertEqual(data, y, file: file, line: line)
  }
}

public func assertDataEqual(
  _ x: Tensor, _ y: Tensor, file: StaticString = #filePath, line: UInt = #line
) async throws {
  let data = try await x.floats()
  let data1 = try await y.floats()
  XCTAssertEqual(data, data1, file: file, line: line)
}
