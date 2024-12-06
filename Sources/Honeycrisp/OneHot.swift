import HCBacktrace

extension Tensor {

  public convenience init(
    oneHot: Int, count: Int, dtype: DType = .float32, function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) {
    Backtrace.record(function: function, file: file, line: line) {
      #alwaysAssert(oneHot >= 0 && oneHot < count, "oneHot \(oneHot) out of range [0, \(count))")
    }
    var data = Array(repeating: Float(0), count: count)
    data[oneHot] = 1
    self.init(data: data, shape: [count], dtype: dtype)
  }

  public convenience init(
    oneHot: [Int], count: Int, dtype: DType = .float32, function: StaticString = #function,
    file: StaticString = #file, line: UInt = #line
  ) {
    var data = Array(repeating: Float(0), count: count * oneHot.count)
    Backtrace.record(function: function, file: file, line: line) {
      for (i, idx) in oneHot.enumerated() {
        #alwaysAssert(idx >= 0 && idx < count, "oneHot index \(idx) out of range [0, \(count))")
        data[i * count + idx] = 1
      }
    }
    self.init(data: data, shape: [oneHot.count, count], dtype: dtype)
  }

}
