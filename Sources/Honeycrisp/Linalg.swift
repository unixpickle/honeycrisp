import HCBacktrace

extension Tensor {

  /// Create an identity matrix of the given size.
  public convenience init(
    identity count: Int, dtype: DType = .float32, function: StaticString = #function,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    let result = Backtrace.record(function: function, file: file, line: line) {
      #alwaysAssert(count >= 0, "identity size must not be negative, but got \(count)")
      let idxs = Tensor(data: 0..<count)
      return (idxs == idxs[..., NewAxis()]).cast(dtype)
    }
    self.init(dataTask: result.dataTask, shape: [count, count], dtype: dtype)
  }

  /// Create a matrix with the given diagonal elements.
  ///
  /// If offset > 0, then the elements are put above the diagonal.
  /// If offset < 0, then the elements are put below the diagonal.
  public convenience init(
    diagonal: Tensor, offset: Int = 0, function: StaticString = #function,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    let result = Backtrace.record(function: function, file: file, line: line) {
      #alwaysAssert(
        diagonal.shape.count == 1, "diagonal must be 1-D, but got shape \(diagonal.shape)")
      let matrixSize = diagonal.shape[0] + (offset < 0 ? -offset : offset)
      var indices = Tensor(data: 0..<diagonal.shape[0], dtype: .int64) * (matrixSize + 1)
      if offset > 0 {
        indices = indices + offset
      } else if offset < 0 {
        indices = indices - matrixSize * offset
      }
      return diagonal.scatter(
        axis: 0, count: matrixSize * matrixSize, indices: indices, indicesAreUnique: true
      ).reshape([matrixSize, matrixSize])
    }
    self.init(dataTask: result.dataTask, shape: result.shape, dtype: result.dtype)
  }

  @recordCaller
  private static func _outer(_ a: Tensor, _ b: Tensor) -> Tensor {
    #alwaysAssert(
      a.shape.count == 1 && b.shape.count == 1,
      "invalid shapes for outer product: \(a.shape), \(b.shape)")
    #alwaysAssert(a.dtype == b.dtype, "dtype mismatch for outer product: \(a.dtype), \(b.dtype)")
    return matmul(
      a: a.reshape([a.shape[0], 1]), transA: false, b: b.reshape([1, b.shape[0]]), transB: false,
      transOut: false)
  }

  @recordCaller
  private static func _matmul(
    a: Tensor, transA: Bool, b: Tensor, transB: Bool, transOut: Bool, aGradBackend: Backend? = nil,
    bGradBackend: Backend? = nil
  )
    -> Tensor
  {
    #alwaysAssert(
      a.shape.count == 2 && b.shape.count == 2,
      "invalid shapes for matmul: \(a.shape), \(b.shape)")
    #alwaysAssert(a.dtype == b.dtype, "mismatched dtypes for matmul: \(a.dtype) and \(b.dtype)")
    let aShape = transA ? [a.shape[1], a.shape[0]] : a.shape
    let bShape = transB ? [b.shape[1], b.shape[0]] : b.shape
    #alwaysAssert(
      aShape[1] == bShape[0], "shape mismatch for matmul (with transposes): \(aShape), \(bShape)")
    let outShape = transOut ? [bShape[1], aShape[0]] : [aShape[0], bShape[1]]
    let backend = Backend.current
    let newData = createDataTask(a, b) { a, b in
      try await backend.matmul(
        a: try await a.data, transA: transA, b: try await b.data, transB: transB,
        transOut: transOut, rows: aShape[0],
        inner: aShape[1], cols: bShape[1], dtype: a.dtype)
    }
    if !Tensor.isGradEnabled || (!a.needsGrad && !b.needsGrad) {
      return Tensor(dataTask: newData, shape: outShape, dtype: a.dtype)
    } else {
      let handleA = a.saveForBackward()
      let handleB = b.saveForBackward()
      return Tensor(dataTask: newData, shape: outShape, dtype: a.dtype) { grad in
        handleA.backward(aGradBackend ?? backend) {
          matmul(a: grad, transA: transOut, b: b.noGrad(), transB: !transB, transOut: transA)
        }
        handleB.backward(bGradBackend ?? backend) {
          matmul(a: a.noGrad(), transA: !transA, b: grad, transB: transOut, transOut: transB)
        }
      }
    }
  }

  @recordCaller
  private static func _batchedMatmul(
    a: Tensor, transA: Bool, b: Tensor, transB: Bool, transOut: Bool
  )
    -> Tensor
  {
    #alwaysAssert(
      a.shape.count > 2 && b.shape.count > 2
        && a.shape[..<(a.shape.count - 2)] == b.shape[..<(b.shape.count - 2)],
      "invalid shapes for batched matmul: \(a.shape), \(b.shape)")
    #alwaysAssert(
      a.dtype == b.dtype, "mismatched dtypes for batched matmul: \(a.dtype) and \(b.dtype)")
    let batchShape: [Int] = Array(a.shape[..<(a.shape.count - 2)])
    let d0 = a.shape.count - 2
    let d1 = a.shape.count - 1
    let aShape = transA ? [a.shape[d1], a.shape[d0]] : [a.shape[d0], a.shape[d1]]
    let bShape = transB ? [b.shape[d1], b.shape[d0]] : [b.shape[d0], b.shape[d1]]
    #alwaysAssert(
      aShape[1] == bShape[0],
      "shape mismatch for batched matmul: \(a.shape) (trans=\(transA)), \(b.shape) (trans\(transB))"
    )
    let outShape = batchShape + (transOut ? [bShape[1], aShape[0]] : [aShape[0], bShape[1]])
    let backend = Backend.current
    let newData = createDataTask(a, b) { a, b in
      return try await backend.batchedMatmul(
        matrixCount: batchShape.product(), a: try await a.data, transA: transA, b: try await b.data,
        transB: transB, transOut: transOut, rows: aShape[0], inner: aShape[1], cols: bShape[1],
        dtype: a.dtype)
    }
    if !Tensor.isGradEnabled || (!a.needsGrad && !b.needsGrad) {
      return Tensor(dataTask: newData, shape: outShape, dtype: a.dtype)
    } else {
      let handleA = a.saveForBackward()
      let handleB = b.saveForBackward()
      return Tensor(dataTask: newData, shape: outShape, dtype: a.dtype) { grad in
        handleA.backward(backend) {
          batchedMatmul(
            a: grad, transA: transOut, b: b.noGrad(), transB: !transB, transOut: transA)
        }
        handleB.backward(backend) {
          batchedMatmul(
            a: a.noGrad(), transA: !transA, b: grad, transB: transOut, transOut: transB)
        }
      }
    }
  }

  public static func &* (_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    if lhs.shape.count <= 2 {
      matmul(a: lhs, transA: false, b: rhs, transB: false, transOut: false)
    } else {
      batchedMatmul(a: lhs, transA: false, b: rhs, transB: false, transOut: false)
    }
  }

  @recordCaller
  private func _tril(offset: Int = 0) -> Tensor {
    triangular(upper: false, offset: offset)
  }

  @recordCaller
  private func _triu(offset: Int = 0) -> Tensor {
    triangular(upper: true, offset: offset)
  }

  private func triangular(upper: Bool, offset: Int = 0) -> Tensor {
    #alwaysAssert(shape.count >= 2, "tensor of shape \(shape) is not a matrix")
    let backend = Backend.current
    let newData = createDataTask { t in
      return try await backend.triangular(
        try await t.data,
        batch: t.shape[..<(t.shape.count - 2)].product(),
        rows: t.shape[t.shape.count - 2],
        cols: t.shape[t.shape.count - 1],
        upper: upper,
        offset: offset,
        dtype: t.dtype
      )
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    } else {
      let handle = saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        handle.backward(backend) { grad.triangular(upper: upper, offset: offset) }
      }
    }
  }

  @recordCaller
  private func _qrDecomposition(full: Bool = false) -> (q: Tensor, r: Tensor) {
    #alwaysAssert(shape.count >= 2, "tensor of shape \(shape) is not a matrix")
    let batchShape = shape[..<(shape.count - 2)]
    let batch = batchShape.product()
    let rows = shape[shape.count - 2]
    let cols = shape[shape.count - 1]
    let full = full || rows <= cols
    let qShape = Array(batchShape + (full ? [rows, rows] : [rows, cols]))
    let rShape = Array(batchShape + (full ? [rows, cols] : [cols, cols]))
    let backend = Backend.current
    let newData = createDataTask { t in
      return try await backend.qrDecomposition(
        try await t.data,
        batch: batch,
        rows: rows,
        cols: cols,
        full: full,
        dtype: t.dtype
      )
    }
    #alwaysAssert(
      !needsGrad || !Tensor.isGradEnabled, "QR decomposition does not currently support gradients")
    return (
      q: Tensor(dataTask: Task { try await newData.value.q }, shape: qShape, dtype: dtype),
      r: Tensor(dataTask: Task { try await newData.value.r }, shape: rShape, dtype: dtype)
    )
  }

}
