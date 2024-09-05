extension Tensor {
  public static func outer(_ a: Tensor, _ b: Tensor, backend: Backend? = nil) -> Tensor {
    assert(
      a.shape.count == 1 && b.shape.count == 1,
      "invalid shapes for outer product: \(a.shape), \(b.shape)")
    assert(a.dtype == b.dtype, "dtype mismatch for outer product: \(a.dtype), \(b.dtype)")
    return matmul(
      a: a.reshape([a.shape[0], 1]), transA: false, b: b.reshape([1, b.shape[0]]), transB: false,
      transOut: false, backend: backend)
  }

  public static func matmul(
    a: Tensor, transA: Bool, b: Tensor, transB: Bool, transOut: Bool, backend: Backend? = nil
  )
    -> Tensor
  {
    assert(
      a.shape.count == 2 && b.shape.count == 2,
      "invalid shapes for matmul: \(a.shape), \(b.shape)")
    let aShape = transA ? [a.shape[1], a.shape[0]] : a.shape
    let bShape = transB ? [b.shape[1], b.shape[0]] : b.shape
    assert(
      aShape[1] == bShape[0], "shape mismatch for matmul (with transposes): \(aShape), \(bShape)")
    let outShape = transOut ? [bShape[1], aShape[0]] : [aShape[0], bShape[1]]
    let backend = backend ?? Backend.defaultBackend
    let newData = Task {
      let (dataA, dataB) = try await backend.waitForData(await a.data, await b.data)
      return try await backend.execute { handle in
        try handle.matmul(
          a: dataA, transA: transA, b: dataB, transB: transB, transOut: transOut, rows: aShape[0],
          inner: aShape[1], cols: bShape[1], dtype: a.dtype)
      }
    }
    if !a.needsGrad && !b.needsGrad {
      return Tensor(dataTask: newData, shape: outShape, dtype: a.dtype)
    } else {
      let handleA = a.saveForBackward()
      let handleB = b.saveForBackward()
      return Tensor(dataTask: newData, shape: outShape, dtype: a.dtype) { grad in
        try handleA.backward(
          matmul(a: grad, transA: false, b: b.noGrad(), transB: true, transOut: false))
        try handleB.backward(
          matmul(a: a.noGrad(), transA: true, b: grad, transB: false, transOut: false))
      }
    }
  }

  public static func &* (_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    return matmul(a: lhs, transA: false, b: rhs, transB: false, transOut: false)
  }
}
