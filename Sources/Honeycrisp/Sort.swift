import HCBacktrace

extension Tensor {
  @recordCaller
  private func _argsort(axis origAxis: Int, descending: Bool = false, stable: Bool = false)
    -> Tensor
  {
    let axis = positiveAxis(origAxis)
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.argsort(
        try await t.data,
        dims: t.reduceDims(axis),
        descending: descending,
        stable: stable,
        dtype: t.dtype
      )
    }
    return Tensor(dataTask: newData, shape: shape, dtype: .int64)
  }
}
