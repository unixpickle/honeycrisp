import HCBacktrace

extension Tensor {

  @recordCaller
  private func _clamp<T: NumericTensorElement>(min: T) -> Tensor {
    return clampInternal(T.self, minScalar: min)
  }

  @recordCaller
  private func _clamp<T: NumericTensorElement>(max: T) -> Tensor {
    return clampInternal(T.self, maxScalar: max)
  }

  @recordCaller
  private func _clamp<T: NumericTensorElement>(min: T, max: T) -> Tensor {
    return clampInternal(T.self, minScalar: min, maxScalar: max)
  }

  @recordCaller
  private func _clamp(min: Tensor) -> Tensor {
    return clampInternal(Float.self, minTensor: min)
  }

  @recordCaller
  private func _clamp(max: Tensor) -> Tensor {
    return clampInternal(Float.self, maxTensor: max)
  }

  @recordCaller
  private func _clamp(min: Tensor, max: Tensor) -> Tensor {
    return clampInternal(Float.self, minTensor: min, maxTensor: max)
  }

  @recordCaller
  private func _clamp<T: NumericTensorElement>(min: Tensor, max: T) -> Tensor {
    return clampInternal(T.self, minTensor: min, maxScalar: max)
  }

  @recordCaller
  private func _clamp<T: NumericTensorElement>(min: T, max: Tensor) -> Tensor {
    return clampInternal(T.self, minScalar: min, maxTensor: max)
  }

  @recordCaller
  internal func _clampInternal<T: NumericTensorElement>(
    _ t: T.Type, minScalar: T? = nil, minTensor: Tensor? = nil, maxScalar: T? = nil,
    maxTensor: Tensor? = nil
  ) -> Tensor {
    #alwaysAssert(dtype.isNumeric, "cannot use clamp() with dtype \(dtype)")
    let backend = Backend.current

    var bcastTensors = [self]
    if let mt = minTensor {
      bcastTensors.append(mt)
      #alwaysAssert(
        dtype == mt.dtype, "mismatched dtype of self (\(dtype)) and min tensor (\(mt.dtype))")
    }
    if let mt = maxTensor {
      bcastTensors.append(mt)
      #alwaysAssert(
        dtype == mt.dtype, "mismatched dtype of self (\(dtype)) and max tensor (\(mt.dtype))")
    }
    let (bcastShape, stridesImmutable) = Tensor.lazyBroadcast(bcastTensors)
    var strides = stridesImmutable

    let tStrides = strides.remove(at: 0)
    let minStrides: BroadcastStrides? = (minTensor == nil ? nil : strides.remove(at: 0))
    let maxStrides: BroadcastStrides? = (maxTensor == nil ? nil : strides.remove(at: 0))

    let t = self.noGrad()
    let minData = minTensor?.noGrad()
    let maxData = maxTensor?.noGrad()

    let newData: Task<Tensor.Data, Error> = Tensor.createDataTask {
      let minArg: Backend.TensorOrScalar<T>? =
        if let t = minData, let s = minStrides {
          .tensor(BroadcastData(strides: s, data: try await t.data))
        } else if let val = minScalar {
          .scalar(val, bcastShape)
        } else {
          nil
        }
      let maxArg: Backend.TensorOrScalar<T>? =
        if let t = maxData, let s = maxStrides {
          .tensor(BroadcastData(strides: s, data: try await t.data))
        } else if let val = maxScalar {
          .scalar(val, bcastShape)
        } else {
          nil
        }
      return try await backend.clamp(
        BroadcastData(strides: tStrides, data: try await t.data),
        T.self,
        min: minArg,
        max: maxArg,
        dtype: t.dtype
      )
    }

    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: bcastShape, dtype: dtype)
    } else {
      let handle = saveForBackward()
      let minHandle = minTensor?.saveForBackward()
      let maxHandle = maxTensor?.saveForBackward()
      let rawResult = Tensor(dataTask: newData, shape: bcastShape, dtype: dtype)
      return rawResult.onGrad { grad in
        handle.backward(backend) {
          let mask =
            if let max = maxScalar {
              // Fast paths when the gradient can be assigned to self
              if let min = minScalar, min < max {
                rawResult == self
              } else if minScalar == nil && minTensor == nil {
                rawResult == self
              } else {
                (rawResult == self) & (rawResult != max)
              }
            } else if let max = maxTensor {
              (rawResult == self) & (rawResult != max)
            } else {
              rawResult == self
            }
          return mask.when(isTrue: grad, isFalse: 0).reduceBroadcast(tStrides, as: self)
        }
        if let t = minTensor, let h = minHandle, let s = minStrides {
          h.backward(backend) {
            ((rawResult == t) & (rawResult != self)).when(isTrue: grad, isFalse: 0).reduceBroadcast(
              s, as: t)
          }
        }
        if let t = maxTensor, let h = maxHandle, let s = maxStrides {
          h.backward(backend) {
            (rawResult == t).when(isTrue: grad, isFalse: 0).reduceBroadcast(s, as: t)
          }
        }
      }
    }
  }

}
