#if canImport(Accelerate)
  import Accelerate
#endif

extension Tensor {

  public func transpose() -> Tensor {
    assert(shape.count == 2, "can only transpose two-dimensional tensors")
    let newShape = [shape[1], shape[0]]
    var newData = Array(repeating: Float(0), count: data.count)
    var dstIdx = 0
    for i in 0..<shape[1] {
      for j in 0..<shape[0] {
        newData[dstIdx] = data[i + j * shape[1]]
        dstIdx += 1
      }
    }
    if !needsGrad {
      return Tensor(data: newData, shape: newShape)
    } else {
      let handle = saveForBackward()
      return Tensor(data: newData, shape: newShape) { grad in
        handle.backward(grad: grad.transpose())
      }
    }
  }

  public static func &* (lhs: Tensor, rhs: Tensor) -> Tensor {
    return tensorMatmul(lhs, rhs)
  }

  #if canImport(Accelerate)
    func arrayDescriptor<T>(_ f: (BNNSNDArrayDescriptor) throws -> T) rethrows -> T {
      assert(shape.count <= 8, "tensor shape must be 8 or less for BNNS")
      let layout =
        switch shape.count {
        case 0, 1: BNNSDataLayout1DFirstMajor
        case 2: BNNSDataLayout2DFirstMajor
        case 3: BNNSDataLayout3DFirstMajor
        case 4: BNNSDataLayout4DFirstMajor
        case 5: BNNSDataLayout5DFirstMajor
        case 6: BNNSDataLayout6DFirstMajor
        case 7: BNNSDataLayout7DFirstMajor
        default: BNNSDataLayout8DFirstMajor
        }
      return try data.withUnsafeBufferPointer { buffer in
        try f(
          BNNSNDArrayDescriptor(
            flags: BNNSNDArrayFlags(0),
            layout: layout,
            size: (
              shape.count > 0 ? shape[0] : 1,
              shape.count > 1 ? shape[1] : 0,
              shape.count > 2 ? shape[2] : 0,
              shape.count > 3 ? shape[3] : 0,
              shape.count > 4 ? shape[4] : 0,
              shape.count > 5 ? shape[5] : 0,
              shape.count > 6 ? shape[6] : 0,
              shape.count > 7 ? shape[7] : 0
            ),
            stride: (0, 0, 0, 0, 0, 0, 0, 0),
            data: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
            data_type: .float,
            table_data: nil,
            table_data_type: .float,
            data_scale: 1,
            data_bias: 0
          ))
      }
    }
  #endif

}

private func tensorMatmul(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
  assert(
    lhs.shape.count == 2 && rhs.shape.count == 2,
    "matrices must be two-dimensional: lhs.shape=\(lhs.shape) rhs.shape=\(rhs.shape)"
  )
  assert(
    lhs.shape[1] == rhs.shape[0],
    "matmul shape mismatch: lhs.shape=\(lhs.shape) rhs.shape=\(rhs.shape)"
  )
  let shape = [lhs.shape[0], rhs.shape[1]]
  var data = Array(repeating: Float(0), count: lhs.shape[0] * rhs.shape[1])

  func cpuImplementation() {
    for outRow in 0..<shape[0] {
      for outCol in 0..<shape[1] {
        var sum: Float = 0.0
        for k in 0..<lhs.shape[1] {
          let lhsIdx = outRow * lhs.shape[1] + k
          let rhsIdx = k * rhs.shape[1] + outCol
          sum += lhs.data[lhsIdx] * rhs.data[rhsIdx]
        }
        data[outRow * shape[1] + outCol] = sum
      }
    }
  }

  #if canImport(Accelerate)
    let success = lhs.arrayDescriptor { lhsArray in
      rhs.arrayDescriptor { rhsArray in
        let outTensor = Tensor(data: data, shape: shape)
        let result = outTensor.arrayDescriptor { outArray in
          do {
            let size = BNNS.matrixMultiplicationWorkspaceSize(
              inputA: lhsArray, transposed: false,
              inputB: rhsArray, transposed: false,
              output: outArray,
              alpha: 1.0
            )
            guard let size = size else {
              return false
            }
            var workspace = [UInt8](repeating: 0, count: size)
            try workspace.withUnsafeMutableBufferPointer { workspacePtr in
              try BNNS.applyMatrixMultiplication(
                inputA: lhsArray, transposed: false,
                inputB: rhsArray, transposed: false,
                output: outArray,
                alpha: 1.0,
                workspace: UnsafeMutableRawBufferPointer(workspacePtr)
              )
            }
          } catch {
            return false
          }
          return true
        }

        // Technically this isn't required because of CoW semantics
        // being violated, but it's still nice to include.
        data = outTensor.data
        return result
      }
    }
    if !success {
      cpuImplementation()
    }
  #else
    cpuImplementation()
  #endif
  if !lhs.needsGrad && !rhs.needsGrad {
    return Tensor(data: data, shape: shape)
  } else {
    let lhsHandle = lhs.saveForBackward()
    let rhsHandle = rhs.saveForBackward()
    return Tensor(data: data, shape: shape) { grad in
      lhsHandle.backward(grad: tensorMatmul(grad, rhs.noGrad().transpose()))
      rhsHandle.backward(grad: tensorMatmul(lhs.noGrad().transpose(), grad))
    }
  }
}
