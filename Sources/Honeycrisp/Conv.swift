public enum Conv2DError: Error {
  case invalidGroups(String)
}

public struct Conv2DConfig: Hashable {
  /// Get the value of a tensor at a 4D coordinate.
  public typealias TensorGetter<T: NumericTensorElement> = (Int, Int, Int, Int) -> T

  public typealias HWCSize = (h: Int, w: Int, c: Int)
  public typealias HWSize = (h: Int, w: Int)
  public typealias AxisPadding = (before: Int, after: Int)

  public var kernelSize: HWCSize
  public var imageSize: HWCSize
  public var stride: HWSize
  public var dilation: HWSize
  public var paddingH: AxisPadding
  public var paddingW: AxisPadding
  public var groups: Int
  public var channelsLast: Bool

  private func allFields() -> [Int] {
    [
      kernelSize.c, kernelSize.h, kernelSize.w, imageSize.c, imageSize.h, imageSize.w, stride.h,
      stride.w, dilation.h, dilation.w, paddingH.before, paddingH.after, paddingW.before,
      paddingW.after, groups, channelsLast ? 1 : 0,
    ]
  }

  public func hash(into hasher: inout Hasher) {
    for x in allFields() {
      hasher.combine(x)
    }
  }

  public static func == (lhs: Conv2DConfig, rhs: Conv2DConfig) -> Bool {
    lhs.allFields() == rhs.allFields()
  }

  public func outputShape() throws -> HWCSize {
    if kernelSize.c % groups != 0 {
      throw Conv2DError.invalidGroups(
        "kernel channels \(kernelSize.c) is not divisible by groups \(groups)")
    }
    if imageSize.c % groups != 0 {
      throw Conv2DError.invalidGroups(
        "image channels \(imageSize.c) is not divisible by groups \(groups)")
    }
    let outHeight =
      (imageSize.h + paddingH.before + paddingH.after - dilation.h * (kernelSize.h - 1) + stride.h
        - 1)
      / stride.h
    let outWidth =
      (imageSize.w + paddingW.before + paddingW.after - dilation.w * (kernelSize.w - 1) + stride.w
        - 1)
      / stride.w
    return HWCSize(h: outHeight, w: outWidth, c: kernelSize.c)
  }

  public func kernelTensorShape() throws -> [Int] {
    if imageSize.c % groups != 0 {
      throw Conv2DError.invalidGroups(
        "groups \(groups) does not divide image channels \(imageSize.c)")
    }
    return [kernelSize.c, imageSize.c / groups, kernelSize.h, kernelSize.w]
  }

  public func imageTensorShape(batch: Int) -> [Int] {
    if channelsLast {
      [batch, imageSize.h, imageSize.w, imageSize.c]
    } else {
      [batch, imageSize.c, imageSize.h, imageSize.w]
    }
  }

  public func outputTensorShape(batch: Int) throws -> [Int] {
    let (outH, outW, outC) = try outputShape()
    if channelsLast {
      return [batch, outH, outW, outC]
    } else {
      return [batch, outC, outH, outW]
    }
  }

  internal func paddedNCHWInput<T>(_ nhwcImage: @escaping TensorGetter<T>) -> TensorGetter<T> {
    { b, y, x, c in
      assert(c >= 0 && c < imageSize.c)
      let y = y - paddingH.before
      let x = x - paddingW.before
      if y < 0 || y >= imageSize.h || x < 0 || x >= imageSize.w {
        return T(0.0)
      } else {
        return nhwcImage(b, y, x, c)
      }
    }
  }

  /// Create a lazy output of the convolution.
  ///
  /// Kernel is always [n_out, n_in/groups, kernel_height, kernel_width]
  public func lazyForward<T: NumericTensorElement>(
    batch: Int, image: @escaping TensorGetter<T>, kernel: @escaping TensorGetter<T>
  ) throws -> TensorGetter<T> {
    let nhwcImage =
      if channelsLast {
        image
      } else {
        { b, y, x, c in image(b, c, y, x) }
      }
    let paddedImage = paddedNCHWInput(nhwcImage)

    let (outH, outW, outC) = try outputShape()
    let nhwcResult = { (b: Int, y: Int, x: Int, c: Int) in
      assert(y >= 0 && y < outH, "y \(y) out of out of range [0, \(outH))")
      assert(x >= 0 && x < outW, "x \(x) out of out of range [0, \(outW))")
      assert(c >= 0 && c < outC, "c \(c) out of out of range [0, \(outC))")

      let inGroupSize = imageSize.c / groups
      let outGroupSize = outC / groups
      let groupIdx = c / outGroupSize

      var sum = T(0.0)
      for kernelY in 0..<kernelSize.h {
        let imageY = y * stride.h + kernelY * dilation.h
        for kernelX in 0..<kernelSize.w {
          let imageX = x * stride.w + kernelX * dilation.w
          for inC in 0..<inGroupSize {
            let imageVal = paddedImage(b, imageY, imageX, inC + inGroupSize * groupIdx)
            let kernelVal = kernel(c, inC, kernelY, kernelX)
            sum = sum + imageVal * kernelVal
          }
        }
      }
      return sum
    }
    let result =
      if channelsLast {
        nhwcResult
      } else {
        { b, c, y, x in nhwcResult(b, y, x, c) }
      }
    return result
  }

  public func lazyTranspose<T: NumericTensorElement>(
    batch: Int, image: @escaping TensorGetter<T>, kernel: @escaping TensorGetter<T>
  ) throws -> TensorGetter<T> {
    let nhwcImage =
      if channelsLast {
        image
      } else {
        { b, y, x, c in image(b, c, y, x) }
      }

    let (outH, outW, outC) = try outputShape()
    let nhwcResult = { (b: Int, y: Int, x: Int, c: Int) in
      assert(y >= 0 && y < imageSize.h, "y \(y) out of out of range [0, \(imageSize.h))")
      assert(x >= 0 && x < imageSize.w, "x \(x) out of out of range [0, \(imageSize.w))")
      assert(c >= 0 && c < imageSize.c, "c \(c) out of out of range [0, \(imageSize.c))")

      let inGroupSize = imageSize.c / groups
      let outGroupSize = outC / groups
      let groupIdx = c / inGroupSize

      var sum = T(0.0)
      for kernelY in 0..<kernelSize.h {
        guard let sourceY = exactDiv(y + paddingH.before - kernelY * dilation.h, stride.h) else {
          continue
        }
        if sourceY < 0 || sourceY >= outH {
          continue
        }
        for kernelX in 0..<kernelSize.w {
          guard let sourceX = exactDiv(x + paddingW.before - kernelX * dilation.w, stride.w) else {
            continue
          }
          if sourceX < 0 || sourceX >= outW {
            continue
          }
          for sourceC in 0..<outGroupSize {
            let imageVal = nhwcImage(b, sourceY, sourceX, sourceC + outGroupSize * groupIdx)
            let kernelVal = kernel(
              sourceC + outGroupSize * groupIdx, c % inGroupSize, kernelY, kernelX)
            sum = sum + imageVal * kernelVal
          }
        }
      }
      return sum
    }
    let result =
      if channelsLast {
        nhwcResult
      } else {
        { b, c, y, x in nhwcResult(b, y, x, c) }
      }
    return result
  }

  public func lazyKernelGrad<T: NumericTensorElement>(
    batch: Int, image: @escaping TensorGetter<T>, outGrad: @escaping TensorGetter<T>
  ) throws -> TensorGetter<T> {
    let (nhwcImage, nhwcOutGrad) =
      if channelsLast {
        (image, outGrad)
      } else {
        ({ b, y, x, c in image(b, c, y, x) }, { b, y, x, c in outGrad(b, c, y, x) })
      }
    let paddedImage = paddedNCHWInput(nhwcImage)

    let (outH, outW, outC) = try outputShape()
    return { (cOut: Int, cIn: Int, kernelY: Int, kernelX: Int) in
      assert(cOut >= 0 && cOut < kernelSize.c)
      assert(cIn >= 0 && cIn < imageSize.c / groups)
      assert(kernelY >= 0 && kernelY < kernelSize.h)
      assert(kernelX >= 0 && kernelX < kernelSize.w)

      let outGroupSize = outC / groups
      let inGroupSize = imageSize.c / groups
      let groupIdx = cOut / outGroupSize

      var sum = T(0.0)
      for batchIdx in 0..<batch {
        for outY in 0..<outH {
          let imageY = outY * stride.h + kernelY * dilation.h
          for outX in 0..<outW {
            let imageX = outX * stride.w + kernelX * dilation.w
            sum =
              sum + paddedImage(batchIdx, imageY, imageX, cIn + groupIdx * inGroupSize)
              * nhwcOutGrad(batchIdx, outY, outX, cOut)
          }
        }
      }
      return sum
    }
  }
}

extension Tensor {
  public static func conv2d(_ conv: Conv2DConfig, image: Tensor, kernel: Tensor) -> Tensor {
    alwaysAssert(
      image.dtype == kernel.dtype,
      "image and kernel dtypes differ: \(image.dtype) vs \(kernel.dtype)")
    alwaysAssert(image.shape.count == 4, "invalid image shape: \(image.shape)")
    alwaysAssert(kernel.shape.count == 4, "invalid kernel shape: \(kernel.shape)")

    let expectedImageShape = conv.imageTensorShape(batch: image.shape[0])
    alwaysAssert(
      image.shape == expectedImageShape,
      "invalid image shape \(image.shape), expected \(expectedImageShape)")

    let outShape: [Int]
    do {
      outShape = try conv.outputTensorShape(batch: image.shape[0])
      let kernelShape = try conv.kernelTensorShape()
      alwaysAssert(kernel.shape == kernelShape)
    } catch {
      fatalError("\(error)")
    }
    let backend = Backend.current
    let newData = Task {
      try await backend.conv2d(
        conv, batch: image.shape[0], image: try await image.data, kernel: try await kernel.data,
        dtype: image.dtype)
    }
    if !Tensor.isGradEnabled || (!image.needsGrad && !kernel.needsGrad) {
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype)
    } else {
      let imageHandle = image.saveForBackward()
      let kernelHandle = kernel.saveForBackward()
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype) { grad in
        imageHandle.backward(
          backend.use { Tensor.conv2dTranspose(conv, image: grad, kernel: kernel.noGrad()) })
        kernelHandle.backward(
          backend.use { Tensor.conv2dKernelGrad(conv, image: image.noGrad(), outGrad: grad) })
      }
    }
  }

  public static func conv2dTranspose(_ conv: Conv2DConfig, image: Tensor, kernel: Tensor) -> Tensor
  {
    alwaysAssert(
      image.dtype == kernel.dtype,
      "image and kernel dtypes differ: \(image.dtype) vs \(kernel.dtype)")
    alwaysAssert(image.shape.count == 4, "invalid image shape: \(image.shape)")
    alwaysAssert(kernel.shape.count == 4, "invalid image shape: \(kernel.shape)")
    alwaysAssert(
      kernel.shape == [
        conv.kernelSize.c, conv.imageSize.c / conv.groups, conv.kernelSize.h, conv.kernelSize.w,
      ],
      "invalid kernel shape \(kernel.shape) for conv \(conv)")

    let outShape: [Int]
    let inShape: [Int]
    do {
      inShape = try conv.outputTensorShape(batch: image.shape[0])
      outShape = conv.imageTensorShape(batch: image.shape[0])
      alwaysAssert(
        image.shape == inShape,
        "invalid input shape for transposed conv2D: \(image.shape) (expected \(inShape))")
    } catch {
      fatalError("\(error)")
    }

    let backend = Backend.current
    let newData = Task {
      try await backend.conv2dTranspose(
        conv, batch: image.shape[0], image: try await image.data, kernel: try await kernel.data,
        dtype: image.dtype)
    }
    if !Tensor.isGradEnabled || (!image.needsGrad && !kernel.needsGrad) {
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype)
    } else {
      fatalError("conv2dTranspose gradient is not yet implemented")
    }
  }

  public static func conv2dKernelGrad(_ conv: Conv2DConfig, image: Tensor, outGrad: Tensor)
    -> Tensor
  {
    alwaysAssert(
      image.dtype == outGrad.dtype,
      "image and outGrad dtypes differ: \(image.dtype) vs \(outGrad.dtype)")
    alwaysAssert(image.shape.count == 4, "invalid image shape: \(image.shape)")
    alwaysAssert(outGrad.shape.count == 4, "invalid outGrad shape: \(outGrad.shape)")
    if conv.channelsLast {
      alwaysAssert(
        image.shape[1...] == [conv.imageSize.h, conv.imageSize.w, conv.imageSize.c],
        "invalid image shape \(image.shape) for conv \(conv)")
    } else {
      alwaysAssert(
        image.shape[1...] == [conv.imageSize.c, conv.imageSize.h, conv.imageSize.w],
        "invalid image shape \(image.shape) for conv \(conv)")
    }

    let kernelShape: [Int]
    do {
      kernelShape = try conv.kernelTensorShape()
      let outShape = try conv.outputTensorShape(batch: image.shape[0])
      alwaysAssert(
        outGrad.shape == outShape, "unexpected outGrad shape \(outGrad.shape), expected \(outShape)"
      )
    } catch {
      fatalError("failed to compute output shape for conv: \(conv): \(error)")
    }

    let backend = Backend.current
    let newData = Task {
      try await backend.conv2dKernelGrad(
        conv, batch: image.shape[0], image: try await image.data,
        outGrad: try await outGrad.data, dtype: image.dtype)
    }
    if !Tensor.isGradEnabled || (!image.needsGrad && !outGrad.needsGrad) {
      return Tensor(dataTask: newData, shape: kernelShape, dtype: image.dtype)
    } else {
      fatalError("conv2dKernelGrad gradient is not yet implemented")
    }
  }
}

func exactDiv(_ x: Int, _ y: Int) -> Int? {
  if x % y != 0 {
    nil
  } else {
    x / y
  }
}
