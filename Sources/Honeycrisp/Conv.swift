public enum Conv2DError: Error {
  case invalidGroups(String)
}

public struct Conv2DConfig {
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

    let paddedImage = { b, y, x, c in
      assert(c >= 0 && c < imageSize.c)
      let y = y - paddingH.before
      let x = x - paddingW.before
      if y < 0 || y >= imageSize.h || x < 0 || x >= imageSize.w {
        return T(0.0)
      } else {
        return nhwcImage(b, y, x, c)
      }
    }

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
}

extension Tensor {
  public static func conv2d(_ conv: Conv2DConfig, image: Tensor, kernel: Tensor) -> Tensor {
    alwaysAssert(
      image.dtype == kernel.dtype,
      "image and kernel dtypes differ: \(image.dtype) vs \(kernel.dtype)")
    alwaysAssert(image.shape.count == 4, "invalid image shape: \(image.shape)")
    alwaysAssert(kernel.shape.count == 4, "invalid image shape: \(kernel.shape)")
    if conv.channelsLast {
      alwaysAssert(
        image.shape[1...] == [conv.imageSize.h, conv.imageSize.w, conv.imageSize.c],
        "invalid image shape \(image.shape) for conv \(conv)")
    } else {
      alwaysAssert(
        image.shape[1...] == [conv.imageSize.c, conv.imageSize.h, conv.imageSize.w],
        "invalid image shape \(image.shape) for conv \(conv)")
    }
    alwaysAssert(
      kernel.shape == [
        conv.kernelSize.c, conv.imageSize.c / conv.groups, conv.kernelSize.h, conv.kernelSize.w,
      ],
      "invalid kernel shape \(kernel.shape) for conv \(conv)")
    guard let (outH, outW, outC) = try? conv.outputShape() else {
      fatalError("failed to compute output shape for conv: \(conv)")
    }
    let outShape =
      if conv.channelsLast {
        [image.shape[0], outH, outW, outC]
      } else {
        [image.shape[0], outC, outH, outW]
      }
    let backend = Backend.current
    let newData = Task {
      try await backend.conv2d(
        conv, batchSize: image.shape[0], image: try await image.data, kernel: try await kernel.data,
        dtype: image.dtype)
    }
    if !Tensor.isGradEnabled || (!image.needsGrad && !kernel.needsGrad) {
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype)
    } else {
      return Tensor(dataTask: newData, shape: outShape, dtype: image.dtype) { grad in
        alwaysAssert(false, "conv2d gradient is not yet implemented")
      }
    }
  }
}
