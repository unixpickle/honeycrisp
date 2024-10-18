import Honeycrisp

class SSIM {
  let kernelSize: Int
  let kernel: Tensor

  init(kernelSize: Int = 11, sigma: Float = 1.5) {
    assert(kernelSize % 2 == 1)
    self.kernelSize = kernelSize
    let kSize: Int = (kernelSize - 1) / 2
    let xs = Tensor(range: (-kSize)...kSize).cast(.float32)
    let gauss = (-0.5 * (xs / sigma).pow(2)).exp()
    let kernel1D = gauss / gauss.sum().expand(as: gauss)
    let kernel2D = Tensor.outer(kernel1D, kernel1D)
    self.kernel = kernel2D.reshape([1, 1, kernelSize, kernelSize])
  }

  func callAsFunction(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    // Based on https://pytorch.org/ignite/_modules/ignite/metrics/ssim.html#SSIM

    let groups = lhs.shape[1]
    let inputs = [lhs, rhs, lhs * lhs, rhs * rhs, lhs * rhs]
    let config = try! Conv2DConfig(
      inChannels: groups, outChannels: groups, kernelSize: .init(x: kernelSize, y: kernelSize),
      imageSize: .init(x: lhs.shape[3], y: lhs.shape[2]), stride: .init(x: 1, y: 1),
      dilation: .init(x: 1, y: 1),
      padding: .init(before: .init(x: 0, y: 0), after: .init(x: 0, y: 0)), groups: groups,
      channelsLast: false
    )
    let outputs = Tensor.conv2D(
      config, image: Tensor(concat: inputs, axis: 0),
      kernel: kernel.repeating(axis: 0, count: groups)
    )
    .split(axis: 0, counts: [Int](repeating: lhs.shape[0], count: inputs.count))

    let lhsMeanSq = outputs[0].pow(2)
    let rhsMeanSq = outputs[1].pow(2)
    let lhsRhsMean = outputs[0] * outputs[1]

    let lhsVar = outputs[2] - lhsMeanSq
    let rhsVar = outputs[3] - rhsMeanSq
    let lhsRhsCov = outputs[4] - lhsRhsMean

    let k1 = 0.01
    let k2 = 0.03
    let c1 = k1 * k1
    let c2 = k2 * k2

    let a1 = 2 * lhsRhsMean + c1
    let a2 = 2 * lhsRhsCov + c2
    let b1 = lhsMeanSq + rhsMeanSq + c1
    let b2 = lhsVar + rhsVar + c2
    let ssim = (a1 * a2) / (b1 * b2)

    return ssim.flatten(startAxis: 1).mean(axis: 1)
  }
}
