extension Tensor {

  public convenience init(uniform shape: [Int]) {
    self.init(
      data: Array((0..<shape.product()).map { _ in Float.random(in: 0..<1.0) }),
      shape: shape
    )
  }

  public convenience init(gaussian shape: [Int]) {
    var count = shape.product()
    if count % 2 == 1 {
      count += 1
    }
    count /= 2
    let u1 = Tensor(uniform: [count]).elemwise { Swift.max(1e-5, $0) }
    let u2 = Tensor(uniform: [count])
    let r = (-2 * u1.log()).sqrt()
    let phi = 2 * Float.pi * u2
    let z1 = r * phi.cos()
    let z2 = r * phi.sin()
    self.init(
      data: Array((z1.data + z2.data)[..<shape.product()]),
      shape: shape
    )
  }

}
