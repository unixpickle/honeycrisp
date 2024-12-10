import Honeycrisp

extension Tensor {
  public func onGradUnsafe(_ x: @escaping (Tensor) -> Void) -> Tensor {
    struct X: @unchecked Sendable {
      let x: (Tensor) -> Void
    }
    let cb = X(x: x)
    return self.onGrad { g in cb.x(g) }
  }
}
