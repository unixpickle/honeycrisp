import Metal

public class Tensor {

  public enum DType {
    case int64
    case bool
    case float16
    case float32

    var supportsGrad: Bool {
      self == .float16 || self == .float32
    }

    var isNumeric: Bool {
      self != .bool
    }

    var byteSize: Int {
      switch self {
      case .int64:
        8
      case .bool:
        1
      case .float16:
        2
      case .float32:
        4
      }
    }
  }

  public struct Data {
    public let backend: Backend
    public let buffer: MTLBuffer
    public let completeOnAllDevices: Task<Void, Error>? = nil
  }

  public final class BackwardHandle {
    private var addGrad: ((Tensor) throws -> Void)?
    private var cancel: (() -> Void)?

    init() {
      self.addGrad = { _ in () }
      self.cancel = { () in () }
    }

    init(addGrad: @escaping (Tensor) throws -> Void, cancel: @escaping () -> Void) {
      self.addGrad = addGrad
      self.cancel = cancel
    }

    public func backward(_ grad: Tensor) throws {
      assert(!grad.needsGrad, "second-order gradients are not supported")
      assert(addGrad != nil, "cannot re-use backward handle")
      let ag = addGrad!
      addGrad = nil
      cancel = nil
      try ag(grad)
    }

    deinit {
      if let cancel = cancel {
        cancel()
      }
    }
  }

  public let dataTask: Task<Data, Error>
  public let shape: [Int]
  public let dtype: DType
  public let needsGrad: Bool

  public var data: Data {
    get async throws {
      try await dataTask.value
    }
  }

  private var backwardImpl: (((Tensor) throws -> Void))?

  // State used during backward pass to accumulate gradients.
  private var curGrad: Tensor? = nil
  private var numBackwardHandles: Int = 0

  public init(
    dataTask: Task<Data, Error>, shape: [Int], dtype: DType,
    backwardImpl: (((Tensor) throws -> Void))? = nil
  ) {
    self.dataTask = dataTask
    self.shape = shape
    self.dtype = dtype
    self.backwardImpl = backwardImpl
    self.needsGrad = backwardImpl != nil
  }

  public init<T: TensorElement>(
    data: [T], shape: [Int], dtype: DType? = nil, backwardImpl: ((Tensor) throws -> Void)? = nil
  ) {
    let dtype = dtype ?? T.dtype
    if !dtype.supportsGrad {
      assert(backwardImpl == nil, "cannot specify gradient for dtype \(dtype)")
    }
    assert(data.count == shape.product(), "data count \(data.count) does not match shape \(shape)")
    let backend = Backend.current
    self.dataTask = Task {
      let buf = try await backend.allocate(length: dtype.byteSize * shape.product())
      try arrayToPointer(data, output: buf.contents(), dtype: dtype)
      return Data(backend: backend, buffer: buf)
    }
    self.shape = shape
    self.dtype = dtype
    self.backwardImpl = backwardImpl
    self.needsGrad = backwardImpl != nil
  }

  public convenience init<T: NumericTensorElement>(
    range: Range<T>, dtype: DType? = nil
  ) where Range<T>: Collection {
    let arr = Array(range)
    self.init(data: arr, shape: [arr.count], dtype: dtype ?? T.dtype)
  }

  public convenience init(zerosLike: Tensor) {
    self.init(constant: Float(0), like: zerosLike)
  }

  public convenience init(onesLike: Tensor) {
    self.init(constant: Float(1), like: onesLike)
  }

  public convenience init(zeros shape: [Int]) {
    self.init(constant: Float(0), shape: shape)
  }

  public convenience init(ones shape: [Int]) {
    self.init(constant: Float(1), shape: shape)
  }

  public convenience init<T: TensorElement>(constant: T, like: Tensor) {
    self.init(constant: constant, shape: like.shape, dtype: like.dtype)
  }

  public convenience init<T: TensorElement>(
    constant: T, shape: [Int], dtype: DType? = nil
  ) {
    self.init(
      data: [T](repeating: constant, count: shape.product()), shape: shape, dtype: dtype)
  }

  public func copyToArray<T: TensorElement>(_ out: inout [T]) async throws {
    assert(out.count == shape.product(), "out size must match our size")
    let data = try await data
    if let c = data.completeOnAllDevices {
      try await c.value
    }
    try pointerToArray(data.buffer.contents(), output: &out, dtype: dtype)
  }

  public func floats() async throws -> [Float] {
    var out = [Float](repeating: 0, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  public func item() async throws -> Float {
    assert(shape.product() == 1, "cannot call item() on Tensor of shape \(shape)")
    let data = try await floats()
    return data[0]
  }

  public func noGrad() -> Tensor {
    return Tensor(dataTask: dataTask, shape: shape, dtype: dtype)
  }

  public func onGrad(_ action: @escaping ((Tensor) throws -> Void)) -> Tensor {
    if !needsGrad {
      return Tensor(dataTask: dataTask, shape: shape, dtype: dtype, backwardImpl: action)
    }
    let handle = self.saveForBackward()
    return Tensor(dataTask: dataTask, shape: shape, dtype: dtype) { grad in
      try action(grad)
      try handle.backward(grad)
    }
  }

  public func reshape(_ newShape: [Int]) -> Tensor {
    if shape == newShape {
      return self
    }
    assert(shape.product() == newShape.product())
    if !needsGrad {
      return Tensor(dataTask: dataTask, shape: newShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: dataTask, shape: newShape, dtype: dtype) { [self] grad in
        try handle.backward(grad.reshape(shape))
      }
    }
  }

  public func flatten() -> Tensor {
    return reshape([shape.product()])
  }

  public func cast(as t: Tensor) -> Tensor {
    cast(t.dtype)
  }

  public func cast(_ newType: DType) -> Tensor {
    if newType == dtype {
      return self
    }
    let backend = Backend.current
    let newData = Task {
      try await backend.cast(
        try await self.data, count: shape.product(), inType: dtype, outType: newType)
    }
    if !needsGrad || !newType.supportsGrad {
      return Tensor(dataTask: newData, shape: shape, dtype: newType)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: newType) { grad in
        try handle.backward(grad.cast(self.dtype))
      }
    }
  }

  public func elemwise(op: ElemwiseOp, grad gradOp: ElemwiseOp? = nil) -> Tensor {
    let backend = Backend.current
    let newData = Task {
      try await backend.elemwise(
        try await self.data, op: op, count: shape.product(), dtype: dtype)
    }
    if let gradOp = gradOp, needsGrad {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        try handle.backward(backend.use { grad * self.noGrad().elemwise(op: gradOp) })
      }
    } else {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    }
  }

  public func sin() -> Tensor {
    self.elemwise(op: .sin, grad: .cos)
  }

  public func cos() -> Tensor {
    self.elemwise(op: .cos, grad: .minusSin)
  }

  public func exp() -> Tensor {
    self.elemwise(op: .exp, grad: .exp)
  }

  public func log() -> Tensor {
    self.elemwise(op: .log, grad: .recip)
  }

  public func sigmoid() -> Tensor {
    self.elemwise(op: .sigmoid, grad: .sigmoidGrad)
  }

  public func relu() -> Tensor {
    self.elemwise(op: .relu, grad: .reluGrad)
  }

  public func tanh() -> Tensor {
    2 * (2 * self).sigmoid() - 1
  }

  public func gelu() -> Tensor {
    0.5 * self * (1 + (0.797884561 * (self + 0.044715 * self.pow(3))).tanh())
  }

  public func silu() -> Tensor {
    return self * self.sigmoid()
  }

  public func sqrt() -> Tensor {
    pow(0.5)
  }

  public func rsqrt() -> Tensor {
    pow(-0.5)
  }

  public static func + <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    assert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with + operator")
    let backend = Backend.current
    let newData = Task {
      try await backend.binaryOp(
        try await lhs.data, rhs, op: .add, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    if !lhs.needsGrad {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      return Tensor(
        dataTask: newData, shape: lhs.shape, dtype: lhs.dtype, backwardImpl: lhsHandle.backward)
    }
  }

  public static func + <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    rhs + lhs
  }

  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(
      lhs.shape == rhs.shape,
      "shape mismatch for + operator: lhs=\(lhs.shape) rhs=\(rhs.shape)"
    )
    assert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with + operator")
    assert(
      lhs.dtype == rhs.dtype, "dtypes for + operator do not match: \(lhs.dtype) and \(rhs.dtype)")
    let backend = Backend.current
    let newData = Task {
      try await backend.binaryOp(
        try await lhs.data, try await rhs.data, op: .add, count: lhs.shape.product(),
        dtype: lhs.dtype)
    }
    if !lhs.needsGrad && !rhs.needsGrad {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype) { grad in
        try lhsHandle.backward(grad)
        try rhsHandle.backward(grad)
      }
    }
  }

  public static func * <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    let backend = Backend.current
    let newData = Task {
      try await backend.binaryOp(
        try await lhs.data, rhs, op: .mul, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    if !lhs.needsGrad {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype) { grad in
        try lhsHandle.backward(backend.use { grad * rhs })
      }
    }
  }

  public static func * <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    return rhs * lhs
  }

  public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(
      lhs.shape == rhs.shape,
      "shape mismatch for * operator: lhs=\(lhs.shape) rhs=\(rhs.shape)"
    )
    assert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with * operator")
    assert(
      lhs.dtype == rhs.dtype, "dtypes for * operator do not match: \(lhs.dtype) and \(rhs.dtype)")
    let backend = Backend.current
    let newData = Task {
      try await backend.binaryOp(
        try await lhs.data, try await rhs.data, op: .mul, count: lhs.shape.product(),
        dtype: lhs.dtype)
    }
    if !lhs.needsGrad && !rhs.needsGrad {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype) { grad in
        try lhsHandle.backward(backend.use { grad * rhs.noGrad() })
        try rhsHandle.backward(backend.use { grad * lhs.noGrad() })
      }
    }
  }

  prefix public static func - (t: Tensor) -> Tensor {
    assert(t.dtype.isNumeric, "dtype \(t.dtype) cannot be used with - operator")
    return t * -1
  }

  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs + -1 * rhs
  }

  public static func - <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    return lhs + -rhs
  }

  public static func - <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    return lhs + -rhs
  }

  public func pow<T: NumericTensorElement>(_ exponent: T) -> Tensor {
    assert(dtype.isNumeric, "cannot use pow() with dtype \(dtype)")
    let backend = Backend.current
    let newData = Task {
      try await backend.pow(
        try await self.data, exponent, count: self.shape.product(), dtype: self.dtype)
    }
    if !needsGrad {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    } else {
      let lhsHandle = saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        try lhsHandle.backward(
          backend.use { grad * exponent * self.noGrad().pow(exponent - T(1.0)) })
      }
    }
  }

  public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs * rhs.pow(-1)
  }

  public static func / <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    return lhs * rhs.pow(-1)
  }

  public static func / <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    return lhs * (T(1.0) / rhs)
  }

  public static func == <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    let backend = Backend.current
    let newData = Task {
      try await backend.equals(
        try await lhs.data, rhs, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    return Tensor(dataTask: newData, shape: lhs.shape, dtype: .bool)
  }

  public static func == (lhs: Float, rhs: Tensor) -> Tensor {
    return rhs == lhs
  }

  public static func == (lhs: Tensor, rhs: Tensor) -> Tensor {
    assert(
      lhs.shape == rhs.shape,
      "shape mismatch for == operator: lhs=\(lhs.shape) rhs=\(rhs.shape)"
    )
    assert(
      lhs.dtype == rhs.dtype, "dtypes for == operator do not match: \(lhs.dtype) and \(rhs.dtype)")
    let backend = Backend.current
    let newData = Task {
      try await backend.equals(
        try await lhs.data, try await rhs.data, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    return Tensor(dataTask: newData, shape: lhs.shape, dtype: .bool)
  }

  public func backward(_ grad: Tensor? = nil) throws {
    assert(needsGrad, "backward called on Tensor that does not need grad")
    let grad =
      if let grad = grad {
        grad
      } else {
        Tensor(onesLike: self)
      }

    assert(numBackwardHandles == 0, "cannot call backward() on tensor that is used elsewhere")
    try self.saveForBackward().backward(grad)
  }

  public func saveForBackward() -> BackwardHandle {
    if !self.needsGrad {
      return BackwardHandle()
    }
    assert(self.backwardImpl != nil, "cannot backward a second time")
    numBackwardHandles += 1

    return BackwardHandle(
      addGrad: { [self] grad in
        assert(numBackwardHandles > 0)
        assert(
          grad.shape == shape,
          "gradient shape \(grad.shape) must match tensor shape \(shape)"
        )
        assert(grad.dtype == dtype, "gradient dtype \(grad.dtype) must match tensor dtype \(dtype)")
        if let cg = curGrad {
          curGrad = cg + grad
        } else {
          curGrad = grad
        }
        numBackwardHandles -= 1
        if let grad = curGrad, numBackwardHandles == 0 {
          let bwd = backwardImpl!
          backwardImpl = nil
          try bwd(grad)
          curGrad = nil
        }
      },
      cancel: { [self] in
        numBackwardHandles -= 1
        if self.curGrad != nil && numBackwardHandles == 0 {
          assert(false, "backward pass was incompleted due to an unused reference")
        }
      })
  }

}
