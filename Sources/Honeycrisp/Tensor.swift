import Metal

public final class Tensor {

  public enum DType: Codable {
    case int64
    case bool
    case float16
    case float32

    var supportsGrad: Bool {
      isFloat
    }

    var isNumeric: Bool {
      self != .bool
    }

    var isFloat: Bool {
      self == .float16 || self == .float32
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

    internal func canUseScalarType<T: TensorElement>(_: T.Type) -> Bool {
      switch self {
      case .bool:
        !T.isBoolLossy
      case .int64:
        !T.isInt64Lossy
      default:
        true
      }
    }
  }

  public struct Data {
    public let backend: Backend
    public let buffer: MTLBuffer
    public let completeOnAllDevices: Task<Void, Error>?

    public init(backend: Backend, buffer: MTLBuffer, completeOnAllDevices: Task<Void, Error>? = nil)
    {
      self.backend = backend
      self.buffer = buffer
      self.completeOnAllDevices = completeOnAllDevices
    }
  }

  public final class BackwardHandle {
    private var addGrad: ((Tensor) -> Void)?
    private var cancel: (() -> Void)?
    private var canSkip: Bool

    init() {
      self.addGrad = { _ in () }
      self.cancel = { () in () }
      canSkip = true
    }

    init(addGrad: @escaping (Tensor) -> Void, cancel: @escaping () -> Void) {
      self.addGrad = addGrad
      self.cancel = cancel
      canSkip = false
    }

    public func backward(_ backend: Backend, _ gradFn: () -> Tensor) {
      alwaysAssert(addGrad != nil, "cannot re-use backward handle")
      let ag = addGrad!
      addGrad = nil
      cancel = nil
      if !canSkip {
        let grad = backend.use { gradFn() }
        alwaysAssert(!grad.needsGrad, "second-order gradients are not supported")
        ag(grad)
      }
    }

    deinit {
      if let cancel = cancel {
        cancel()
      }
    }
  }

  private static let GradEnabledThreadKey = "HONEYCRISP_GRAD_ENABLED"

  public static var isGradEnabled: Bool {
    if let enabled = Thread.current.threadDictionary[GradEnabledThreadKey] {
      enabled as! Bool
    } else {
      true
    }
  }

  public static func withGrad<T>(enabled flag: Bool, _ fn: () throws -> T) rethrows -> T {
    if let enabled = Thread.current.threadDictionary[GradEnabledThreadKey] {
      let old = enabled as! Bool
      defer {
        Thread.current.threadDictionary[GradEnabledThreadKey] = old
      }
      Thread.current.threadDictionary[GradEnabledThreadKey] = flag
      return try fn()
    } else {
      defer {
        Thread.current.threadDictionary.removeObject(forKey: GradEnabledThreadKey)
      }
      Thread.current.threadDictionary[GradEnabledThreadKey] = flag
      return try fn()
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

  private var backwardImpl: (((Tensor) -> Void))?

  // State used during backward pass to accumulate gradients.
  private var curGrad: Tensor? = nil
  private var numBackwardHandles: Int = 0

  public init(
    dataTask: Task<Data, Error>, shape: [Int], dtype: DType,
    backwardImpl: (((Tensor) -> Void))? = nil
  ) {
    #if DEBUG
      self.dataTask = Task {
        let result = try await dataTask.value
        let allocSize = result.buffer.allocatedSize
        let minSize = shape.product() * dtype.byteSize
        alwaysAssert(allocSize >= minSize, "buffer of size \(allocSize) underflows shape \(shape)")
        return result
      }
    #else
      self.dataTask = dataTask
    #endif
    self.shape = shape
    self.dtype = dtype
    if Tensor.isGradEnabled {
      self.backwardImpl = backwardImpl
      self.needsGrad = backwardImpl != nil
    } else {
      self.needsGrad = false
    }
  }

  public init<T: TensorElement>(
    data: [T], shape: [Int], dtype: DType? = nil, backwardImpl: ((Tensor) -> Void)? = nil
  ) {
    let dtype = dtype ?? T.dtype
    if !dtype.supportsGrad {
      alwaysAssert(backwardImpl == nil, "cannot specify gradient for dtype \(dtype)")
    }
    alwaysAssert(
      data.count == shape.product(), "data count \(data.count) does not match shape \(shape)")
    alwaysAssert(
      dtype.canUseScalarType(T.self),
      "cannot create Tensor with dtype \(dtype) with scalar type \(T.self)")
    let backend = Backend.current
    self.dataTask = Task {
      let buf = try await backend.allocate(length: dtype.byteSize * shape.product())
      try arrayToPointer(data, output: buf.contents(), dtype: dtype)
      return Data(backend: backend, buffer: buf)
    }
    self.shape = shape
    self.dtype = dtype
    if Tensor.isGradEnabled {
      self.backwardImpl = backwardImpl
      self.needsGrad = backwardImpl != nil
    } else {
      self.needsGrad = false
    }
  }

  public convenience init<T: NumericTensorElement>(
    range: Range<T>, dtype: DType? = nil
  ) where Range<T>: Collection {
    let arr = Array(range)
    self.init(data: arr, shape: [arr.count], dtype: dtype ?? T.dtype)
  }

  public convenience init<T: NumericTensorElement>(
    range: ClosedRange<T>, dtype: DType? = nil
  ) where ClosedRange<T>: Collection {
    let arr = Array(range)
    self.init(data: arr, shape: [arr.count], dtype: dtype ?? T.dtype)
  }

  public convenience init(zerosLike: Tensor) {
    self.init(constant: 0, like: zerosLike)
  }

  public convenience init(onesLike: Tensor) {
    self.init(constant: 1, like: onesLike)
  }

  public convenience init(zeros shape: [Int], dtype: DType = .float32) {
    self.init(constant: 0, shape: shape, dtype: dtype)
  }

  public convenience init(ones shape: [Int], dtype: DType = .float32) {
    self.init(constant: 1, shape: shape, dtype: dtype)
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
    alwaysAssert(out.count == shape.product(), "out size must match our size")
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

  public func ints() async throws -> [Int] {
    var out = [Int](repeating: 0, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  public func int64s() async throws -> [Int64] {
    var out = [Int64](repeating: 0, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  public func bools() async throws -> [Bool] {
    var out = [Bool](repeating: false, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  public func item() async throws -> Float {
    alwaysAssert(shape.product() == 1, "cannot call item() on Tensor of shape \(shape)")
    let data = try await floats()
    return data[0]
  }

  public func wait() async throws {
    if let c = (try await data).completeOnAllDevices {
      try await c.value
    }
  }

  public func noGrad() -> Tensor {
    return Tensor(dataTask: dataTask, shape: shape, dtype: dtype)
  }

  internal func createDataTask(_ fn: @escaping (Tensor) async throws -> Data) -> Task<Data, Error> {
    Tensor.createDataTask(self, fn)
  }

  static internal func createDataTask(
    _ x: Tensor, _ fn: @escaping (Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = x.noGrad()
    return Task {
      try await fn(safeRef1)
    }
  }

  static internal func createDataTask(
    _ x: Tensor, _ y: Tensor, _ fn: @escaping (Tensor, Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = x.noGrad()
    let safeRef2 = y.noGrad()
    return Task {
      try await fn(safeRef1, safeRef2)
    }
  }

  static internal func createDataTask(
    _ x: Tensor, _ y: Tensor, _ z: Tensor,
    _ fn: @escaping (Tensor, Tensor, Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = x.noGrad()
    let safeRef2 = y.noGrad()
    let safeRef3 = z.noGrad()
    return Task {
      try await fn(safeRef1, safeRef2, safeRef3)
    }
  }

  static internal func createDataTask(
    _ w: Tensor, _ x: Tensor, _ y: Tensor, _ z: Tensor,
    _ fn: @escaping (Tensor, Tensor, Tensor, Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = w.noGrad()
    let safeRef2 = x.noGrad()
    let safeRef3 = y.noGrad()
    let safeRef4 = z.noGrad()
    return Task {
      try await fn(safeRef1, safeRef2, safeRef3, safeRef4)
    }
  }

  public func onGrad(_ action: @escaping ((Tensor) -> Void)) -> Tensor {
    alwaysAssert(dtype.supportsGrad, "cannot compute gradients for dtype \(dtype)")
    if !Tensor.isGradEnabled {
      return Tensor(dataTask: dataTask, shape: shape, dtype: dtype)
    }
    if !needsGrad {
      return Tensor(dataTask: dataTask, shape: shape, dtype: dtype, backwardImpl: action)
    }
    let handle = self.saveForBackward()
    return Tensor(dataTask: dataTask, shape: shape, dtype: dtype) { grad in
      action(grad)
      handle.backward(Backend.current) { grad }
    }
  }

  public func reshape(_ newShape: [Int]) -> Tensor {
    if shape == newShape {
      return self
    }
    let useShape = fillNegativeOneInShape(newShape)
    alwaysAssert(
      shape.product() == useShape.product(), "invalid reshape from \(shape) to \(newShape)")
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: dataTask, shape: useShape, dtype: dtype)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: dataTask, shape: useShape, dtype: dtype) { [self] grad in
        handle.backward(Backend.current) { grad.reshape(shape) }
      }
    }
  }

  public func reshape(as x: Tensor) -> Tensor {
    reshape(x.shape)
  }

  internal func fillNegativeOneInShape(_ newShape: [Int]) -> [Int] {
    guard let idx = newShape.firstIndex(of: -1) else {
      return newShape
    }
    alwaysAssert(newShape.filter({ x in x < 0 }).count == 1, "invalid shape: \(newShape)")
    let otherProduct = newShape[..<idx].product() * newShape[(idx + 1)...].product()
    alwaysAssert(
      otherProduct != 0,
      "cannot infer axis \(idx) for shape \(newShape) because value is ambiguous")
    let fullProduct = shape.product()
    alwaysAssert(
      fullProduct % otherProduct == 0,
      "shape \(newShape) is incompatible with tensor shape \(shape)")
    var result = newShape
    result[idx] = fullProduct / otherProduct
    return result
  }

  public func flatten(startAxis: Int = 0, endAxis: Int = -1) -> Tensor {
    let startAxis = positiveAxis(startAxis)
    let endAxis = shape.count == 0 && endAxis == -1 ? 0 : positiveAxis(endAxis)
    if startAxis == 0 && endAxis == 0 && shape.count == 0 {
      return reshape([1])
    }
    alwaysAssert(endAxis >= startAxis, "invalid axes for flatten(): [\(startAxis), \(endAxis)]")
    return reshape(
      shape[..<startAxis] + [shape[startAxis...endAxis].product()] + shape[(endAxis + 1)...])
  }

  public func squeeze(axis: Int) -> Tensor {
    let posAxis = positiveAxis(axis)
    alwaysAssert(shape[posAxis] == 1, "cannot squeeze axis \(axis) for shape \(shape)")
    var newShape = shape
    newShape.remove(at: posAxis)
    return reshape(newShape)
  }

  public func unsqueeze(axis: Int) -> Tensor {
    let posAxis = axis < 0 ? axis + shape.count + 1 : axis
    alwaysAssert(posAxis >= 0, "axis \(axis) out of bounds for shape \(shape)")
    var newShape = shape
    newShape.insert(1, at: posAxis)
    return reshape(newShape)
  }

  public func cast(as t: Tensor) -> Tensor {
    cast(t.dtype)
  }

  public func cast(_ newType: DType) -> Tensor {
    if newType == dtype {
      return self
    }
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.cast(
        try await t.data, count: t.shape.product(), inType: t.dtype, outType: newType)
    }
    if !needsGrad || !Tensor.isGradEnabled || !newType.supportsGrad {
      return Tensor(dataTask: newData, shape: shape, dtype: newType)
    } else {
      let handle = self.saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: newType) { grad in
        handle.backward(backend) { grad.cast(self.dtype) }
      }
    }
  }

  public static func + <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    alwaysAssert(
      lhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(lhs.dtype)")
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with + operator")
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.binaryOp(
        try await lhs.data, rhs, op: .add, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    if !lhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      return Tensor(
        dataTask: newData, shape: lhs.shape, dtype: lhs.dtype
      ) { grad in lhsHandle.backward(backend) { grad } }
    }
  }

  public static func + <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    rhs + lhs
  }

  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with + operator")
    alwaysAssert(
      lhs.dtype == rhs.dtype, "dtypes for + operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let (outputShape, ((lhs, lhsStrides), (rhs, rhsStrides))) = Tensor.lazyBroadcast(lhs, rhs)

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.binaryOp(
        a: BroadcastData(strides: lhsStrides, data: try await lhs.data),
        b: BroadcastData(strides: rhsStrides, data: try await rhs.data),
        op: .add, count: outputShape.product(), dtype: lhs.dtype)
    }
    if !Tensor.isGradEnabled || (!lhs.needsGrad && !rhs.needsGrad) {
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype) { grad in
        lhsHandle.backward(backend) { grad.reduceBroadcast(lhsStrides, as: lhs) }
        rhsHandle.backward(backend) { grad.reduceBroadcast(rhsStrides, as: rhs) }
      }
    }
  }

  public static func * <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    alwaysAssert(
      lhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(lhs.dtype)")
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.binaryOp(
        try await lhs.data, rhs, op: .mul, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    if !lhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype) { grad in
        lhsHandle.backward(backend) { grad * rhs }
      }
    }
  }

  public static func * <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    return rhs * lhs
  }

  public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with * operator")
    alwaysAssert(
      lhs.dtype == rhs.dtype, "dtypes for * operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let (outputShape, ((lhs, lhsStrides), (rhs, rhsStrides))) = Tensor.lazyBroadcast(lhs, rhs)

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.binaryOp(
        a: BroadcastData(strides: lhsStrides, data: try await lhs.data),
        b: BroadcastData(strides: rhsStrides, data: try await rhs.data),
        op: .mul, count: outputShape.product(), dtype: lhs.dtype)
    }
    if !Tensor.isGradEnabled || (!lhs.needsGrad && !rhs.needsGrad) {
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype) { grad in
        lhsHandle.backward(backend) { (grad * rhs.noGrad()).reduceBroadcast(lhsStrides, as: lhs) }
        rhsHandle.backward(backend) { (grad * lhs.noGrad()).reduceBroadcast(rhsStrides, as: rhs) }
      }
    }
  }

  public static func - <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    alwaysAssert(
      lhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(lhs.dtype)")
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with - operator")
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.binaryOp(
        try await lhs.data, rhs, op: .sub, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    if !lhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      return Tensor(
        dataTask: newData, shape: lhs.shape, dtype: lhs.dtype
      ) { grad in
        lhsHandle.backward(backend) { grad }
      }
    }
  }

  public static func - <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    alwaysAssert(
      rhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(rhs.dtype)")
    let backend = Backend.current
    let newData = createDataTask(rhs) { rhs in
      try await backend.binaryOp(
        lhs, try await rhs.data, op: .sub, count: rhs.shape.product(), dtype: rhs.dtype)
    }
    if !rhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: rhs.shape, dtype: rhs.dtype)
    } else {
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: rhs.shape, dtype: rhs.dtype) { grad in
        rhsHandle.backward(backend) { -grad }
      }
    }
  }

  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with + operator")
    alwaysAssert(
      lhs.dtype == rhs.dtype, "dtypes for - operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let (outputShape, ((lhs, lhsStrides), (rhs, rhsStrides))) = Tensor.lazyBroadcast(lhs, rhs)

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.binaryOp(
        a: BroadcastData(strides: lhsStrides, data: try await lhs.data),
        b: BroadcastData(strides: rhsStrides, data: try await rhs.data),
        op: .sub, count: outputShape.product(), dtype: lhs.dtype)
    }
    if !Tensor.isGradEnabled || (!lhs.needsGrad && !rhs.needsGrad) {
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype) { grad in
        lhsHandle.backward(backend) { grad.reduceBroadcast(lhsStrides, as: lhs) }
        rhsHandle.backward(backend) { -grad.reduceBroadcast(rhsStrides, as: rhs) }
      }
    }
  }

  prefix public static func - (t: Tensor) -> Tensor {
    alwaysAssert(t.dtype.isNumeric, "dtype \(t.dtype) cannot be used with - operator")
    return t * -1
  }

  public static func / <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    alwaysAssert(
      lhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(lhs.dtype)")
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with / operator")
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.binaryOp(
        try await lhs.data, rhs, op: .div, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    if !lhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype) { grad in
        lhsHandle.backward(backend) { grad / rhs }
      }
    }
  }

  public static func / <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    alwaysAssert(
      rhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(rhs.dtype)")
    alwaysAssert(rhs.dtype.isNumeric, "dtype \(rhs.dtype) cannot be used with / operator")
    let backend = Backend.current
    let newData = createDataTask(rhs) { rhs in
      try await backend.binaryOp(
        lhs, try await rhs.data, op: .div, count: rhs.shape.product(), dtype: rhs.dtype)
    }
    if !rhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: rhs.shape, dtype: rhs.dtype)
    } else {
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: rhs.shape, dtype: rhs.dtype) { grad in
        rhsHandle.backward(backend) { -lhs * rhs.noGrad().pow(-2) * grad }
      }
    }
  }

  public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with + operator")
    alwaysAssert(
      lhs.dtype == rhs.dtype, "dtypes for - operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let (outputShape, ((lhs, lhsStrides), (rhs, rhsStrides))) = Tensor.lazyBroadcast(lhs, rhs)

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.binaryOp(
        a: BroadcastData(strides: lhsStrides, data: try await lhs.data),
        b: BroadcastData(strides: rhsStrides, data: try await rhs.data),
        op: .div, count: outputShape.product(), dtype: lhs.dtype)
    }
    if !Tensor.isGradEnabled || (!lhs.needsGrad && !rhs.needsGrad) {
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype) { grad in
        lhsHandle.backward(backend) { (grad / rhs.noGrad()).reduceBroadcast(lhsStrides, as: lhs) }
        rhsHandle.backward(backend) {
          (-lhs.noGrad() * (rhs.noGrad().pow(-2) * grad)).reduceBroadcast(rhsStrides, as: rhs)
        }
      }
    }
  }

  public static func % <T: NumericTensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    alwaysAssert(
      lhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(lhs.dtype)")
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with % operator")
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.binaryOp(
        try await lhs.data, rhs, op: .mod, count: lhs.shape.product(), dtype: lhs.dtype)
    }
    if !lhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: lhs.shape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      return Tensor(
        dataTask: newData, shape: lhs.shape, dtype: lhs.dtype
      ) { grad in
        lhsHandle.backward(backend) { grad }
      }
    }
  }

  public static func % <T: NumericTensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    alwaysAssert(
      rhs.dtype.canUseScalarType(T.self),
      "scalar type \(T.self) cannot be used with dtype \(rhs.dtype)")
    let backend = Backend.current
    let newData = createDataTask(rhs) { rhs in
      try await backend.binaryOp(
        lhs, try await rhs.data, op: .mod, count: rhs.shape.product(), dtype: rhs.dtype)
    }
    if !rhs.needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: rhs.shape, dtype: rhs.dtype)
    } else {
      let rhsHandle = rhs.saveForBackward()
      return Tensor(dataTask: newData, shape: rhs.shape, dtype: rhs.dtype) { grad in
        rhsHandle.backward(backend) { -grad * (lhs % rhs.noGrad() == 0).cast(as: grad) }
      }
    }
  }

  public static func % (lhs: Tensor, rhs: Tensor) -> Tensor {
    alwaysAssert(lhs.dtype.isNumeric, "dtype \(lhs.dtype) cannot be used with + operator")
    alwaysAssert(
      lhs.dtype == rhs.dtype, "dtypes for - operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let (outputShape, ((lhs, lhsStrides), (rhs, rhsStrides))) = Tensor.lazyBroadcast(lhs, rhs)

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.binaryOp(
        a: BroadcastData(strides: lhsStrides, data: try await lhs.data),
        b: BroadcastData(strides: rhsStrides, data: try await rhs.data),
        op: .mod, count: outputShape.product(), dtype: lhs.dtype)
    }
    if !Tensor.isGradEnabled || (!lhs.needsGrad && !rhs.needsGrad) {
      return Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype)
    } else {
      let lhsHandle = lhs.saveForBackward()
      let rhsHandle = rhs.saveForBackward()
      let output = Tensor(dataTask: newData, shape: outputShape, dtype: lhs.dtype)
      return output.onGrad { grad in
        lhsHandle.backward(backend) { grad.reduceBroadcast(lhsStrides, as: lhs) }
        rhsHandle.backward(backend) {
          (-grad * (output == 0).cast(as: grad)).reduceBroadcast(rhsStrides, as: rhs)
        }
      }
    }
  }

  public func pow<T: NumericTensorElement>(_ exponent: T, outScale: T = 1.0) -> Tensor {
    alwaysAssert(dtype.isNumeric, "cannot use pow() with dtype \(dtype)")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.pow(
        try await t.data, exponent, outScale: outScale, count: t.shape.product(), dtype: t.dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    } else {
      let lhsHandle = saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        lhsHandle.backward(backend) {
          if exponent == T(2.0) {
            grad * self.noGrad() * 2
          } else {
            grad * self.noGrad().pow(exponent - T(1.0), outScale: exponent)
          }
        }
      }
    }
  }

  public func clamp<T: NumericTensorElement>(min: T? = nil, max: T? = nil) -> Tensor {
    alwaysAssert(dtype.isNumeric, "cannot use clamp() with dtype \(dtype)")
    alwaysAssert(min != nil || max != nil, "cannot use clamp() without bounds")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.clamp(
        try await t.data, min: min, max: max, count: t.shape.product(), dtype: t.dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    } else {
      let handle = saveForBackward()
      let rawResult = Tensor(dataTask: newData, shape: shape, dtype: dtype)
      return rawResult.onGrad { grad in
        handle.backward(backend) { (rawResult == self).when(isTrue: grad, isFalse: 0) }
      }
    }
  }

  internal static func compare(lhs: Tensor, rhs: Tensor, op: ComparisonOp) -> Tensor {
    alwaysAssert(
      lhs.shape == rhs.shape,
      "shape mismatch for == operator: lhs=\(lhs.shape) rhs=\(rhs.shape)"
    )
    alwaysAssert(
      lhs.dtype == rhs.dtype, "dtypes for == operator do not match: \(lhs.dtype) and \(rhs.dtype)")

    let backend = Backend.current
    let newData = createDataTask(lhs, rhs) { lhs, rhs in
      try await backend.compare(
        try await lhs.data, try await rhs.data, op: op, count: lhs.shape.product(), dtype: lhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: lhs.shape, dtype: .bool)
  }

  internal static func compare<T: TensorElement>(lhs: Tensor, rhs: T, op: ComparisonOp) -> Tensor {
    let backend = Backend.current
    let newData = createDataTask(lhs) { lhs in
      try await backend.compare(
        try await lhs.data, rhs, op: op, count: lhs.shape.product(), dtype: lhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: lhs.shape, dtype: .bool)
  }

  internal static func compare<T: TensorElement>(lhs: T, rhs: Tensor, op: ComparisonOp) -> Tensor {
    let backend = Backend.current
    let newData = createDataTask(rhs) { rhs in
      try await backend.compare(
        lhs, try await rhs.data, op: op, count: rhs.shape.product(), dtype: rhs.dtype
      )
    }
    return Tensor(dataTask: newData, shape: rhs.shape, dtype: .bool)
  }

  /*
  for op, name in [
    ("==", "equal"),
    ("<", "less"),
    (">", "greater"),
    ("<=", "lessEqual"),
    (">=", "greaterEqual"),
  ]:
    print(
        f"""
  public static func {op} <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {{
    compare(lhs: lhs, rhs: rhs, op: .{name})
  }}

  public static func {op} <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {{
    compare(lhs: lhs, rhs: rhs, op: .{name})
  }}

  public static func {op} (lhs: Tensor, rhs: Tensor) -> Tensor {{
    compare(lhs: lhs, rhs: rhs, op: .{name})
  }}
        """
    )
  */

  public static func == <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .equal)
  }

  public static func == <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .equal)
  }

  public static func == (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .equal)
  }

  public static func < <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .less)
  }

  public static func < <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .less)
  }

  public static func < (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .less)
  }

  public static func > <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greater)
  }

  public static func > <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greater)
  }

  public static func > (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greater)
  }

  public static func <= <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .lessEqual)
  }

  public static func <= <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .lessEqual)
  }

  public static func <= (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .lessEqual)
  }

  public static func >= <T: TensorElement>(lhs: Tensor, rhs: T) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greaterEqual)
  }

  public static func >= <T: TensorElement>(lhs: T, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greaterEqual)
  }

  public static func >= (lhs: Tensor, rhs: Tensor) -> Tensor {
    compare(lhs: lhs, rhs: rhs, op: .greaterEqual)
  }

  public func backward(_ grad: Tensor? = nil) {
    alwaysAssert(needsGrad, "backward called on Tensor that does not need grad")
    let grad =
      if let grad = grad {
        grad
      } else {
        Tensor(onesLike: self)
      }

    alwaysAssert(numBackwardHandles == 0, "cannot call backward() on tensor that is used elsewhere")
    Tensor.withGrad(enabled: true) { self.saveForBackward() }.backward(Backend.current) { grad }
  }

  public func saveForBackward() -> BackwardHandle {
    alwaysAssert(Tensor.isGradEnabled, "backward handle cannot be saved while grads are disabled")
    if !self.needsGrad {
      return BackwardHandle()
    } else if self.backwardImpl == nil {
      return BackwardHandle(
        addGrad: { _ in alwaysAssert(false, "cannot backward a second time") }, cancel: {})
    }

    numBackwardHandles += 1
    let backend = Backend.current

    return BackwardHandle(
      addGrad: { [self] grad in
        alwaysAssert(numBackwardHandles > 0)
        alwaysAssert(
          grad.shape == shape,
          "gradient shape \(grad.shape) must match tensor shape \(shape)"
        )
        alwaysAssert(
          grad.dtype == dtype, "gradient dtype \(grad.dtype) must match tensor dtype \(dtype)")
        if let cg = curGrad {
          curGrad = backend.use { cg + grad }
        } else {
          curGrad = grad
        }
        numBackwardHandles -= 1
        if let grad = curGrad, numBackwardHandles == 0 {
          let bwd = backwardImpl!
          backwardImpl = nil
          bwd(grad)
          curGrad = nil
        }
      },
      cancel: { [self] in
        numBackwardHandles -= 1
        if self.curGrad != nil && numBackwardHandles == 0 {
          alwaysAssert(false, "backward pass was incompleted due to an unused reference")
        }
      })
  }

  internal func positiveAxis(_ axis: Int?) -> Int? {
    if let axis = axis {
      // Type annotation needed to clarify overload of method.
      let result: Int = positiveAxis(axis)
      return result
    } else {
      return nil
    }
  }
  internal func positiveAxis(_ axis: Int) -> Int {
    let result = axis < 0 ? axis + shape.count : axis
    alwaysAssert(result >= 0, "axis \(axis) out of bounds for shape \(shape)")
    return result
  }

}
