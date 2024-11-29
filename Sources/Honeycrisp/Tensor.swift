import Foundation
import HCBacktrace

/// An asynchronous, multi-dimensional array of integral, floating point, or boolean values.
///
/// The type of data stored in a Tensor can be checked via the ``Tensor/dtype`` attribute.
///
/// Each Tensor has a corresponding shape, accessed via ``Tensor/shape``, which indicates the
/// dimensions of each axis of the Tensor. The shape may be empty, in which case the Tensor is
/// a single scalar value.
///
/// Tensors may be differentiable, which can be checked via ``Tensor/needsGrad``. When a
/// Tensor is differentiable, downstream operations will keep track of this Tensor and support
/// backpropagation through it for gradient calculations.
///
/// The data of a ``Tensor`` is computed asynchronously, and can be accessed via the
/// asynchronous ``Tensor/data`` attribute. The data itself might not exist on the CPU, and
/// may instead be stored in a place specific to the ``Backend`` which produced it. However,
/// all implementations of ``Tensor/Data`` must provide the ability to access the data on the
/// CPU via the ``Tensor/Data/onCPU(_:)`` and ``Tensor/Data/mutateOnCPU(_:)`` methods.
public final class Tensor {

  /// A type of numeric or boolean data stored in a ``Tensor`` object.
  public enum DType: Codable {
    case int64
    case bool
    case float16
    case float32

    public var supportsGrad: Bool {
      isFloat
    }

    public var isNumeric: Bool {
      self != .bool
    }

    public var isFloat: Bool {
      self == .float16 || self == .float32
    }

    public var byteSize: Int {
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

  /// An abstract protocol representing a glob of tensor data that may be stored anywhere and/or
  /// computed lazily.
  public protocol Data {
    /// Get the amount of allocated bytes. Note that the actual amount of data which is used
    /// by a tensor might be less than the size of the data.
    var byteCount: Int { get }

    /// Get the data as a concrete, CPU-mapped buffer.
    func onCPU<T>(_ fn: (_: UnsafeRawPointer) async throws -> T) async throws -> T

    /// Modify the data on the CPU.
    /// This should only be used by ``Backend`` implementations when creating an output.
    func mutateOnCPU<T>(_ fn: (_: UnsafeMutableRawPointer) async throws -> T) async throws -> T
  }

  /// A single-use reference to a ``Tensor`` that is stored in the forward pass and used in the
  /// backward pass to propagate gradients to the Tensor.
  ///
  /// These objects will typically be created via ``Tensor/saveForBackward(function:file:line:)``.
  ///
  /// When a `BackwardHandle` is created but never used, its `deinit` implementation will inform
  /// the underlying ``Tensor`` that one fewer operation is waiting for gradients.
  public final class BackwardHandle {
    private var addGrad: ((Tensor) -> Void)?
    private var cancel: (() -> Void)?
    private var canSkip: Bool

    private static let BackwardStackKey = "HONEYCRISP_BACKWARD_STACK"

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

    @recordCaller
    private func _backward(_ backend: Backend, _ gradFn: @escaping () -> Tensor) {
      typealias StackType = (Backend, (() -> Tensor), ((Tensor) -> Void))
      alwaysAssert(addGrad != nil, "cannot re-use backward handle")
      let ag = addGrad!
      addGrad = nil
      cancel = nil
      if canSkip {
        return
      }

      // Why do we need this manual stack, when we could just use the "real"
      // callstack to implement these recursive depth-first backwards?
      //
      // The answer is that our computation graph might be somewhat deep,
      // and we can actually hit stack overflows, even for somewhat small
      // models like a 12 layer Transformer.
      if let stack = Thread.current.threadDictionary[BackwardHandle.BackwardStackKey] {
        var newStack = (stack as! [StackType])
        newStack.append((backend, gradFn, ag))
        Thread.current.threadDictionary[BackwardHandle.BackwardStackKey] = newStack
      } else {
        var fullStack = [(backend, gradFn, ag)]
        while let (backend, gradFn, ag) = fullStack.popLast() {
          Thread.current.threadDictionary[BackwardHandle.BackwardStackKey] = [StackType]()
          let grad = backend.use { gradFn() }
          ag(grad)
          let newEntries =
            Thread.current.threadDictionary[BackwardHandle.BackwardStackKey]
            as! [StackType]
          Thread.current.threadDictionary.removeObject(forKey: BackwardHandle.BackwardStackKey)
          fullStack.append(contentsOf: newEntries.reversed())
        }
      }
    }

    deinit {
      if let cancel = cancel {
        cancel()
      }
    }
  }

  @TaskLocal
  private static var taskGradEnabled: Bool? = nil

  public static var isGradEnabled: Bool {
    taskGradEnabled ?? true
  }

  @recordCaller
  private static func _withGrad<T>(enabled flag: Bool, _ fn: () async throws -> T) async rethrows
    -> T
  {
    try await $taskGradEnabled.withValue(flag) { try await fn() }
  }

  @recordCaller
  private static func _withGrad<T>(enabled flag: Bool, _ fn: () throws -> T) rethrows -> T {
    try $taskGradEnabled.withValue(flag) { try fn() }
  }

  @TaskLocal
  private static var taskDataDependencies: [Task<(), Error>]? = nil

  /// Run a block such that all of the `Tensor`s created within the block will
  /// wait for the current `Tensor`'s data to be available before beginning
  /// execution of any backend routines.
  ///
  /// Note that, if the backward pass is not performed within the block, then
  /// tensors in the backward pass may, in theory, not wait for the data. This
  /// is unlikely, however, since gradients typically depend on results from
  /// the forward pass, which will correctly wait for the data.
  ///
  /// This has no effect on `Tensor`s explicitly created with the `dataTask:`
  /// initializer. It only affects data tasks created with the `createDataTask`
  /// helpers, which should be used internally whenever a backend method is
  /// called.
  public func asDependency<T>(waitForCPU: Bool = true, _ fn: () async throws -> T) async rethrows
    -> T
  {
    let t = self.noGrad()
    let task = Task {
      let data = try await t.data
      if waitForCPU {
        try await data.onCPU { _ in () }
      }
    }

    return try await Tensor.$taskDataDependencies.withValue(
      (Tensor.taskDataDependencies ?? []) + [task]
    ) {
      try await fn()
    }
  }

  /// Run a block such that all of the `Tensor`s created within the block will
  /// wait for the current `Tensor`'s data to be available before beginning
  /// execution of any backend routines.
  ///
  /// Note that, if the backward pass is not performed within the block, then
  /// tensors in the backward pass may, in theory, not wait for the data. This
  /// is unlikely, however, since gradients typically depend on results from
  /// the forward pass, which will correctly wait for the data.
  ///
  /// This has no effect on `Tensor`s explicitly created with the `dataTask:`
  /// initializer. It only affects data tasks created with the `createDataTask`
  /// helpers, which should be used internally whenever a backend method is
  /// called.
  public func asDependency<T>(waitForCPU: Bool = true, _ fn: () throws -> T) rethrows -> T {
    let t = self.noGrad()
    let task = Task {
      let data = try await t.data
      if waitForCPU {
        try await data.onCPU { _ in () }
      }
    }

    return try Tensor.$taskDataDependencies.withValue((Tensor.taskDataDependencies ?? []) + [task])
    {
      try fn()
    }
  }

  /// The `Task` which is responsible for computing the data of the Tensor.
  ///
  /// Rather than accessing this directly, you may use the ``Tensor/data`` attribute.
  public let dataTask: Task<Data, Error>

  /// The dimensions of each axis of the Tensor.
  ///
  /// For a normal array of values, this will look like `[N]` where `N` is the total count.
  /// For a matrix, this could be `[rows, columns]`.
  ///
  /// The shape may be empty, in which case the Tensor is a single scalar value.
  public let shape: [Int]

  /// The type of data stored as elements in this ``Tensor``.
  public let dtype: DType

  /// Whether or not this ``Tensor`` can handle backpropagation.
  ///
  /// When this is `true`, downstream operations should store a handle to this tensor by calling
  /// ``saveForBackward(function:file:line:)`` and then calling the resulting handle's
  /// ``BackwardHandle/backward(_:_:function:file:line:)`` method in the backward pass.
  public let needsGrad: Bool

  /// The asynchronously computed data representing the values of this ``Tensor``.
  public var data: Data {
    get async throws {
      try await dataTask.value
    }
  }

  private class BackwardState {
    public var backwardImpl: (((Tensor) -> Void))?
    public var shape: [Int]
    public var dtype: DType

    // State used during backward pass to accumulate gradients.
    public var curGrad: Tensor? = nil

    // Incremented when references are created, and decremented when
    // references are destroyed.
    public var numBackwardHandles: Int = 0

    public init(backwardImpl: (((Tensor) -> Void))?, shape: [Int], dtype: DType) {
      self.backwardImpl = backwardImpl
      self.shape = shape
      self.dtype = dtype
    }

    @recordCaller
    private func _createHandle() -> BackwardHandle {
      numBackwardHandles += 1

      let backend = Backend.current

      let creationTrace = Backtrace.trace()

      return BackwardHandle(
        addGrad: { [self] grad in addGrad(backend, grad) },
        cancel: { [self] in
          numBackwardHandles -= 1
          if curGrad != nil && numBackwardHandles == 0 {
            alwaysAssert(
              false,
              "backward pass was incompleted due to an unused reference.\n\nTraceback of reference creation:\n\n\(Backtrace.format(creationTrace))"
            )
          }
        })
    }

    @recordCaller
    internal func _addGrad(_ backend: Backend, _ grad: Tensor) {
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
    }
  }

  private let backwardState: BackwardState

  public init(
    dataTask: Task<Data, Error>,
    shape: [Int],
    dtype: DType,
    backwardImpl: (((Tensor) -> Void))? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    #if DEBUG
      // This can be helpful to catch a common error case where a backend
      // accidentally miscalculates the size of a buffer and silently
      // overflows it.
      self.dataTask = Tensor.createDataTask(
        {
          let result = try await dataTask.value
          let allocSize = result.byteCount
          let minSize = shape.product() * dtype.byteSize
          alwaysAssert(
            allocSize >= minSize, "buffer of size \(allocSize) underflows shape \(shape)")
          return result
        }, function: function, file: file, line: line)
    #else
      self.dataTask = dataTask
    #endif
    self.shape = shape
    self.dtype = dtype
    if Tensor.isGradEnabled {
      self.backwardState = BackwardState(backwardImpl: backwardImpl, shape: shape, dtype: dtype)
      self.needsGrad = backwardImpl != nil
    } else {
      Backtrace.record(function: function, file: file, line: line) {
        alwaysAssert(
          backwardImpl == nil, "cannot provide a backward function if !Tensor.isGradEnabled")
      }
      self.needsGrad = false
      self.backwardState = BackwardState(backwardImpl: nil, shape: shape, dtype: dtype)
    }
  }

  /// Create a `Tensor` with the elements of a collection.
  ///
  /// If `shape` is specified, the product of the values within the shape must match
  /// the count of the collection.
  /// Otherwise, the default 1-dimensional shape `[data.count]` is used.
  ///
  /// The `reverse` argument may be used to flip the order of the data in the collection.
  convenience public init<T: TensorElement>(
    data: some Collection<T>,
    shape: [Int]? = nil,
    dtype: DType? = nil,
    reverse: Bool = false,
    backwardImpl: ((Tensor) -> Void)? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    let shape = shape ?? [data.count]
    let dtype = dtype ?? T.dtype
    if !dtype.supportsGrad {
      alwaysAssert(backwardImpl == nil, "cannot specify gradient for dtype \(dtype)")
    }
    alwaysAssert(
      data.count == shape.product(), "data count \(data.count) does not match shape \(shape)")
    alwaysAssert(
      dtype.canUseScalarType(T.self),
      "cannot create Tensor with dtype \(dtype) with scalar type \(T.self)")
    let dataTask = Backtrace.record(function: function, file: file, line: line) {
      let backend = Backend.current
      return Tensor.createDataTask {
        try await backend.collection(data, reverse: reverse, dtype: dtype)
      }
    }
    self.init(dataTask: dataTask, shape: shape, dtype: dtype, backwardImpl: backwardImpl)
  }

  public convenience init(
    zerosLike: Tensor,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(constant: 0, like: zerosLike, function: function, file: file, line: line)
  }

  public convenience init(
    onesLike: Tensor,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(constant: 1, like: onesLike, function: function, file: file, line: line)
  }

  public convenience init(
    zeros shape: [Int],
    dtype: DType = .float32,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(constant: 0, shape: shape, dtype: dtype, function: function, file: file, line: line)
  }

  public convenience init(
    ones shape: [Int],
    dtype: DType = .float32,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(constant: 1, shape: shape, dtype: dtype, function: function, file: file, line: line)
  }

  public convenience init<T: TensorElement>(
    constant: T,
    like: Tensor,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    self.init(
      constant: constant, shape: like.shape, dtype: like.dtype, function: function, file: file,
      line: line)
  }

  public convenience init<T: TensorElement>(
    constant: T,
    shape: [Int],
    dtype: DType? = nil,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    let dtype = dtype ?? T.dtype
    alwaysAssert(
      dtype.canUseScalarType(T.self),
      "cannot create Tensor with dtype \(dtype) with scalar type \(T.self)")
    let dataTask = Backtrace.record(function: function, file: file, line: line) {
      let backend = Backend.current
      return Tensor.createDataTask {
        try await backend.constant(constant, count: shape.product(), dtype: dtype)
      }
    }
    self.init(dataTask: dataTask, shape: shape, dtype: dtype)
  }

  @recordCaller
  private func _copyToArray<T: TensorElement>(_ out: inout [T]) async throws {
    alwaysAssert(out.count == shape.product(), "out size must match our size")
    try await data.onCPU { buf in
      try pointerToArray(buf, output: &out, dtype: dtype)
    }
  }

  @recordCaller
  private func _floats() async throws -> [Float] {
    var out = [Float](repeating: 0, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  @recordCaller
  private func _ints() async throws -> [Int] {
    var out = [Int](repeating: 0, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  @recordCaller
  private func _int64s() async throws -> [Int64] {
    var out = [Int64](repeating: 0, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  @recordCaller
  private func _bools() async throws -> [Bool] {
    var out = [Bool](repeating: false, count: shape.product())
    try await copyToArray(&out)
    return out
  }

  @recordCaller
  private func _item() async throws -> Float {
    alwaysAssert(shape.product() == 1, "cannot call item() on Tensor of shape \(shape)")
    let data = try await floats()
    return data[0]
  }

  @recordCaller
  private func _wait() async throws {
    let _ = try await (try await data).onCPU { _ in () }
  }

  public func noGrad() -> Tensor {
    return Tensor(dataTask: dataTask, shape: shape, dtype: dtype)
  }

  @recordCaller
  internal func _createDataTask(_ fn: @escaping (Tensor) async throws -> Data) -> Task<Data, Error>
  {
    Tensor.createDataTask(self, fn)
  }

  static internal func createDataTask<T>(
    _ fn: @escaping () async throws -> T,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) -> Task<T, Error> {
    let callers = Backtrace.current + [CodeLocation(function: function, file: file, line: line)]
    return if let deps = taskDataDependencies {
      Task {
        try await Backtrace.override(callers) {
          for dep in deps {
            let _ = try await dep.value
          }
          return try await fn()
        }
      }
    } else {
      Task {
        try await Backtrace.override(callers) {
          try await fn()
        }
      }
    }
  }

  @recordCaller
  static internal func _createDataTask(
    _ x: Tensor, _ fn: @escaping (Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = x.noGrad()
    return createDataTask {
      try await fn(safeRef1)
    }
  }

  @recordCaller
  static internal func _createDataTask(
    _ x: Tensor, _ y: Tensor, _ fn: @escaping (Tensor, Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = x.noGrad()
    let safeRef2 = y.noGrad()
    return createDataTask {
      try await fn(safeRef1, safeRef2)
    }
  }

  @recordCaller
  static internal func _createDataTask(
    _ x: Tensor, _ y: Tensor, _ z: Tensor,
    _ fn: @escaping (Tensor, Tensor, Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = x.noGrad()
    let safeRef2 = y.noGrad()
    let safeRef3 = z.noGrad()
    return createDataTask {
      try await fn(safeRef1, safeRef2, safeRef3)
    }
  }

  @recordCaller
  static internal func _createDataTask(
    _ w: Tensor, _ x: Tensor, _ y: Tensor, _ z: Tensor,
    _ fn: @escaping (Tensor, Tensor, Tensor, Tensor) async throws -> Data
  ) -> Task<Data, Error> {
    let safeRef1 = w.noGrad()
    let safeRef2 = x.noGrad()
    let safeRef3 = y.noGrad()
    let safeRef4 = z.noGrad()
    return createDataTask {
      try await fn(safeRef1, safeRef2, safeRef3, safeRef4)
    }
  }

  @recordCaller
  private func _onGrad(_ action: @escaping ((Tensor) -> Void)) -> Tensor {
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

  @recordCaller
  private func _reshape(_ newShape: [Int]) -> Tensor {
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
        let shape = shape
        handle.backward(Backend.current) { grad.reshape(shape) }
      }
    }
  }

  @recordCaller
  private func _reshape(as x: Tensor) -> Tensor {
    reshape(x.shape)
  }

  @recordCaller
  internal func _fillNegativeOneInShape(_ newShape: [Int]) -> [Int] {
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

  @recordCaller
  private func _flatten(startAxis: Int = 0, endAxis: Int = -1) -> Tensor {
    let startAxis = positiveAxis(startAxis)
    let endAxis = shape.count == 0 && endAxis == -1 ? 0 : positiveAxis(endAxis)
    if startAxis == 0 && endAxis == 0 && shape.count == 0 {
      return reshape([1])
    }
    alwaysAssert(endAxis >= startAxis, "invalid axes for flatten(): [\(startAxis), \(endAxis)]")
    return reshape(
      shape[..<startAxis] + [shape[startAxis...endAxis].product()] + shape[(endAxis + 1)...])
  }

  @recordCaller
  private func _squeeze(axis: Int) -> Tensor {
    let posAxis = positiveAxis(axis)
    alwaysAssert(shape[posAxis] == 1, "cannot squeeze axis \(axis) for shape \(shape)")
    var newShape = shape
    newShape.remove(at: posAxis)
    return reshape(newShape)
  }

  @recordCaller
  private func _unsqueeze(axis: Int) -> Tensor {
    let posAxis = axis < 0 ? axis + shape.count + 1 : axis
    alwaysAssert(posAxis >= 0, "axis \(axis) out of bounds for shape \(shape)")
    var newShape = shape
    newShape.insert(1, at: posAxis)
    return reshape(newShape)
  }

  @recordCaller
  private func _cast(as t: Tensor) -> Tensor {
    cast(t.dtype)
  }

  @recordCaller
  private func _cast(_ newType: DType) -> Tensor {
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
        BroadcastData(strides: lhsStrides, data: try await lhs.data),
        BroadcastData(strides: rhsStrides, data: try await rhs.data),
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
        BroadcastData(strides: lhsStrides, data: try await lhs.data),
        BroadcastData(strides: rhsStrides, data: try await rhs.data),
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
        BroadcastData(strides: lhsStrides, data: try await lhs.data),
        BroadcastData(strides: rhsStrides, data: try await rhs.data),
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
        BroadcastData(strides: lhsStrides, data: try await lhs.data),
        BroadcastData(strides: rhsStrides, data: try await rhs.data),
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
        BroadcastData(strides: lhsStrides, data: try await lhs.data),
        BroadcastData(strides: rhsStrides, data: try await rhs.data),
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

  @recordCaller
  private func _pow<T: NumericTensorElement>(_ exponent: T) -> Tensor {
    alwaysAssert(dtype.isNumeric, "cannot use pow() with dtype \(dtype)")
    let backend = Backend.current
    let newData = createDataTask { t in
      try await backend.pow(
        try await t.data,
        exponent,
        scale: T(1.0),
        scales: nil,
        count: t.shape.product(),
        dtype: t.dtype)
    }
    if !needsGrad || !Tensor.isGradEnabled {
      return Tensor(dataTask: newData, shape: shape, dtype: dtype)
    } else {
      let lhsHandle = saveForBackward()
      return Tensor(dataTask: newData, shape: shape, dtype: dtype) { grad in
        lhsHandle.backward(backend) { self.noGrad().powGrad(exponent, grad: grad) }
      }
    }
  }

  @recordCaller
  internal func _powGrad<T: NumericTensorElement>(_ exponent: T, grad: Tensor) -> Tensor {
    alwaysAssert(dtype.isNumeric, "cannot use pow() with dtype \(dtype)")
    alwaysAssert(
      grad.dtype == dtype,
      "dtype of self \(dtype) does not match dtype of gradient \(grad.dtype)")
    alwaysAssert(!self.needsGrad && !grad.needsGrad, "second derivatives are not supported")
    let backend = Backend.current
    let newData = Tensor.createDataTask(self, grad) { t, grad in
      try await backend.pow(
        try await t.data, exponent - T(1.0), scale: exponent, scales: try await grad.data,
        count: t.shape.product(), dtype: t.dtype)
    }
    return Tensor(dataTask: newData, shape: shape, dtype: dtype)
  }

  @recordCaller
  private func _clamp<T: NumericTensorElement>(min: T? = nil, max: T? = nil) -> Tensor {
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

  @recordCaller
  private func _backward(_ grad: Tensor? = nil) {
    alwaysAssert(needsGrad, "backward called on Tensor that does not need grad")
    let grad =
      if let grad = grad {
        grad
      } else {
        Tensor(onesLike: self)
      }

    alwaysAssert(
      backwardState.numBackwardHandles == 0,
      "cannot call backward() on tensor that is used elsewhere")
    Tensor.withGrad(enabled: true) { self.saveForBackward() }.backward(Backend.current) { grad }
  }

  @recordCaller
  private func _saveForBackward() -> BackwardHandle {
    alwaysAssert(Tensor.isGradEnabled, "backward handle cannot be saved while grads are disabled")
    if !self.needsGrad {
      return BackwardHandle()
    } else if backwardState.backwardImpl == nil {
      return BackwardHandle(
        addGrad: { _ in alwaysAssert(false, "cannot backward a second time") }, cancel: {})
    }
    return backwardState.createHandle()
  }

  @recordCaller
  internal func _positiveAxis(_ axis: Int?) -> Int? {
    if let axis = axis {
      // Type annotation needed to clarify overload of method.
      let result: Int = positiveAxis(axis)
      return result
    } else {
      return nil
    }
  }

  @recordCaller
  internal func _positiveAxis(_ axis: Int) -> Int {
    let result = axis < 0 ? axis + shape.count : axis
    alwaysAssert(result >= 0, "axis \(axis) out of bounds for shape \(shape)")
    return result
  }

}
