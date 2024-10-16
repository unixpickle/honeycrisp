import Foundation

public protocol MaybeTensor {
  var isNil: Bool { get }

  func maybeTensor() -> Tensor?

  func maybeOnGrad(_ action: @escaping (Tensor) -> Void) -> Self

  static func fromTensor(_ x: Tensor) -> Self
}

extension Tensor: MaybeTensor {
  public var isNil: Bool {
    false
  }

  public func maybeTensor() -> Tensor? {
    self
  }

  public func maybeOnGrad(_ action: @escaping (Tensor) -> Void) -> Self {
    onGrad(action) as! Self
  }

  public static func fromTensor(_ x: Tensor) -> Self {
    x as! Self
  }
}

extension Tensor?: MaybeTensor {
  public var isNil: Bool {
    self == nil
  }

  public func maybeTensor() -> Tensor? {
    self
  }

  public func maybeOnGrad(_ action: @escaping (Tensor) -> Void) -> Self {
    if let self = self {
      self.onGrad(action)
    } else {
      nil
    }
  }

  public static func fromTensor(_ x: Tensor) -> Self {
    x
  }
}

open class Trainable {
  public enum Mode {
    case training
    case inference
  }

  public protocol Parameter {
    var name: String? { get }
    var data: Tensor? { get set }
    var grad: Tensor? { get set }
  }

  @propertyWrapper
  public final class Param<TensorType: MaybeTensor>: Parameter {
    public static subscript<T: Trainable>(
      _enclosingInstance instance: T,
      wrapped wrappedKeyPath: ReferenceWritableKeyPath<T, TensorType>,
      storage storageKeyPath: ReferenceWritableKeyPath<T, Param>
    ) -> TensorType {
      get {
        let param = instance[keyPath: storageKeyPath]
        return instance[keyPath: storageKeyPath].maybeData!.maybeOnGrad { g in
          if let existingGrad = param.grad {
            param.grad = existingGrad + g
          } else {
            param.grad = g
          }
        }
      }
      set {
        let param = instance[keyPath: storageKeyPath]
        if param.maybeName == nil {
          param.maybeName = String("\(storageKeyPath)".split(separator: ".").last!)
        }
        alwaysAssert(
          !(newValue.maybeTensor()?.needsGrad ?? false), "parameter value cannot need grad")
        param.maybeData = newValue
        if newValue.isNil {
          instance.registeredParams.removeValue(forKey: param.name!)
        } else {
          instance.registeredParams[param.name!] = (param as Parameter)
        }
      }
    }

    @available(
      *, unavailable,
      message: "@Parameter can only be applied to classes"
    )
    public var wrappedValue: TensorType {
      get { fatalError() }
      set { fatalError() }
    }

    private var maybeName: String? = nil
    public var name: String? {
      maybeName
    }
    private var maybeData: TensorType?
    public var data: Tensor? {
      get {
        maybeData?.maybeTensor()
      }
      set {
        if let t = newValue {
          maybeData = TensorType.fromTensor(t)
        } else {
          alwaysAssert(
            false, "Cannot unset parameter data. Consider setting the property itself to nil.")
        }
      }
    }

    public var grad: Tensor?

    public var projectedValue: Param<TensorType> { self }

    public init(name: String? = nil) {
      maybeName = name
    }
  }

  @propertyWrapper
  public final class Child<Value: Trainable> {
    public static subscript<T: Trainable>(
      _enclosingInstance instance: T,
      wrapped wrappedKeyPath: ReferenceWritableKeyPath<T, Value>,
      storage storageKeyPath: ReferenceWritableKeyPath<T, Child>
    ) -> Value {
      get {
        instance[keyPath: storageKeyPath].value!
      }
      set {
        let child = instance[keyPath: storageKeyPath]
        if child.name == nil {
          child.name = String("\(storageKeyPath)".split(separator: ".").last!)
        }
        child.value = newValue
        instance.registeredChildren[child.name!] = newValue
      }
    }

    @available(
      *, unavailable,
      message: "@Child can only be applied to classes"
    )
    public var wrappedValue: Value {
      get { fatalError() }
      set { fatalError() }
    }

    var name: String? = nil
    private var value: Value?

    public init(name: String? = nil) {
      self.name = name
    }
  }

  internal var registeredParams = [String: Parameter]()
  internal var registeredChildren = [String: Trainable]()

  internal var _mode: Mode = .training

  public var parameters: [(String, Parameter)] {
    var results = Array(registeredParams)
    for (name, child) in registeredChildren {
      results += child.parameters.map { (subName, param) in
        ("\(name).\(subName)", param)
      }
    }
    return results.sorted(by: { $0.0 < $1.0 })
  }

  public var mode: Mode {
    get {
      _mode
    }
    set {
      _mode = newValue
      for child in registeredChildren.values {
        child.mode = newValue
      }
    }
  }

  public init() {
  }

  public func withMode<T>(_ mode: Mode, _ action: () throws -> T) rethrows -> T {
    let oldMode = self.mode
    self.mode = mode
    defer { self.mode = oldMode }
    return try action()
  }

  public enum StateItem: Codable {
    case tensor(TensorState)
    case child([String: StateItem])
  }

  public typealias State = [String: StateItem]

  public func state() async throws -> State {
    var result = State()
    for (name, param) in registeredParams {
      if let data = param.data {
        result[name] = .tensor(try await data.state())
      }
    }
    for (name, child) in registeredChildren {
      result[name] = .child(try await child.state())
    }
    return result
  }

  public enum LoadStateError: Error {
    case stateMissingKey(String)
    case missingParameterInState(String)
    case unexpectedType(String)
    case shapeMismatch(String)
    case dtypeMismatch(String)
  }

  public func loadState(
    _ state: State, mustSetAllParameters: Bool = true, mustUseAllStates: Bool = true
  ) throws {
    var state = state
    for (name, var param) in registeredParams {
      if !state.keys.contains(name) {
        if mustSetAllParameters {
          throw LoadStateError.stateMissingKey(
            "the state has no parameter key \(name) while the module does")
        }
      } else if case .tensor(let x) = state[name]! {
        if let data = param.data {
          if data.dtype != x.dtype {
            throw LoadStateError.dtypeMismatch(
              "key \(name) has parameter dtype \(data.dtype) but dtype in state is \(x.dtype)")
          } else if data.shape != x.shape {
            throw LoadStateError.shapeMismatch(
              "key \(name) has parameter shape \(data.shape) but shape in state is \(x.shape)")
          }
        }
        param.data = Tensor(state: x)
        state.removeValue(forKey: name)
      } else {
        throw LoadStateError.unexpectedType(
          "expected parameter for \(name) but got another kind of object")
      }
    }
    for (name, child) in registeredChildren {
      if !state.keys.contains(name) {
        if mustSetAllParameters {
          throw LoadStateError.stateMissingKey(
            "the state has no child key \(name) while the module does")
        }
      } else if case .child(let x) = state[name]! {
        try child.loadState(
          x, mustSetAllParameters: mustSetAllParameters, mustUseAllStates: mustUseAllStates)
        state.removeValue(forKey: name)
      } else {
        throw LoadStateError.unexpectedType(
          "expected child for \(name) but got another kind of object")
      }
    }
    if mustUseAllStates && !state.keys.isEmpty {
      throw LoadStateError.missingParameterInState(
        "state has keys which do not correspond to keys in the model: \(state.keys)")
    }
  }

}

public class TrainableArray<T: Trainable>: Trainable {
  public let children: [T]

  public init(_ children: [T]) {
    self.children = children
    super.init()
    for (i, ch) in children.enumerated() {
      self.registeredChildren[String(i)] = ch
    }
  }
}

public class Linear: Trainable {
  @Param(name: "weight") public var weight: Tensor
  @Param(name: "bias") public var bias: Tensor?

  public let castParams: Tensor.DType?

  public init(
    inCount: Int, outCount: Int, dtype: Tensor.DType = .float32, castParams: Tensor.DType? = nil,
    bias: Bool = true
  ) {
    self.castParams = castParams
    super.init()
    self.weight =
      (Tensor(rand: [inCount, outCount], dtype: dtype) - 0.5)
      * (sqrt(3.0) / 0.5 / sqrt(Float(inCount)))
    if bias {
      self.bias = Tensor(zeros: [outCount])
    }
  }

  public func callAsFunction(_ x: Tensor) -> Tensor {
    if x.shape.count > 2 {
      let squashedBatch = x.reshape([
        x.shape[..<(x.shape.count - 1)].product(), x.shape[x.shape.count - 1],
      ])
      let out = self(squashedBatch)
      return out.reshape(x.shape[..<(x.shape.count - 1)] + [out.shape[out.shape.count - 1]])
    }
    var h = x &* weight.cast(castParams ?? weight.dtype)
    if let bias = bias {
      h = h + bias.cast(castParams ?? bias.dtype).expand(as: h)
    }
    return h
  }
}

public class Conv2D: Trainable {
  public typealias Dim = Conv2DConfig.Dim
  public typealias Padding = Conv2DConfig.Padding

  public enum PaddingType {
    case none
    case same
    case allSides(Int)
    case xy(x: Int, y: Int)
    case leftRightTopBottom(Int, Int, Int, Int)
  }

  public enum SpatialSize {
    case square(Int)
    case widthHeight(Int, Int)

    public var dim: Dim {
      switch self {
      case .square(let x):
        Dim(x: x, y: x)
      case .widthHeight(let x, let y):
        Dim(x: x, y: y)
      }
    }
  }

  public let inChannels: Int
  public let outChannels: Int
  public let kernelSize: Dim
  public let stride: Dim
  public let dilation: Dim
  public let padding: Padding
  public let groups: Int
  public let channelsLast: Bool

  @Param(name: "weight") public var weight: Tensor
  @Param(name: "bias") public var bias: Tensor?

  public init(
    inChannels: Int, outChannels: Int, kernelSize: SpatialSize, stride: SpatialSize = .square(1),
    padding: PaddingType = .none, dilation: SpatialSize = .square(1), groups: Int = 1,
    channelsLast: Bool = false, bias: Bool = true, dtype: Tensor.DType = .float32
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize.dim
    self.stride = stride.dim
    self.dilation = dilation.dim
    alwaysAssert(
      inChannels % groups == 0, "outChannels \(outChannels) not divisible by groups \(groups)")
    alwaysAssert(
      outChannels % groups == 0, "inChannels \(inChannels) not divisible by groups \(groups)")
    switch padding {
    case .none:
      self.padding = Padding(before: Dim(constant: 0), after: Dim(constant: 0))
    case .same:
      alwaysAssert(
        self.stride == Dim(constant: 1),
        "cannot use padding mode 'same' with stride \(self.stride)")
      self.padding = Conv2DConfig.samePadding(kernelSize: self.kernelSize, dilation: self.dilation)
    case .allSides(let x):
      self.padding = Padding(before: Dim(constant: x), after: Dim(constant: x))
    case .xy(let x, let y):
      self.padding = Padding(before: Dim(x: x, y: y), after: Dim(x: x, y: y))
    case .leftRightTopBottom(let left, let right, let top, let bottom):
      self.padding = Padding(before: Dim(x: left, y: top), after: Dim(x: right, y: bottom))
    }
    self.groups = groups
    self.channelsLast = channelsLast
    super.init()
    self.weight =
      (Tensor(
        rand: [outChannels, inChannels / groups] + self.kernelSize.dims,
        dtype: dtype) - 0.5)
      * (sqrt(3.0) / 0.5 / sqrt(Float(inChannels * self.kernelSize.dims.product())))
    if bias {
      self.bias = Tensor(zeros: [outChannels])
    }
  }

  public func callAsFunction(_ x: Tensor) -> Tensor {
    alwaysAssert(x.shape.count == 4, "invalid input shape for conv2d: \(x.shape)")
    let (height, width, channels) =
      if channelsLast {
        (x.shape[1], x.shape[2], x.shape[3])
      } else {
        (x.shape[2], x.shape[3], x.shape[1])
      }
    alwaysAssert(
      channels == inChannels,
      "channels of input \(channels) doesn't match expected channels \(inChannels)")

    let convDesc: Conv2DConfig
    do {
      convDesc = try Conv2DConfig(
        inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize,
        imageSize: Dim(x: width, y: height), stride: stride, dilation: dilation, padding: padding,
        groups: groups, channelsLast: channelsLast)
    } catch {
      fatalError("failed to instantiate Conv2DConfig: \(error)")
    }

    var h = Tensor.conv2D(convDesc, image: x, kernel: weight)
    if let bias = bias {
      if channelsLast {
        h = h + bias.expand(as: h)
      } else {
        h = h + bias[..., NewAxis(), NewAxis()].expand(as: h)
      }
    }
    return h
  }
}

public class Dropout: Trainable {
  public let dropProb: Float

  public init(dropProb: Float) {
    self.dropProb = dropProb
  }

  public func callAsFunction(_ x: Tensor) -> Tensor {
    if mode == .training {
      (Tensor(randLike: x) >= dropProb).when(isTrue: (1.0 / (1.0 - dropProb)) * x, isFalse: 0)
    } else {
      x
    }
  }
}

public class LayerNorm: Trainable {
  public let shape: [Int]
  public let eps: Float

  @Param(name: "gain") public var gain: Tensor?
  @Param(name: "bias") public var bias: Tensor?

  public init(shape: [Int], dtype: Tensor.DType = .float32, eps: Float = 1e-5, affine: Bool = true)
  {
    self.shape = shape
    self.eps = eps
    super.init()
    if affine {
      self.gain = Tensor(zeros: shape, dtype: dtype)
      self.bias = Tensor(zeros: shape, dtype: dtype)
    }
  }

  public func callAsFunction(_ x: Tensor) -> Tensor {
    alwaysAssert(
      x.shape.count >= shape.count && Array(x.shape[(x.shape.count - shape.count)...]) == shape,
      "LayerNorm shape \(shape) is incompatible with input shape \(x.shape)")

    let batchShape = x.shape[..<(x.shape.count - shape.count)]
    let innerCount = shape.product()
    let tmpShape = [batchShape.product(), innerCount]
    let normedShape = Array(batchShape + Array(repeating: 1, count: shape.count))
    let mean = x.reshape(tmpShape).mean(axis: 1).reshape(normedShape)
    let variance = x.reshape(tmpShape).variance(axis: 1).reshape(normedShape)
    let normalized = (x - mean.expand(as: x)) * (variance.expand(as: x) + eps).rsqrt()
    if let gain = gain, let bias = bias {
      return normalized * (gain.expand(as: x) + 1) + bias.expand(as: x)
    } else {
      return normalized
    }
  }
}

public class GroupNorm: Trainable {
  public let groupCount: Int
  public let channelCount: Int
  public let channelsLast: Bool
  public let eps: Float

  @Param(name: "gian") var gain: Tensor?
  @Param(name: "bias") var bias: Tensor?

  public init(
    groupCount: Int, channelCount: Int, channelsLast: Bool = false, dtype: Tensor.DType = .float32,
    eps: Float = 1e-5, affine: Bool = true
  ) {
    alwaysAssert(
      channelCount % groupCount == 0,
      "channelCount \(channelCount) must be divisible by groupCount \(groupCount)")
    self.groupCount = groupCount
    self.channelCount = channelCount
    self.channelsLast = channelsLast
    self.eps = eps
    super.init()
    if affine {
      self.gain = Tensor(zeros: [channelCount], dtype: dtype)
      self.bias = Tensor(zeros: [channelCount], dtype: dtype)
    }
  }

  public func callAsFunction(_ x: Tensor) -> Tensor {
    if channelsLast {
      alwaysAssert(
        x.shape[x.shape.count - 1] == channelCount,
        "expected \(channelCount) channels but got shape \(x.shape)")
    } else {
      alwaysAssert(
        x.shape[1] == channelCount, "expected \(channelCount) channels but got shape \(x.shape)")
    }
    let cFirst =
      if channelsLast {
        x.move(axis: -1, to: 1)
      } else {
        x
      }

    let grouped = cFirst.reshape(
      [cFirst.shape[0], groupCount, cFirst.shape[1] / groupCount, cFirst.shape[2...].product()])
    let mean = grouped.mean(axis: -1, keepdims: true)
    let variance = grouped.variance(axis: -1, keepdims: true)
    let cFirstNormed =
      ((grouped - mean.expand(as: grouped)) * (variance + eps).rsqrt().expand(as: grouped))
      .reshape(as: cFirst)

    let normalized =
      if channelsLast {
        cFirstNormed.move(axis: 1, to: -1)
      } else {
        cFirstNormed
      }

    let result =
      if let gain = gain, let bias = bias {
        if channelsLast {
          normalized * (1 + gain.expand(as: normalized)) + bias.expand(as: normalized)
        } else {
          normalized * (1 + expandVector(gain, axis: 1, shape: x.shape))
            + expandVector(bias, axis: 1, shape: x.shape)
        }
      } else {
        normalized
      }
    return result
  }
}

private func expandVector(_ vec: Tensor, axis: Int, shape: [Int]) -> Tensor {
  var preShape = [Int](repeating: 1, count: shape.count)
  preShape[axis] = vec.shape[0]
  return vec.reshape(preShape).expand(shape: shape)
}
