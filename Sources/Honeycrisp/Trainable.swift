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

  public init(inCount: Int, outCount: Int, dtype: Tensor.DType = .float32, bias: Bool = true) {
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
    var h = x &* weight
    if let bias = bias {
      h = h + bias.expand(as: h)
    }
    return h
  }
}

public class Conv2D: Trainable {
  public typealias HWCSize = Conv2DConfig.HWCSize
  public typealias HWSize = Conv2DConfig.HWSize
  public typealias AxisPadding = Conv2DConfig.AxisPadding

  public enum Padding {
    case none
    case same
    case allSides(Int)
    case xy(x: Int, y: Int)
    case leftRightTopBottom(Int, Int, Int, Int)
  }

  public enum SpatialSize {
    case square(Int)
    case widthHeight(Int, Int)

    public var hwSize: HWSize {
      switch self {
      case .square(let x):
        HWSize(h: x, w: x)
      case .widthHeight(let x, let y):
        HWSize(h: y, w: x)
      }
    }
  }

  public let inChannels: Int
  public let outChannels: Int
  public let kernelSize: Conv2DConfig.HWSize
  public let stride: Conv2DConfig.HWSize
  public let dilation: Conv2DConfig.HWSize
  public let paddingH: Conv2DConfig.AxisPadding
  public let paddingW: Conv2DConfig.AxisPadding
  public let groups: Int
  public let channelsLast: Bool

  @Param(name: "weight") public var weight: Tensor
  @Param(name: "bias") public var bias: Tensor?

  public init(
    inChannels: Int, outChannels: Int, kernelSize: SpatialSize, stride: SpatialSize = .square(1),
    padding: Padding = .none, dilation: SpatialSize = .square(1), groups: Int = 1,
    channelsLast: Bool = false, bias: Bool = true, dtype: Tensor.DType = .float32
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize.hwSize
    self.stride = stride.hwSize
    self.dilation = dilation.hwSize
    alwaysAssert(
      inChannels % groups == 0, "outChannels \(outChannels) not divisible by groups \(groups)")
    alwaysAssert(
      outChannels % groups == 0, "inChannels \(inChannels) not divisible by groups \(groups)")
    switch padding {
    case .none:
      self.paddingH = AxisPadding(before: 0, after: 0)
      self.paddingW = AxisPadding(before: 0, after: 0)
    case .same:
      alwaysAssert(
        self.stride == HWSize(h: 1, w: 1),
        "cannot use padding mode 'same' with stride \(self.stride)")
      let kernelW = self.kernelSize.w + (self.kernelSize.w - 1) * self.dilation.w
      let kernelH = self.kernelSize.h + (self.kernelSize.h - 1) * self.dilation.h
      self.paddingH = Conv2DConfig.AxisPadding(before: (kernelH - 1) / 2, after: kernelH / 2)
      self.paddingW = Conv2DConfig.AxisPadding(before: (kernelW - 1) / 2, after: kernelW / 2)
    case .allSides(let x):
      self.paddingH = AxisPadding(before: x, after: x)
      self.paddingW = AxisPadding(before: x, after: x)
    case .xy(let x, let y):
      self.paddingH = AxisPadding(before: y, after: y)
      self.paddingW = AxisPadding(before: x, after: x)
    case .leftRightTopBottom(let left, let right, let top, let bottom):
      self.paddingH = AxisPadding(before: top, after: bottom)
      self.paddingW = AxisPadding(before: left, after: right)
    }
    self.groups = groups
    self.channelsLast = channelsLast
    super.init()
    self.weight =
      (Tensor(
        rand: [outChannels, inChannels / groups, self.kernelSize.h, self.kernelSize.w],
        dtype: dtype) - 0.5)
      * (sqrt(3.0) / 0.5 / sqrt(Float(inChannels * self.kernelSize.h * self.kernelSize.w)))
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
    let convDesc = Conv2DConfig(
      kernelSize: Conv2DConfig.HWCSize(h: kernelSize.h, w: kernelSize.w, c: outChannels),
      imageSize: Conv2DConfig.HWCSize(h: height, w: width, c: channels),
      stride: stride, dilation: dilation, paddingH: paddingH, paddingW: paddingW, groups: groups,
      channelsLast: channelsLast)
    var h = Tensor.conv2d(convDesc, image: x, kernel: weight)
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
      (1.0 / (1.0 - dropProb)) * x * (Tensor(randLike: x) > dropProb).cast(.float32)
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
