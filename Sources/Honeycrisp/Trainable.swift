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

  public init() {
  }

  public var parameters: [(String, Parameter)] {
    var results = Array(registeredParams)
    for (name, child) in registeredChildren {
      results += child.parameters.map { (subName, param) in
        ("\(name).\(subName)", param)
      }
    }
    return results.sorted(by: { $0.0 < $1.0 })
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
      (Tensor(rand: [inCount, outCount], dtype: dtype) - 0.5) * (sqrt(3.0) / 0.5 / Float(inCount))
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
