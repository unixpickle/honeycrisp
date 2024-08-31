open class Trainable {

  @propertyWrapper
  public final class Parameter {
    public static subscript<T: Trainable>(
      _enclosingInstance instance: T,
      wrapped wrappedKeyPath: ReferenceWritableKeyPath<T, Tensor>,
      storage storageKeyPath: ReferenceWritableKeyPath<T, Parameter>
    ) -> Tensor {
      get {
        let param = instance[keyPath: storageKeyPath]
        let rawTensor = instance[keyPath: storageKeyPath].data!
        return rawTensor.onGrad { g in
          if let existingGrad = param.grad {
            param.grad = existingGrad + g
          } else {
            param.grad = g
          }
        }
      }
      set {
        let param = instance[keyPath: storageKeyPath]
        if param.name == nil {
          param.name = String("\(storageKeyPath)".split(separator: ".").last!)
        }
        param.data = newValue
        instance.registeredParams[param.name!] = param
      }
    }

    @available(
      *, unavailable,
      message: "@Parameter can only be applied to classes"
    )
    public var wrappedValue: Tensor {
      get { fatalError() }
      set { fatalError() }
    }

    var name: String? = nil
    public var data: Tensor?
    public var grad: Tensor?

    public var projectedValue: Parameter { self }

    public init(name: String? = nil) {
      self.name = name
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
  public let children: [Trainable]

  public init(_ children: [T]) {
    self.children = children
    super.init()
    for (i, ch) in children.enumerated() {
      self.registeredChildren[String(i)] = ch
    }
  }
}
