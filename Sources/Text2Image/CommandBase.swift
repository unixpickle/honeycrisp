import Foundation
import Honeycrisp

class Command {

  public enum ArgumentError: Error {
    case invalidArgs
  }

  private var flopCounter: BackendFLOPCounter?
  private var startTime: DispatchTime?

  internal func startFLOPCounter() {
    startTime = DispatchTime.now()
    flopCounter = BackendFLOPCounter(wrapping: Backend.defaultBackend)
    Backend.defaultBackend = flopCounter!
  }

  internal var flopCount: Int64 {
    if flopCounter == nil || startTime == nil {
      startFLOPCounter()
    }
    return flopCounter!.flopCount
  }

  internal var gflops: Double {
    let fc = flopCount
    let nanos = (DispatchTime.now().uptimeNanoseconds - startTime!.uptimeNanoseconds)
    return Double(fc) / Double(nanos)
  }

  public func run() async throws {
    fatalError("must override run() method")
  }

}
