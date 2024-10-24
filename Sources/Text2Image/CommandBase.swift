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

  internal var gflops: Double {
    if flopCounter == nil || startTime == nil {
      startFLOPCounter()
    }
    let nanos = (DispatchTime.now().uptimeNanoseconds - startTime!.uptimeNanoseconds)
    let flops = flopCounter!.flopCount
    return Double(flops) / Double(nanos)
  }

  public func run() async throws {
    fatalError("must override run() method")
  }

}
