import Foundation

public func loadDataInBackground<T, S: Sequence<Result<T, Error>>>(_ it: S, bufferSize: Int = 2)
  -> AsyncThrowingStream<T, Error>
where T: Sendable, S: Sendable {
  AsyncThrowingStream(bufferingPolicy: .bufferingOldest(bufferSize)) { continuation in
    let thread = Thread {
      for x in it {
        if Thread.current.isCancelled {
          return
        }
        switch x {
        case .failure(let e):
          continuation.finish(throwing: e)
          return
        case .success(let x):
          var sent = false
          while !sent {
            switch continuation.yield(x) {
            case .dropped(_):
              Thread.sleep(forTimeInterval: 0.05)
            default:
              sent = true
            }
          }
        }
      }
      continuation.finish()
    }
    thread.name = "loadDataInBackground-Worker"
    thread.start()
    continuation.onTermination = { [cancel = thread.cancel] _ in cancel() }
  }
}
