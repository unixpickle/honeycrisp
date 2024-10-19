import Foundation

public func loadDataInBackground<T, S: Sequence<T>>(_ it: S, bufferSize: Int = 2) -> AsyncStream<T>
where T: Sendable, S: Sendable {
  AsyncStream(bufferingPolicy: .bufferingOldest(bufferSize)) { continuation in
    let thread = Thread {
      for x in it {
        if Thread.current.isCancelled {
          return
        }
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
      continuation.finish()
    }
    thread.name = "loadDataInBackground-Worker"
    thread.start()
    continuation.onTermination = { [cancel = thread.cancel] _ in cancel() }
  }
}
