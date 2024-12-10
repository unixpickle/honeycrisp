import Foundation

/// Iterate through a synchronous iterator in a background thread and push
/// the results to an asynchronous stream.
///
/// Buffers `bufferSize` elements in memory.
public func loadDataInBackground<T, S: Sequence<Result<T, Error>>>(_ it: S, bufferSize: Int = 2)
  -> AsyncThrowingStream<T, Error>
where T: Sendable, S: Sendable {
  AsyncThrowingStream(bufferingPolicy: .bufferingOldest(bufferSize)) { continuation in
    let thread = Thread {
      var it = it.makeIterator()
      while true {
        var maybeX: Result<T, Error>?
        autoreleasepool {
          maybeX = it.next()
        }
        guard let x = maybeX else { break }
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

    let t = SendableThread(thread: thread)
    continuation.onTermination = { _ in t.thread.cancel() }
  }
}

/// Like loadDataInBackground, but allows using non-sendable sequences.
public func loadDataInBackgroundSending<T, S: Sequence<Result<T, Error>>>(
  _ it: sending S, bufferSize: Int = 2
)
  -> AsyncThrowingStream<T, Error>
where T: Sendable {
  let sendIt = SendS(it)
  return AsyncThrowingStream(bufferingPolicy: .bufferingOldest(bufferSize)) { continuation in
    let thread = Thread {
      var it = sendIt.s.makeIterator()
      while true {
        var maybeX: Result<T, Error>?
        autoreleasepool {
          maybeX = it.next()
        }
        guard let x = maybeX else { break }
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

    let t = SendableThread(thread: thread)
    continuation.onTermination = { _ in t.thread.cancel() }
  }
}

private struct SendableThread: @unchecked Sendable {
  let thread: Thread
}

struct SendS<S>: @unchecked Sendable {
  let s: S
  init(_ s: S) {
    self.s = s
  }
}
