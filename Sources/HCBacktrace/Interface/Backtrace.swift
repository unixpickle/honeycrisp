import Foundation

public struct CodeLocation: CustomStringConvertible {
  public let function: StaticString
  public let file: StaticString
  public let line: UInt

  public init(function: StaticString, file: StaticString, line: UInt) {
    self.function = function
    self.file = file
    self.line = line
  }

  public var description: String {
    "\(function) at \(file):\(line)"
  }
}

public struct TracedError: Error, CustomStringConvertible {
  public let wrapped: Error
  public let trace: [CodeLocation]

  public init(wrapped: Error, trace: [CodeLocation]) {
    self.wrapped = wrapped
    self.trace = trace
  }

  public var description: String {
    "Error at:\n\n\(formatCalls(trace))\n\nError: \(wrapped)"
  }
}

final public class Backtrace {
  @TaskLocal public static var current: [CodeLocation] = []

  internal static func wrapErrors<T>(_ fn: () throws -> T) rethrows -> T {
    do {
      return try fn()
    } catch {
      if (error as? TracedError) != nil {
        throw error
      } else {
        throw TracedError(wrapped: error, trace: current)
      }
    }
  }

  internal static func wrapErrors<T>(_ fn: () async throws -> T) async rethrows -> T {
    do {
      return try await fn()
    } catch {
      if (error as? TracedError) != nil {
        throw error
      } else {
        throw TracedError(wrapped: error, trace: current)
      }
    }
  }

  public static func record<T>(
    function: StaticString,
    file: StaticString,
    line: UInt,
    _ fn: () throws -> T,
    function1: StaticString = #function,
    file1: StaticString = #file,
    line1: UInt = #line
  ) rethrows -> T {
    try $current.withValue(
      current + [
        CodeLocation(function: function, file: file, line: line),
        CodeLocation(function: function1, file: file1, line: line1),
      ]
    ) {
      try wrapErrors {
        try fn()
      }
    }
  }

  public static func record<T>(
    function: StaticString,
    file: StaticString,
    line: UInt,
    _ fn: () async throws -> T,
    function1: StaticString = #function,
    file1: StaticString = #file,
    line1: UInt = #line
  ) async rethrows -> T {
    try await $current.withValue(
      current + [
        CodeLocation(function: function, file: file, line: line),
        CodeLocation(function: function1, file: file1, line: line1),
      ]
    ) {
      try await wrapErrors {
        try await fn()
      }
    }
  }

  public static func record<T>(
    _ fn: () throws -> T,
    function: StaticString = #function, file: StaticString = #file, line: UInt = #line
  ) rethrows -> T {
    try $current.withValue(current + [CodeLocation(function: function, file: file, line: line)]) {
      try wrapErrors {
        try fn()
      }
    }
  }

  public static func record<T>(
    _ fn: () async throws -> T, function: StaticString = #function, file: StaticString = #file,
    line: UInt = #line
  ) async rethrows -> T {
    try await $current.withValue(
      current + [CodeLocation(function: function, file: file, line: line)]
    ) {
      try await wrapErrors {
        try await fn()
      }
    }
  }

  public static func override<T>(_ callers: [CodeLocation], _ fn: () async throws -> T)
    async rethrows -> T
  {
    try await $current.withValue(callers) {
      try await wrapErrors {
        try await fn()
      }
    }
  }

  public static func trace() -> [CodeLocation] {
    current
  }

  public static func format(_ trace: [CodeLocation]? = nil) -> String {
    formatCalls(trace ?? current)
  }
}

private func formatCalls(_ locs: [CodeLocation]) -> String {
  locs.enumerated().map { x in
    let (i, loc) = (x.0, x.1)
    return String(repeating: "  ", count: i) + "\(loc)"
  }.joined(separator: "\n")
}

public func alwaysAssert(
  _ condition: Bool, _ message: String? = nil, function: StaticString = #function,
  file: StaticString = #file, line: UInt = #line
) {
  Backtrace.record(
    {
      if !condition {
        let msg =
          if let message = message {
            "\n\nTraceback:\n\n\(Backtrace.format())\n\nAssertion failure: \(message)"
          } else {
            "\n\nTraceback:\n\n\(Backtrace.format())\n\nAssertion failure"
          }
        fatalError(msg)
      }
    }, function: function, file: file, line: line)
}

public func tracedFatalError(
  _ message: String? = nil, function: StaticString = #function,
  file: StaticString = #file, line: UInt = #line
) -> Never {
  Backtrace.record(
    {
      let msg =
        if let message = message {
          "\n\nTraceback:\n\n\(Backtrace.format())\n\nFatal error: \(message)"
        } else {
          "\n\nTraceback:\n\n\(Backtrace.format())\n\n"
        }
      fatalError(msg)
    }, function: function, file: file, line: line)

  // This will not be reached.
  fatalError()
}