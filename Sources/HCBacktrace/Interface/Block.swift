public struct TracedBlock<T>: Sendable {

  let fn: @Sendable () async throws -> T

  public init(_ fn: @escaping @Sendable () async throws -> T) {
    self.fn = fn
  }

  @recordCaller
  private func _callAsFunction() async throws -> T {
    try await fn()
  }

}
