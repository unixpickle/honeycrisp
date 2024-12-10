import Foundation

/// Make a new copy of the method which records the caller in the current ``Backtrace``.
///
/// In general, a Swift function can only record its caller by adding extra arguments with
/// defaults like `#filePath`, `#line`, and `#function`. It would be quite tedious to manually add
/// these arguments to every method of a class.
///
/// Instead, we can use a macro to make this easier. For example, we might start with this class:
///
/// ```swift
/// class Foo {
///   public func printNumbers(x: Int) {
///     for i in 0..<x {
///       print(i)
///     }
///   }
/// }
/// ```
///
/// If we'd like to automatically record callers to our `printNumbers` method, we can wrap it
/// with a `@recordCaller`. We will do this by renaming `printNumbers` to `_printNumbers`,
/// because the `@recordCaller` macro cannot _replace_ our method, it can only create a new one
/// (so it will do so by removing the underscore). We will also make this method `private`, since
/// we do not want to expose the *original* (untracked) version of the method.
///
/// ```swift
/// class Foo {
///   @recordCaller
///   private func _printNumbers(x: Int) {
///     for i in 0..<x {
///       print(i)
///     }
///   }
/// }
/// ```
///
/// This will expand to the equivalent code:
///
/// ```swift
/// class Foo {
///   private func _printNumbers(x: Int) {
///     for i in 0..<x {
///       print(i)
///     }
///   }
///
///   public func printNumbers(x: Int, function: StaticString = #function, file: StaticString = #filePath, line: UInt = #line) {
///     Backtrace.record(function: function, file: file, line: line) {
///       for i in 0..<x {
///         print(i)
///       }
///     }
///   }
/// }
/// ```
@attached(peer, names: arbitrary)
public macro recordCaller() = #externalMacro(module: "HCBacktraceMacros", type: "RecordCaller")

/// Assert a condition with a backtrace on failure.
@freestanding(expression)
public macro alwaysAssert(_ value: Bool, _ message: String? = nil) =
  #externalMacro(module: "HCBacktraceMacros", type: "AlwaysAssert")
