import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct MyProjectMacros: CompilerPlugin {
  var providingMacros: [Macro.Type] = [AlwaysAssert.self, RecordCaller.self]
}
