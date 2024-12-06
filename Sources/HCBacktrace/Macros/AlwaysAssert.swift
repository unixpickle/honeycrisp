import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public enum AlwaysAssert: ExpressionMacro {
  public static func expansion(
    of node: some FreestandingMacroExpansionSyntax,
    in context: some MacroExpansionContext
  ) throws -> ExprSyntax {
    if ![1, 2].contains(node.arguments.count) {
      throw MacroError.message(
        "alwaysAssert expects 1 or 2 arguments, but got \(node.arguments.count)")
    }
    let condition = node.arguments.first!.expression
    let message = node.arguments.count == 2 ? node.arguments.last!.expression : ExprSyntax("nil")
    if condition.description.trimmingCharacters(in: .whitespacesAndNewlines) == "false" {
      return "alwaysAssert(false, \(message))"
    }
    return "(!(\(condition)) ? alwaysAssert(false, \(message)) : ())"
  }
}
