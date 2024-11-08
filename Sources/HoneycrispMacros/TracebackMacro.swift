import SwiftSyntax
import SwiftSyntaxMacros

public enum MacroError: Error {
  case message(String)
}

public struct TracebackMacro: PeerMacro {
  public static func expansion<
    Context: MacroExpansionContext,
    Declaration: DeclSyntaxProtocol
  >(
    of node: AttributeSyntax,
    providingPeersOf declaration: Declaration,
    in context: Context
  ) throws -> [DeclSyntax] {

    guard var newNode = declaration.as(FunctionDeclSyntax.self) else {
      throw MacroError.message("@addAsync only works on functions")
    }

    var newParams = Array(newNode.signature.parameterClause.parameters)

    if var oldLastParam = newParams.popLast() {
      oldLastParam.trailingComma = .commaToken()
      newParams.append(oldLastParam)
    }

    // Add `file` and `line` parameters
    newParams.append(
      FunctionParameterSyntax(
        firstName: TokenSyntax.identifier("tracebackFile"),
        type: TypeSyntax(IdentifierTypeSyntax(name: TokenSyntax.identifier("StaticString"))),
        defaultValue: InitializerClauseSyntax(value: ExprSyntax("#filePath")),
        trailingComma: TokenSyntax.commaToken())
    )
    newParams.append(
      FunctionParameterSyntax(
        firstName: TokenSyntax.identifier("tracebackLine"),
        type: TypeSyntax(IdentifierTypeSyntax(name: TokenSyntax.identifier("UInt"))),
        defaultValue: InitializerClauseSyntax(value: ExprSyntax("#line")))
    )

    var newSignature = newNode.signature
    var newParamClause = newNode.signature.parameterClause
    newParamClause.parameters = FunctionParameterListSyntax(newParams)
    newSignature.parameterClause = newParamClause
    newNode.signature = newSignature

    // Optionally remove leading _ from name.
    let rawName = "\(newNode.name)"
    let newName =
      if let underscoreIndex = rawName.firstIndex(of: "_") {
        String(rawName[rawName.index(after: underscoreIndex)...])
      } else {
        rawName
      }
    newNode.name = TokenSyntax(stringLiteral: newName)

    // Make method public if it is marked as private.
    var newMods = [DeclModifierSyntax]()
    var doesThrow = false
    for mod in newNode.modifiers {
      let name = "\(mod.name)"
      if name.contains("private") {
        newMods.append(
          DeclModifierSyntax(
            name: TokenSyntax(stringLiteral: "public"), trailingTrivia: mod.trailingTrivia))
      } else if name.contains("throws") {
        doesThrow = true
      } else {
        newMods.append(mod)
      }
    }
    newNode.modifiers = DeclModifierListSyntax(newMods)

    let stackCall = CodeBlockItemSyntax(
      """
      ThreadTraceback.current.push(tracebackFile, tracebackLine)
      defer { ThreadTraceback.current.pop() }
      """
    )

    var newBodyStatements = [CodeBlockItemSyntax]()
    newBodyStatements.append(stackCall)
    if doesThrow {
      newBodyStatements.append(
        CodeBlockItemSyntax("return try {\n\(newNode.body!.statements)\n}()"))
    } else {
      newBodyStatements.append(CodeBlockItemSyntax("return {\n\(newNode.body!.statements)\n}()"))
    }
    var newBody = newNode.body!
    newBody.statements = CodeBlockItemListSyntax(newBodyStatements)
    newNode.body = newBody

    // Don't accidentally use macro recursively.
    newNode.attributes = AttributeListSyntax([])

    // We must add newlines before the new function, but we want to
    // keep any docstrings.
    newNode.leadingTrivia = [.newlines(2)] + newNode.leadingTrivia

    return [DeclSyntax(newNode)]
  }
}
