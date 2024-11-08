import SwiftSyntax
import SwiftSyntaxMacros

public enum MacroError: Error {
  case message(String)
}

public struct RecordCaller: PeerMacro {
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

    let oldParams = Array(newNode.signature.parameterClause.parameters)

    let isAsync = newNode.signature.effectSpecifiers?.asyncSpecifier != nil
    let isThrows = newNode.signature.effectSpecifiers?.throwsClause?.throwsSpecifier != nil
    func maybeTryAndAwait(_ call: FunctionCallExprSyntax) -> ExprSyntax {
      var callExpr = ExprSyntax(call)
      if isAsync {
        callExpr = ExprSyntax(AwaitExprSyntax(expression: callExpr))
      }
      if isThrows {
        callExpr = ExprSyntax(TryExprSyntax(expression: callExpr))
      }
      return callExpr
    }

    let callInnerFunction = maybeTryAndAwait(
      FunctionCallExprSyntax(
        calledExpression: ExprSyntax("\(newNode.name)"),
        leftParen: TokenSyntax.leftParenToken(),
        arguments: LabeledExprListSyntax(
          oldParams.map { param in
            let noName = param.firstName.tokenKind == .wildcard
            let isInOut =
              if case .attributedType(let t) = param.type.as(TypeSyntaxEnum.self) {
                t.specifiers.contains(where: { spec in
                  if let x = spec.as(SimpleTypeSpecifierSyntax.self)?.specifier {
                    x.tokenKind == .keyword(SwiftSyntax.Keyword.inout)
                  } else {
                    false
                  }
                })
              } else {
                false
              }
            let arg = DeclReferenceExprSyntax(baseName: param.secondName ?? param.firstName)
            return LabeledExprSyntax(
              label: noName ? nil : param.firstName, colon: noName ? nil : param.colon,
              expression: isInOut ? ExprSyntax(InOutExprSyntax(expression: arg)) : ExprSyntax(arg),
              trailingComma: param.trailingComma)
          }),
        rightParen: TokenSyntax.rightParenToken()
      ))

    var newParams = oldParams

    if var oldLastParam = newParams.popLast() {
      oldLastParam.trailingComma = .commaToken()
      newParams.append(oldLastParam)
    }

    // Add parameters that capture caller position.
    let argFunc = context.makeUniqueName("function")
    let argFile = context.makeUniqueName("file")
    let argLine = context.makeUniqueName("line")
    newParams.append(
      FunctionParameterSyntax(
        firstName: argFunc,
        type: TypeSyntax(IdentifierTypeSyntax(name: TokenSyntax.identifier("StaticString"))),
        defaultValue: InitializerClauseSyntax(value: ExprSyntax("#function")),
        trailingComma: TokenSyntax.commaToken())
    )
    newParams.append(
      FunctionParameterSyntax(
        firstName: argFile,
        type: TypeSyntax(IdentifierTypeSyntax(name: TokenSyntax.identifier("StaticString"))),
        defaultValue: InitializerClauseSyntax(value: ExprSyntax("#filePath")),
        trailingComma: TokenSyntax.commaToken())
    )
    newParams.append(
      FunctionParameterSyntax(
        firstName: argLine,
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
    for mod in newNode.modifiers {
      let name = "\(mod.name)"
      if name.contains("private") {
        newMods.append(
          DeclModifierSyntax(
            name: TokenSyntax(stringLiteral: "public"), trailingTrivia: mod.trailingTrivia))
      } else {
        newMods.append(mod)
      }
    }
    newNode.modifiers = DeclModifierListSyntax(newMods)

    // Call the function inside `Backtrace.record`.
    let callExpr: ExprSyntax = maybeTryAndAwait(
      FunctionCallExprSyntax(
        calledExpression: ExprSyntax("Backtrace.record"),
        leftParen: TokenSyntax.leftParenToken(),
        arguments: LabeledExprListSyntax([
          LabeledExprSyntax(
            expression: ClosureExprSyntax(
              statements: CodeBlockItemListSyntax([
                CodeBlockItemSyntax(item: .expr(callInnerFunction))
              ])),
            trailingComma: TokenSyntax.commaToken()),
          LabeledExprSyntax(
            label: TokenSyntax.identifier("function"), colon: TokenSyntax.colonToken(),
            expression: DeclReferenceExprSyntax(baseName: argFunc),
            trailingComma: TokenSyntax.commaToken()),
          LabeledExprSyntax(
            label: TokenSyntax.identifier("file"), colon: TokenSyntax.colonToken(),
            expression: DeclReferenceExprSyntax(baseName: argFile),
            trailingComma: TokenSyntax.commaToken()),
          LabeledExprSyntax(
            label: TokenSyntax.identifier("line"), colon: TokenSyntax.colonToken(),
            expression: DeclReferenceExprSyntax(baseName: argLine)),
        ]),
        rightParen: TokenSyntax.rightParenToken()
      ))

    let codeBlock = CodeBlockItemSyntax(item: .expr(callExpr))

    var newBody = newNode.body!
    newBody.statements = CodeBlockItemListSyntax([codeBlock])
    newNode.body = newBody

    // Don't accidentally use macro recursively.
    newNode.attributes = AttributeListSyntax([])

    // We must add newlines before the new function, but we want to
    // keep any docstrings.
    newNode.leadingTrivia = [.newlines(2)] + newNode.leadingTrivia

    return [DeclSyntax(newNode)]
  }
}
