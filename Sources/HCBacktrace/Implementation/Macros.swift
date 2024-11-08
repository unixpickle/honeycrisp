import Foundation

@attached(peer, names: arbitrary)
public macro recordCaller() = #externalMacro(module: "HCBacktraceMacros", type: "RecordCaller")
