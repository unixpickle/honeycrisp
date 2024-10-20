import Cocoa
import Foundation
import Honeycrisp

@main
struct Main {
  static func main() async {
    if CommandLine.arguments.count < 2 {
      printHelp()
      return
    }
    do {
      switch CommandLine.arguments[1] {
      case "vqvae":
        let cmd = try VQVAETrainer(Array(CommandLine.arguments[2...]))
        try await cmd.run()
      case "transformer":
        let cmd = try TransformerTrainer(Array(CommandLine.arguments[2...]))
        try await cmd.run()
      default:
        print("Unrecognized subcommand: \(CommandLine.arguments[1])")
        printHelp()
      }
    } catch {
      print("ERROR: \(error)")
      return
    }
  }

  static func printHelp() {
    print("Usage: Text2Image <subcommand> ...")
    print("Subcommands:")
    print("    vqvae <image_dir> <state_path>")
    print("    transformer <image_dir> <vqvae_path> <state_path>")
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
