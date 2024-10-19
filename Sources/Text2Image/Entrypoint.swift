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
    if CommandLine.arguments[1] != "vqvae" {
      print("Unrecognized subcommand: \(CommandLine.arguments[1])")
      printHelp()
      return
    }
    do {
      let cmd = try VQVAETrainer(Array(CommandLine.arguments[2...]))
      try await cmd.run()
    } catch {
      print("ERROR: \(error)")
      return
    }
  }

  static func printHelp() {
    print("Usage: Text2Image <subcommand> ...")
    print("Subcommands:")
    print("    vqvae <image_dir> <state_path>")
  }
}

func formatFloat(_ x: Float) -> String {
  String(format: "%.5f", x)
}
