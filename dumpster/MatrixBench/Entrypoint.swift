import Foundation
import Honeycrisp

@main
struct Entrypoint {
  static func main() {
    for size in [32, 64, 128, 1024, 4096, 16384, 32768] {
      let memory = 8 * pow(Double(size), 2) / 1_000_000_000
      print("trying matmul of size \(size) (\(memory) GiB)")
      let m1 = Tensor(zeros: [size, size])
      let m2 = Tensor(zeros: [size, size])
      let t1 = DispatchTime.now()
      let _ = (m1 &* m2).sum().item()
      let t2 = DispatchTime.now()
      let x = t2.uptimeNanoseconds - t1.uptimeNanoseconds
      let duration = Double(x) / 1_000_000_000
      let flops = 2 * pow(Double(size), 3)
      print(" => \(Int(round(flops / duration / 1_000_000_000))) GFlops")
    }
  }
}
