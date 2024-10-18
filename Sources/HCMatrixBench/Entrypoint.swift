import Foundation
import Honeycrisp

@main
struct Main {
  static func main() async {
    do {
      // Backend.defaultBackend = CoreMLBackend(wrapping: try MPSBackend())
      Backend.defaultBackend = try MPSBackend()

      for innerSize in [512, 1024, 2048] {
        for outerSize in [512, 1024, 2048, 4096, 16384, 32768] {
          let m1 = Tensor(rand: [outerSize, innerSize], dtype: .float16)
          let m2 = Tensor(rand: [innerSize, innerSize], dtype: .float16)
          func runMatmul() async throws {
            let result = m1 &* m2
            let data = try await result.data
            if let c = data.completeOnAllDevices {
              let _ = try await c.value
            }
          }

          try await runMatmul()
          let t1 = DispatchTime.now()
          try await runMatmul()
          let t2 = DispatchTime.now()
          let flops = innerSize * innerSize * outerSize * 2
          let gflops = Float(flops) / Float(t2.uptimeNanoseconds - t1.uptimeNanoseconds)
          print("size \(outerSize) x \(innerSize) x \(innerSize): \(gflops) GFLOP/s")
        }
      }
    } catch {
      print("fatal error: \(error)")
    }
  }
}
