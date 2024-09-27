import Cocoa
import Honeycrisp

func packPixelsIntoTokens(pixels: Tensor, patchSize: Int) -> Tensor {
  var patches: Tensor?
  for i in 0..<patchSize {
    for j in 0..<patchSize {
      let subPatch = pixels[
        ..., stride(from: i, to: 28, by: patchSize), stride(from: j, to: 28, by: patchSize)]
      let scale = 1 << (i * patchSize + j)
      if let p = patches {
        patches = p + subPatch * scale
      } else {
        patches = subPatch * scale
      }
    }
  }
  return patches!
}

func unpackPixelsInTokens(bitmaps: Tensor, patchSize: Int) async throws -> Tensor {
  func sampleToImage(bitmap: Tensor) async throws -> Tensor {
    let data = try await bitmap.ints()
    var result = Array(repeating: 0, count: 28 * 28)
    for i in 0..<patchSize {
      for j in 0..<patchSize {
        let mask = 1 << (i * patchSize + j)
        for y in stride(from: i, to: 28, by: patchSize) {
          for x in stride(from: j, to: 28, by: patchSize) {
            if data[(y / patchSize) * (28 / patchSize) + x / patchSize] & mask != 0 {
              result[y * 28 + x] = 1
            }
          }
        }
      }
    }
    return Tensor(data: result, shape: [28 * 28], dtype: .int64)
  }
  var results: [Tensor] = []
  for i in 0..<bitmaps.shape[0] {
    results.append(try await sampleToImage(bitmap: bitmaps[i]))
  }
  return Tensor(concat: results).reshape([bitmaps.shape[0], 28 * 28])
}

enum saveImageError: Error {
  case createCGImage
  case createPNG
}

func saveGrayscaleBitmap(pixels pixelTensor: Tensor, to url: URL)
  async throws
{
  let pixels: [UInt8] = (try await pixelTensor.ints()).map({ UInt8($0) })
  let height = pixelTensor.shape[0]
  let width = pixelTensor.shape[1]
  let bitsPerComponent = 8
  let bitsPerPixel = 8
  let bytesPerRow = width
  let colorSpace = CGColorSpaceCreateDeviceGray()
  let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

  guard let dataProvider = CGDataProvider(data: Data(pixels) as CFData),
    let cgImage = CGImage(
      width: width,
      height: height,
      bitsPerComponent: bitsPerComponent,
      bitsPerPixel: bitsPerPixel,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: dataProvider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent
    )
  else {
    throw saveImageError.createCGImage
  }

  let image = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))

  guard let tiffData = image.tiffRepresentation,
    let bitmapRep = NSBitmapImageRep(data: tiffData),
    let pngData = bitmapRep.representation(using: .png, properties: [:])
  else {
    throw saveImageError.createPNG
  }

  try pngData.write(to: url)
}
