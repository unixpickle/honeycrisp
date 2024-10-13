import Cocoa
import Honeycrisp

enum ImageError: Error {
  case createCGContext
  case createCGImage
  case createData
  case encodePNG
}

func loadImage(path: String, imageSize: Int) -> Tensor? {
  guard let loadedImage = NSImage(byReferencingFile: path) else {
    return nil
  }

  let bitsPerComponent = 8
  let bytesPerRow = imageSize * 4
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let bitmapInfo: CGImageAlphaInfo = .premultipliedLast

  guard
    let context = CGContext(
      data: nil,
      width: imageSize,
      height: imageSize,
      bitsPerComponent: bitsPerComponent,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo.rawValue
    )
  else {
    return nil
  }
  context.clear(CGRect(origin: .zero, size: CGSize(width: imageSize, height: imageSize)))

  let size = loadedImage.size
  let scale = CGFloat(imageSize) / min(CGFloat(size.width), CGFloat(size.height))
  let scaledX = (CGFloat(imageSize) - scale * CGFloat(size.width)) / 2
  let scaledY = (CGFloat(imageSize) - scale * CGFloat(size.height)) / 2
  let imageRect = CGRect(
    origin: CGPoint(x: scaledX, y: scaledY), size: CGSize(width: imageSize, height: imageSize))
  guard let loadedCGImage = loadedImage.cgImage(forProposedRect: nil, context: nil, hints: [:])
  else {
    return nil
  }
  context.draw(loadedCGImage, in: imageRect)

  guard let data = context.data else {
    return nil
  }

  let buffer = data.bindMemory(to: UInt8.self, capacity: imageSize * bytesPerRow)
  var floats = [Float]()
  for i in 0..<(imageSize * imageSize * 4) {
    floats.append(Float(buffer[i]) / 255.0)
  }
  return Tensor(data: floats, shape: [imageSize, imageSize, 4])
}

func tensorToImage(tensor: Tensor) async throws -> Data {
  assert(tensor.shape.count == 3)
  assert(tensor.shape[2] == 4, "tensor must be RGBA")
  let height = tensor.shape[0]
  let width = tensor.shape[1]

  let floats = try await tensor.floats()

  let bytesPerRow = width * 4
  var buffer = [UInt8](repeating: 0, count: height * bytesPerRow)
  for (i, f) in floats.enumerated() {
    buffer[i] = UInt8(floor(min(1, max(0, f)) * 255.999))
  }

  return try buffer.withUnsafeMutableBytes { ptr in
    var ptr: UnsafeMutablePointer<UInt8>? = ptr.bindMemory(to: UInt8.self).baseAddress!
    let rep = NSBitmapImageRep(
      bitmapDataPlanes: &ptr, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
      bytesPerRow: width * 4, bitsPerPixel: 32)!
    if let result = rep.representation(using: .png, properties: [:]) {
      return result
    } else {
      throw ImageError.encodePNG
    }
  }
}
