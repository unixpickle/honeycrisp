import Cocoa
import Honeycrisp

enum ImageError: Error {
  case createCGContext
  case createCGImage
  case createData
  case encodeTIFF
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
  assert(tensor.shape[2] == 4, "tensor must be RGBA")
  let height = tensor.shape[0]
  let width = tensor.shape[1]

  let floats = try await tensor.floats()

  let bitsPerComponent = 8
  let bytesPerRow = width * 4
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let bitmapInfo: CGImageAlphaInfo = .premultipliedLast

  guard
    let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: bitsPerComponent,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo.rawValue
    )
  else {
    throw ImageError.createCGContext
  }
  guard let data = context.data else {
    throw ImageError.createData
  }
  let buffer = data.bindMemory(to: UInt8.self, capacity: height * bytesPerRow)
  for (i, f) in floats.enumerated() {
    buffer[i] = UInt8(floor(f * 255.999))
  }

  guard let cgImage = context.makeImage() else {
    throw ImageError.createCGImage
  }
  guard
    let result = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
      .tiffRepresentation
  else {
    throw ImageError.encodeTIFF
  }
  return result
}
