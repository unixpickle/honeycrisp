extension Sequence where Element: Numeric {
  func product() -> Element {
    reduce(Element(exactly: 1)!, *)
  }

  func sum() -> Element {
    reduce(Element(exactly: 0)!, +)
  }
}
