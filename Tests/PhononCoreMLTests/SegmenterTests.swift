import XCTest
@testable import PhononCoreML
final class SegmenterTests: XCTestCase {
    func testShortIsOneWindow() { XCTAssertEqual(Segmenter.plan([Float](repeating: 0.1, count: 16000 * 30)).count, 1) }
    func testLongIsCutInBand() {
        var w = [Float](repeating: 0, count: 16000 * 70)
        for i in 0..<w.count {
            let phase = Double(i) * 0.01
            let speaking: Bool = (i % (16000 * 31)) < 16000 * 30
            w[i] = speaking ? Float(sin(phase)) : 0
        }
        let cuts: [(Int, Int)] = Segmenter.plan(w)
        XCTAssertEqual(cuts.count, 3)
        for pair in cuts.dropLast() { let len: Int = pair.1 - pair.0; XCTAssertGreaterThanOrEqual(len, 400000); XCTAssertLessThanOrEqual(len, 560000) }
    }
    func testFramesFor() { XCTAssertEqual(Transcriber.framesFor(seconds: 15), 1501); XCTAssertEqual(Transcriber.subLen(1501), 188) }
}
