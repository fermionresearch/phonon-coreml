import XCTest
@testable import PhononCoreML

final class WindowerTests: XCTestCase {
    func testShortAudioIsOneWindow() {
        let a = [Float](repeating: 0.1, count: 16000 * 30)
        XCTAssertEqual(Windower.plan(a).count, 1)
    }
    func testPausesGiveNonOverlappingWindows() {
        // 60 s: speech-like noise with 0.4 s silences every 12 s -> cuts land in the silences, no overlap
        var a = [Float](repeating: 0, count: 16000 * 60); var g = SystemRandomNumberGenerator()
        for i in 0..<a.count { let t = Double(i) / 16000; let inPause = (t.truncatingRemainder(dividingBy: 12.0)) > 11.6; a[i] = inPause ? 0 : Float.random(in: -0.3...0.3, using: &g) }
        let w = Windower.plan(a)
        XCTAssertTrue(w.count >= 4 && w.count <= 6, "\(w.count)")
        XCTAssertFalse(w.dropFirst().contains { $0.overlapFromPrevious })
        for (x, y) in zip(w, w.dropFirst()) { XCTAssertEqual(x.end, y.start) }
        XCTAssertEqual(w.last!.end, a.count)
        for x in w { XCTAssertLessThanOrEqual(x.end - x.start, 16000 * 15) }
    }
    func testNoPauseGivesOverlap() {
        var a = [Float](repeating: 0, count: 16000 * 40); var g = SystemRandomNumberGenerator()
        for i in 0..<a.count { a[i] = Float.random(in: -0.3...0.3, using: &g) }
        let w = Windower.plan(a)
        XCTAssertEqual(w.count, 3)
        XCTAssertTrue(w[1].overlapFromPrevious); XCTAssertNotNil(w[1].splitSample)
        XCTAssertEqual(w[0].end - w[1].start, 32000)
        XCTAssertTrue(w[1].splitSample! > w[1].start && w[1].splitSample! < w[0].end)
    }
    func testWordsFromTokensPipRule() {
        let toks = [TimedToken(piece: " trans", start: 0.0, duration: 0.08), TimedToken(piece: "cri", start: 0.16, duration: 0.08), TimedToken(piece: "ption", start: 0.24, duration: 0.08),
                    TimedToken(piece: " ,", start: 0.4, duration: 0.0), TimedToken(piece: " ok", start: 0.56, duration: 0.08)]
        let w = Words.fromTokens(toks, offset: 10.0, limit: 1.0)
        XCTAssertEqual(w.map { $0.text }, ["transcription ,", "ok"])
        XCTAssertEqual(w[0].start, 10.0); XCTAssertEqual(w[0].end, 10.4); XCTAssertEqual(w[1].end, 10.64)
    }
    func testStitchKeepsEachWordOnce() {
        var kept = [Word(text: "one", start: 0.0, end: 0.3), Word(text: "two", start: 13.0, end: 13.4), Word(text: "three", start: 13.9, end: 14.3)]
        let incoming = [Word(text: "two", start: 13.05, end: 13.4), Word(text: "three", start: 13.9, end: 14.3), Word(text: "four", start: 14.5, end: 14.9)]
        Words.stitch(kept: &kept, incoming: incoming, split: 13.8)
        XCTAssertEqual(kept.map { $0.text }, ["one", "two", "three", "four"])
    }
}
