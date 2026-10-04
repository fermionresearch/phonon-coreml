import XCTest
@testable import PhononCoreML

final class DecoderV2Tests: XCTestCase {
    /// Needs PHONON_TEST_DECODER_V1 / _V2 (decoder.bin in both formats) and PHONON_TEST_ENCREF (encoder outputs, [T,640] f32); skipped otherwise.
    /// decoder.bin v2 (6-bit codes packed 4 per 3 bytes) decodes token for token like v1, and clones share one table set.
    func testPackedDecoderMatchesV1() throws {
        let v1 = URL(fileURLWithPath: ProcessInfo.processInfo.environment["PHONON_TEST_DECODER_V1"] ?? "/nonexistent")
        let v2 = URL(fileURLWithPath: ProcessInfo.processInfo.environment["PHONON_TEST_DECODER_V2"] ?? "/nonexistent")
        guard FileManager.default.fileExists(atPath: v1.path), FileManager.default.fileExists(atPath: v2.path) else { throw XCTSkip("decoder files not present") }
        let a = try TDTDecoder(url: v1), b = try TDTDecoder(url: v2)
        let tb = try DecoderTables(url: v2); let c1 = try TDTDecoder(tables: tb, threads: 1), c4 = try TDTDecoder(tables: tb, threads: 4)
        for trial in 0..<6 {
            let f = URL(fileURLWithPath: (ProcessInfo.processInfo.environment["PHONON_TEST_ENCREF"] ?? "/nonexistent") + "/\(trial).f32")
            guard let d = try? Data(contentsOf: f) else { throw XCTSkip("encoder reference dumps not present") }
            let enc = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }; let T = enc.count / 640
            let ra = a.decodeTimed(enc, frames: T), rb = b.decodeTimed(enc, frames: T), r1 = c1.decodeTimed(enc, frames: T), r4 = c4.decodeTimed(enc, frames: T)
            XCTAssertEqual(ra.tokens, rb.tokens); XCTAssertEqual(ra.frame, rb.frame); XCTAssertEqual(ra.duration, rb.duration)
            XCTAssertEqual(rb.tokens, r1.tokens); XCTAssertEqual(rb.tokens, r4.tokens); XCTAssertEqual(rb.frame, r4.frame)
            XCTAssertGreaterThan(ra.tokens.count, 0)
        }
    }
}
