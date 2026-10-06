import AVFoundation
import XCTest
@testable import PhononCoreML

/// Files at other sample rates, channel counts and formats come out as 16 kHz mono with the same timing; bad paths throw clear errors.
final class FileSourceTests: XCTestCase {
    var dir: URL!
    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("phonon-coreml-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    /// `seconds` of a 440 Hz tone at amplitude 0.25 with a click (+0.9 on every channel) at `clickS`; channel c is scaled by 1 / (c + 1)
    /// for the tone so the averaged channels differ from channel 0.
    func writeTone(_ name: String, rate: Double, channels: Int, seconds: Double, clickS: Double, settings extra: [String: Any]? = nil) throws -> URL {
        let url = dir.appendingPathComponent(name)
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: AVAudioChannelCount(channels), interleaved: false)!
        let n = Int(rate * seconds)
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n))!; buf.frameLength = AVAudioFrameCount(n)
        let click = Int(rate * clickS)
        for c in 0..<channels {
            let p = buf.floatChannelData![c]
            for i in 0..<n { p[i] = 0.25 / Float(c + 1) * Float(sin(2 * Double.pi * 440 * Double(i) / rate)) }
            p[click] = 0.9
        }
        var settings = fmt.settings
        if let extra { settings = extra }
        let f = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try f.write(from: buf)
        return url
    }

    func check(_ url: URL, seconds: Double, clickS: Double, tolerance: Int, file: StaticString = #filePath, line: UInt = #line) throws {
        let s = try FileSource(url: url)
        XCTAssertEqual(Double(s.count), seconds * 16000, accuracy: Double(tolerance), "16 kHz length", file: file, line: line)
        let x = try s.readAll()
        let peak = x.indices.max(by: { abs(x[$0]) < abs(x[$1]) })!
        XCTAssertEqual(Double(peak), clickS * 16000, accuracy: Double(tolerance), "click lands at the same time", file: file, line: line)
        let rms = (x[8000..<24000].map { $0 * $0 }.reduce(0, +) / 16000).squareRoot()
        XCTAssertGreaterThan(rms, 0.05, "the tone survives the conversion", file: file, line: line)
        // a window read from the middle equals the same samples of the whole read (windows are read by seek)
        let mid = try s.read(20000..<28000)
        XCTAssertEqual(mid, Array(x[20000..<28000]), file: file, line: line)
    }

    func testWav16kIsReadInPlace() throws {
        let url = try writeTone("a16.wav", rate: 16000, channels: 1, seconds: 3, clickS: 1.25, settings: [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false])
        let s = try FileSource(url: url)
        XCTAssertNil(s.temporary); XCTAssertEqual(s.count, 48000)
        try check(url, seconds: 3, clickS: 1.25, tolerance: 0)
    }
    func testWav44kStereo() throws {
        let url = try writeTone("a44.wav", rate: 44100, channels: 2, seconds: 3, clickS: 1.25, settings: [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 44100, AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false])
        try check(url, seconds: 3, clickS: 1.25, tolerance: 2)
    }
    func testWav48kMono() throws {
        let url = try writeTone("a48.wav", rate: 48000, channels: 1, seconds: 4, clickS: 2.5)
        try check(url, seconds: 4, clickS: 2.5, tolerance: 2)
    }
    func testM4a44k() throws {
        let url = try writeTone("a44.m4a", rate: 44100, channels: 1, seconds: 3, clickS: 1.5, settings: [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44100, AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 128000])
        try check(url, seconds: 3, clickS: 1.5, tolerance: 40)   // AAC frames: the length and click hold to a few milliseconds
    }
    func testChannelsAreAveraged() throws {
        let url = try writeTone("st48.wav", rate: 48000, channels: 2, seconds: 2, clickS: 1.0)
        let x = try FileSource(url: url).readAll()
        let rms = (x[4000..<12000].map { $0 * $0 }.reduce(0, +) / 8000).squareRoot()
        XCTAssertEqual(Double(rms), 0.75 * 0.25 / 2.0.squareRoot(), accuracy: 0.01)   // (1 + 1/2) / 2 of the tone
    }
    func testTemporaryFileIsRemoved() throws {
        let url = try writeTone("t48.wav", rate: 48000, channels: 1, seconds: 1, clickS: 0.5)
        var s: FileSource? = try FileSource(url: url)
        let tmp = try XCTUnwrap(s?.temporary)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmp.path))
        s = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: tmp.path))
    }
    func testMissingFile() {
        let url = dir.appendingPathComponent("nope.wav")
        XCTAssertThrowsError(try FileSource(url: url)) { e in
            XCTAssertEqual(e.localizedDescription, "\(url.path): no such file")
        }
        XCTAssertThrowsError(try FileSource(url: dir)) { e in XCTAssertTrue(e.localizedDescription.hasSuffix("no such file")) }
    }
    func testNotAudio() throws {
        let url = dir.appendingPathComponent("notes.wav")
        try Data("not audio".utf8).write(to: url)
        XCTAssertThrowsError(try FileSource(url: url)) { e in
            XCTAssertTrue(e.localizedDescription.hasPrefix("\(url.path): not an audio file"), e.localizedDescription)
        }
    }
}
