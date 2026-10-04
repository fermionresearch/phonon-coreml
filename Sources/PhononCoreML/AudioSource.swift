// Audio sources for the transcriber: whole arrays, or files read window by window so an hour of audio never sits in memory at once.
import AVFoundation
import Foundation

public protocol AudioSource {
    var count: Int { get }                                   // samples at 16 kHz mono
    func read(_ range: Range<Int>) throws -> [Float]         // samples of the range (mono, 16 kHz)
}

public struct ArraySource: AudioSource {
    public let samples: [Float]
    public init(_ s: [Float]) { samples = s }
    public var count: Int { samples.count }
    public func read(_ r: Range<Int>) throws -> [Float] { Array(samples[r]) }
}

/// 16 kHz wav/flac/caf… via AVAudioFile; channels are averaged. Reads are seeks + short reads, 10 s at a time.
public final class FileSource: AudioSource {
    let file: AVAudioFile; let channels: Int; public let count: Int
    let lock = NSLock()
    public init(url: URL) throws {
        file = try AVAudioFile(forReading: url)
        guard file.processingFormat.sampleRate == 16000 else { throw NSError(domain: "PhononCoreML", code: 5, userInfo: [NSLocalizedDescriptionKey: "\(url.lastPathComponent): need 16 kHz audio"]) }
        channels = Int(file.processingFormat.channelCount); count = Int(file.length)
    }
    public func read(_ r: Range<Int>) throws -> [Float] {
        lock.lock(); defer { lock.unlock() }
        var out = [Float](repeating: 0, count: r.count); var pos = r.lowerBound
        let chunk = 16000 * 10
        let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(min(chunk, max(1, r.count))))!
        file.framePosition = AVAudioFramePosition(r.lowerBound)
        while pos < r.upperBound {
            try file.read(into: buf, frameCount: AVAudioFrameCount(min(chunk, r.upperBound - pos)))
            let m = Int(buf.frameLength); if m == 0 { break }
            let o = pos - r.lowerBound
            if let fd = buf.floatChannelData { for c in 0..<channels { let p = fd[c]; for i in 0..<m { out[o + i] += p[i] / Float(channels) } } }
            else if let id = buf.int16ChannelData { for c in 0..<channels { let p = id[c]; for i in 0..<m { out[o + i] += Float(p[i]) / 32768.0 / Float(channels) } } }
            pos += m
        }
        return out
    }
}

extension AudioSource {
    /// RMS of 50 ms blocks over the whole source, streamed (10 s at a time); the windower's only whole-file input.
    public func blockRMS(block: Int = 800) throws -> [Float] {
        let nBlocks = (count + block - 1) / block; var rms = [Float](repeating: 0, count: nBlocks)
        let step = 16000 * 10; var start = 0
        while start < count {
            let end = min(count, start + step); let x = try read(start..<end)
            var b = start / block
            while b * block < end {
                let lo = b * block, hi = min(end, lo + block); var s: Float = 0
                for i in lo..<hi { s += x[i - start] * x[i - start] }
                rms[b] += s / Float(block); b += 1
            }
            start = end
        }
        for i in 0..<nBlocks { rms[i] = rms[i].squareRoot() }
        return rms
    }
}
