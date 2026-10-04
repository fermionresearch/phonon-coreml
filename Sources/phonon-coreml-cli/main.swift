// phonon-coreml-cli <bundle-dir> <wav> [<wav> ...] [--json out.json] [--passes N] [--workers N] [--dec-threads N] [--fast-threads N]
//                   [--enc-inflight N] [--reference] [--all-functions] [--eager 5,15] [--background-load] [--band-min S]
//                   [--rescue-mode halves|shift|wide|halves+shift] [--no-rescue] [--words] [--cpu-gpu|--all|--cpu]
// Prints one transcript per line; --json writes per-file wall time (model loaded once, best of N passes), the split, words and peak RSS.
import AVFoundation
import CoreML
import Foundation
import PhononCoreML

func readWav(_ url: URL) throws -> [Float] {
    let f = try AVAudioFile(forReading: url)
    guard f.processingFormat.sampleRate == 16000 else { throw NSError(domain: "cli", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(url.lastPathComponent): need 16 kHz audio"]) }
    let n = Int(f.length), ch = Int(f.processingFormat.channelCount), chunk = 16000 * 10
    var out = [Float](repeating: 0, count: n); var pos = 0
    let buf = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(chunk))!
    while pos < n {
        try f.read(into: buf, frameCount: AVAudioFrameCount(min(chunk, n - pos)))
        let m = Int(buf.frameLength); if m == 0 { break }
        if let fd = buf.floatChannelData { for c in 0..<ch { let p = fd[c]; for i in 0..<m { out[pos + i] += p[i] / Float(ch) } } }
        else if let id = buf.int16ChannelData { for c in 0..<ch { let p = id[c]; for i in 0..<m { out[pos + i] += Float(p[i]) / 32768.0 / Float(ch) } } }
        pos += m
    }
    return out
}
var args: [String] = []
for a in CommandLine.arguments.dropFirst() {   // @file = one path per line (paths may contain spaces)
    if a.hasPrefix("@"), let txt = try? String(contentsOfFile: String(a.dropFirst()), encoding: .utf8) { args.append(contentsOf: txt.split(separator: "\n").map(String.init).filter { !$0.isEmpty }) } else { args.append(a) }
}
var opt = Transcriber.Options(); var jsonOut: String? = nil; var passes = 1; var printWords = false; var allFunctions = false
let take: (String) -> String? = { flag in if let i = args.firstIndex(of: flag) { let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v }; return nil }
let has: (String) -> Bool = { flag in if let i = args.firstIndex(of: flag) { args.remove(at: i); return true }; return false }
jsonOut = take("--json"); if let v = take("--passes") { passes = Int(v)! }
if let v = take("--workers") { opt.decoderWorkers = Int(v)! }; if let v = take("--dec-threads") { opt.decoderThreads = Int32(v)! }
if let v = take("--fast-threads") { opt.fastDecoderThreads = Int32(v)! }; if let v = take("--enc-inflight") { opt.encoderInFlight = Int(v)! }
if let v = take("--band-min") { opt.bandMinSeconds = Double(v)! }; if let v = take("--window-s") { opt.windowSeconds = Double(v)! }; if let v = take("--rescue-mode") { opt.rescueMode = v }
if let v = take("--eager") { opt.eagerFunctions = v.split(separator: ",").map { Double($0)! } }; if let v = take("--functions") { opt.functions = v.split(separator: ",").map { Double($0)! } }
if has("--background-load") { opt.backgroundLoad = true }; if has("--parallel-load") { opt.parallelLoad = true }; let waitLoads = has("--wait-loads")
if has("--reference") { opt.longAudio = "reference" }; if has("--no-rescue") { opt.rescue = false }; if has("--windows15") { opt.longAudio = "windows15" }
if let v = take("--single-shot-max") { opt.singleShotMaxSeconds = Double(v)! }
if has("--cpu-gpu") { opt.computeUnits = .cpuAndGPU }; if has("--all") { opt.computeUnits = .all }; if has("--cpu") { opt.computeUnits = .cpuOnly }
printWords = has("--words"); allFunctions = has("--all-functions"); let inMemory = has("--in-memory")   // default: files are streamed window by window
let noWarmup = has("--no-warmup")
var progressLog: [[String: Any]] = []
opt.progress = { phase, t in progressLog.append(["phase": phase, "t_s": t]); FileHandle.standardError.write("PROGRESS \(String(format: "%.2f", t)) s \(phase)\n".data(using: .utf8)!) }
if let i = args.firstIndex(of: "--mel-out") {   // parity mode: write [128*T] Float32 features of one wav, no model
    let outPath = args[i + 1], wav = args[i + 2]
    let w = try readWav(URL(fileURLWithPath: wav)); let (f, T) = LogMel().features(w)
    var d = Data(); f.withUnsafeBufferPointer { d.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self), count: $0.count * 4)) }
    try d.write(to: URL(fileURLWithPath: outPath)); print("T", T); exit(0)
}
guard args.count >= 2 else { FileHandle.standardError.write("usage: phonon-coreml-cli <bundle> <wav>... [--json out] [--passes N] [--workers N] [--words]\n".data(using: .utf8)!); exit(2) }
var ruL = rusage(); getrusage(RUSAGE_SELF, &ruL)
let t0 = Date()
let tr = try Transcriber(bundle: URL(fileURLWithPath: args[0]), options: opt)
if allFunctions { try tr.preloadAllFunctions() }
var firstTextS: Double? = nil
if let v = ProcessInfo.processInfo.environment["FIRST_TEXT_WAV"] {   // first-run UX: time from launch to the first transcript (a short hold), while other functions still compile
    _ = try tr.transcribe(readWav(URL(fileURLWithPath: v))); firstTextS = Date().timeIntervalSince(t0)
}
if waitLoads { tr.waitForBackgroundLoads() }
let allLoadedS = Date().timeIntervalSince(t0)
let loadS = Date().timeIntervalSince(t0)
var loadedAfterInit = tr.loadedFunctions
if !noWarmup {
    _ = try? tr.transcribe((0..<16000).map { _ in Float.random(in: -0.01...0.01) })   // warm the function a 1 s input uses
    do { _ = try? tr.transcribe((0..<(16000 * 14)).map { _ in Float.random(in: -0.01...0.01) }) }   // and the 15 s function for long audio
}
tr.resetTiming()
FileHandle.standardError.write("READY load \(String(format: "%.2f", loadS)) s (compile \(String(format: "%.2f", tr.compileSeconds)) s)\n".data(using: .utf8)!)
var ru0 = rusage(); getrusage(RUSAGE_SELF, &ru0)
func cpuSeconds(_ r: rusage) -> Double { Double(r.ru_utime.tv_sec + r.ru_stime.tv_sec) + Double(r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6 }
let tRunStart = Date()
var best: [String: Double] = [:]; var first: [String: Double] = [:]; var outT: [String: Transcript] = [:]
for _ in 0..<passes {
    for path in args.dropFirst() {
        try autoreleasepool {
            let t1 = Date(); let t = inMemory ? try tr.transcribe(readWav(URL(fileURLWithPath: path))) : try tr.transcribe(url: URL(fileURLWithPath: path)); let dt = Date().timeIntervalSince(t1)
            if first[path] == nil { first[path] = dt }; best[path] = min(best[path] ?? 1e9, dt); outT[path] = t
        }
    }
}
var ta = 0.0, tw = 0.0; var rows: [[String: Any]] = []
for path in args.dropFirst() {
    let t = outT[path]!; ta += t.audioSeconds; tw += best[path]!; outT[path] = nil
    var row: [String: Any] = ["path": (path as NSString).lastPathComponent, "audio_s": t.audioSeconds, "wall_s_best": best[path]!, "wall_s_first": first[path]!, "words": t.words.count, "windows": t.windows, "text": t.text]
    if printWords { row["word_list"] = t.words.map { ["text": $0.text, "start": $0.start, "end": $0.end] }; row["segments"] = t.segments.map { ["id": $0.id, "start": $0.start, "end": $0.end, "text": $0.text] } }
    rows.append(row); print(t.text)
}
var ru1 = rusage(); getrusage(RUSAGE_SELF, &ru1)
let cpuS = cpuSeconds(ru1) - cpuSeconds(ru0); let runWall = Date().timeIntervalSince(tRunStart)
if let j = jsonOut {
    let t = tr.timing
    let res: [String: Any] = ["runtime": "phonon-coreml-cli (Core ML encoder functions, C TDT loop, pipelined), model loaded once, best of \(passes) passes", "compute_units": "\(opt.computeUnits.rawValue)",
                              "decoder_workers": opt.decoderWorkers, "decoder_threads": opt.decoderThreads, "fast_decoder_threads": opt.fastDecoderThreads, "encoder_in_flight": opt.encoderInFlight,
                              "rescue_mode": opt.rescue ? opt.rescueMode : "off", "band_min_s": opt.bandMinSeconds,
                              "long_audio": opt.longAudio ?? tr.manifest.long_audio ?? "windows15", "load_s": loadS, "compile_s": tr.compileSeconds, "function_load_s": Dictionary(uniqueKeysWithValues: tr.functionLoadSeconds.map { ("\($0.key)", $0.value) }),
                              "functions_loaded_after_init": loadedAfterInit, "first_text_s": firstTextS ?? -1, "all_loaded_s": allLoadedS, "functions_loaded_end": tr.loadedFunctions, "progress": progressLog, "compiled_url": tr.compiledURL.path,
                              "audio_s": ta, "decode_wall_s": tw, "x_realtime": ta / tw, "ms_per_audio_s": 1000 * tw / ta, "passes": passes, "run_wall_s_all_passes": runWall, "cpu_s_all_passes": cpuS, "peak_rss_gb": Double(ru1.ru_maxrss) / 1e9,
                              "split_last_pass": ["mel_s": t.melS, "enc_s": t.encS, "dec_s_summed_over_workers": t.decS, "wall_s": t.wallS, "audio_s": t.audioS, "windows": t.windows, "overlapped_boundaries": t.overlapped, "rescue_tried": t.rescueTried, "rescued": t.rescued, "rescue_words": t.rescueWords], "rows": rows]
    try JSONSerialization.data(withJSONObject: res, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: j))
    FileHandle.standardError.write("ROW audio \(ta)s wall \(tw)s = \(ta / tw)x realtime, peak RSS \(Double(ru1.ru_maxrss) / 1e9) GB\n".data(using: .utf8)!)
}
_ = ruL; loadedAfterInit = []
