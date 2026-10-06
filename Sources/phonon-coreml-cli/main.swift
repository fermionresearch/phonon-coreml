// phonon-coreml-cli <Phonon-2-CoreML folder> <audio>... [--words] [--json out.json] [--verbose] [--warmup]
//                   benchmark options: [--passes N] [--workers N] [--dec-threads N] [--fast-threads N] [--enc-inflight N] [--reference]
//                   [--all-functions] [--eager 5,15] [--background-load] [--band-min S] [--rescue-mode halves|shift|wide|halves+shift]
//                   [--no-rescue] [--in-memory] [--cpu-gpu|--all|--cpu]
// Prints one transcript per file; --words adds one line per word (start and end in seconds, then the word); --json writes per-file wall
// time (model loaded once, best of N passes), the split, words and peak RSS. Audio: any file AVAudioFile reads, any sample rate.
import AVFoundation
import CoreML
import Foundation
import PhononCoreML

let usage = """
usage: phonon-coreml-cli <Phonon-2-CoreML folder> <audio file>... [--words] [--json out.json]
  Prints the text of each file. --words adds one line per word: start and end in seconds, then the word.
  Audio: wav, m4a, mp3, aiff, caf or flac, any sample rate; channels are averaged.
  The model folder: hf download FermionResearch/Phonon-2-CoreML --local-dir Phonon-2-CoreML

"""
func stderr(_ s: String) { FileHandle.standardError.write(s.data(using: .utf8)!) }
func fail(_ message: String, code: Int32 = 1) -> Never { stderr("error: \(message)\n"); exit(code) }
func readWav(_ url: URL) throws -> [Float] { try FileSource(url: url).readAll() }   // any rate -> 16 kHz mono, in memory

var args: [String] = []
for a in CommandLine.arguments.dropFirst() {   // @file = one path per line (paths may contain spaces)
    if a.hasPrefix("@"), let txt = try? String(contentsOfFile: String(a.dropFirst()), encoding: .utf8) { args.append(contentsOf: txt.split(separator: "\n").map(String.init).filter { !$0.isEmpty }) } else { args.append(a) }
}
if args.contains("--help") || args.contains("-h") { print(usage, terminator: ""); exit(0) }
var opt = Transcriber.Options(); var jsonOut: String? = nil; var passes = 1; var printWords = false; var allFunctions = false
let take: (String) -> String? = { flag in
    guard let i = args.firstIndex(of: flag) else { return nil }
    guard i + 1 < args.count else { stderr(usage); fail("\(flag) needs a value", code: 2) }
    let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
}
let has: (String) -> Bool = { flag in if let i = args.firstIndex(of: flag) { args.remove(at: i); return true }; return false }
func num<T: LosslessStringConvertible>(_ flag: String, _ v: String) -> T { guard let x = T(v) else { fail("\(flag): not a number: \(v)", code: 2) }; return x }
func nums(_ flag: String, _ v: String) -> [Double] { v.split(separator: ",").map { num(flag, String($0)) } }
jsonOut = take("--json"); if let v = take("--passes") { passes = num("--passes", v) }
if let v = take("--workers") { opt.decoderWorkers = num("--workers", v) }; if let v = take("--dec-threads") { opt.decoderThreads = num("--dec-threads", v) }
if let v = take("--fast-threads") { opt.fastDecoderThreads = num("--fast-threads", v) }; if let v = take("--enc-inflight") { opt.encoderInFlight = num("--enc-inflight", v) }
if let v = take("--band-min") { opt.bandMinSeconds = num("--band-min", v) }; if let v = take("--window-s") { opt.windowSeconds = num("--window-s", v) }; if let v = take("--rescue-mode") { opt.rescueMode = v }
if let v = take("--eager") { opt.eagerFunctions = nums("--eager", v) }; if let v = take("--functions") { opt.functions = nums("--functions", v) }
if has("--background-load") { opt.backgroundLoad = true }; if has("--parallel-load") { opt.parallelLoad = true }; let waitLoads = has("--wait-loads")
if has("--reference") { opt.longAudio = "reference" }; if has("--no-rescue") { opt.rescue = false }; if has("--windows15") { opt.longAudio = "windows15" }
if let v = take("--single-shot-max") { opt.singleShotMaxSeconds = num("--single-shot-max", v) }
if has("--cpu-gpu") { opt.computeUnits = .cpuAndGPU }; if has("--all") { opt.computeUnits = .all }; if has("--cpu") { opt.computeUnits = .cpuOnly }
printWords = has("--words"); allFunctions = has("--all-functions"); let inMemory = has("--in-memory")   // default: files are streamed window by window
let warmup = has("--warmup"); _ = has("--no-warmup")   // warm-up (two noise inputs before the first file) is for benchmarks only
let verbose = has("--verbose")
let melOut = take("--mel-out")
if let bad = args.first(where: { $0.hasPrefix("--") }) { stderr(usage); fail("unknown option \(bad)", code: 2) }

// First run on a Mac: the Neural Engine prepares each encoder function once (about a minute each); say so instead of sitting silent.
// A compile means a first run; a function load still going after 1.5 s means the Neural Engine is preparing it.
final class FirstRunNotice: @unchecked Sendable {
    let lock = NSLock(); var shown = false; var pending: DispatchWorkItem? = nil
    func show() { lock.lock(); defer { lock.unlock() }; pending?.cancel(); pending = nil; if !shown { shown = true; stderr("Preparing the Neural Engine for this Mac, a minute or two; this happens once.\n") } }
    func loading() { lock.lock(); defer { lock.unlock() }; if shown { return }; let w = DispatchWorkItem { [weak self] in self?.show() }; pending?.cancel(); pending = w; DispatchQueue.global().asyncAfter(deadline: .now() + 1.5, execute: w) }
    func loaded() { lock.lock(); defer { lock.unlock() }; pending?.cancel(); pending = nil }
}
let notice = FirstRunNotice()
var progressLog: [[String: Any]] = []; let progressLock = NSLock()
opt.progress = { phase, t in
    if phase == "compiling" { notice.show() } else if phase.hasPrefix("loading ") { notice.loading() } else if phase.hasPrefix("loaded ") { notice.loaded() }
    progressLock.lock(); progressLog.append(["phase": phase, "t_s": t]); progressLock.unlock()
    if verbose { stderr("PROGRESS \(String(format: "%.2f", t)) s \(phase)\n") }
}
if let outPath = melOut {   // parity mode: write [128*T] Float32 features of one file, no model
    guard let wav = args.first else { fail("--mel-out needs an audio file", code: 2) }
    do {
        let w = try readWav(URL(fileURLWithPath: wav)); let (f, T) = LogMel().features(w)
        var d = Data(); f.withUnsafeBufferPointer { d.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress!).assumingMemoryBound(to: UInt8.self), count: $0.count * 4)) }
        try d.write(to: URL(fileURLWithPath: outPath)); print("T", T); exit(0)
    } catch { fail(error.localizedDescription) }
}
guard args.count >= 2 else { stderr(usage); exit(2) }

// Check the inputs before the model loads, so a typo fails in a second rather than after the first-run preparation.
let bundleURL = URL(fileURLWithPath: args[0])
guard FileManager.default.fileExists(atPath: bundleURL.appendingPathComponent("manifest.json").path) else {
    fail("\(args[0]) is not a Phonon-2-CoreML model folder (it has no manifest.json). Download the folder with:\n  hf download FermionResearch/Phonon-2-CoreML --local-dir Phonon-2-CoreML")
}
var seconds: [Double] = []
for path in args.dropFirst() {
    do { let f = try FileSource.open(URL(fileURLWithPath: path)); seconds.append(Double(f.length) / f.processingFormat.sampleRate) }
    catch { fail(error.localizedDescription) }
}
// Load only what the files need: a short file needs the one function that holds it; long audio runs 15 s windows (the library's
// default eager function). The benchmark warm-up keeps the library default.
if opt.eagerFunctions == nil && !warmup && !allFunctions, let man = try? JSONDecoder().decode(Manifest.self, from: Data(contentsOf: bundleURL.appendingPathComponent("manifest.json"))) {
    let fs = (opt.functions ?? man.functions_s).sorted()
    let single = opt.singleShotMaxSeconds ?? min(man.single_shot_max_s ?? .infinity, fs.last ?? .infinity)
    if let first = seconds.first, seconds.allSatisfy({ $0 <= single }), opt.longAudio != "reference", let f = fs.first(where: { $0 >= first }) { opt.eagerFunctions = [f] }
}

@MainActor func run() throws {
var ruL = rusage(); getrusage(RUSAGE_SELF, &ruL)
let t0 = Date()
let tr = try Transcriber(bundle: bundleURL, options: opt)
if allFunctions { try tr.preloadAllFunctions() }
var firstTextS: Double? = nil
if let v = ProcessInfo.processInfo.environment["FIRST_TEXT_WAV"] {   // first-run UX: time from launch to the first transcript (a short hold), while other functions still compile
    _ = try tr.transcribe(readWav(URL(fileURLWithPath: v))); firstTextS = Date().timeIntervalSince(t0)
}
if waitLoads { tr.waitForBackgroundLoads() }
let allLoadedS = Date().timeIntervalSince(t0)
let loadS = Date().timeIntervalSince(t0)
var loadedAfterInit = tr.loadedFunctions
if warmup {
    _ = try? tr.transcribe((0..<16000).map { _ in Float.random(in: -0.01...0.01) })   // warm the function a 1 s input uses
    do { _ = try? tr.transcribe((0..<(16000 * 14)).map { _ in Float.random(in: -0.01...0.01) }) }   // and the 15 s function for long audio
}
tr.resetTiming()
if verbose { stderr("READY load \(String(format: "%.2f", loadS)) s (compile \(String(format: "%.2f", tr.compileSeconds)) s)\n") }
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
    if printWords { for w in t.words { print(String(format: "%.2f %.2f ", w.start, w.end) + w.text) } }
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
}
do { try run() } catch { fail(error.localizedDescription) }
