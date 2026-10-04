// The README's snippet, compiled and run by CI: phonon-coreml-example <bundle-folder> <recording.wav>
import Foundation
import PhononCoreML

let args = CommandLine.arguments
let bundlePath = args[1], recordingPath = args[2]

// --- README: transcribe a file
let transcriber = try Transcriber(bundle: URL(fileURLWithPath: bundlePath))
let result = try transcriber.transcribe(url: URL(fileURLWithPath: recordingPath))
print(result.text)
for word in result.words { print(word.start, word.end, word.text) }
