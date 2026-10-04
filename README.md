# phonon-coreml

Swift package, command-line tool and Python example for running Phonon-2 on the Apple Neural Engine.

The model folder (one Core ML package, a decoder table and a manifest) is published with the model at
[FermionResearch/Phonon-2-CoreML](https://huggingface.co/FermionResearch/Phonon-2-CoreML).

## Install

```swift
// Package.swift
.package(url: "https://github.com/fermionresearch/phonon-coreml", from: "1.1.0")
```

Requires macOS 15 or iOS 18 and Apple silicon.

## Transcribe a file

```bash
swift build -c release
.build/release/phonon-coreml-cli <Phonon-2-CoreML folder> recording.wav --words
```

```swift
import PhononCoreML

let transcriber = try Transcriber(bundle: URL(fileURLWithPath: bundlePath))
let result = try transcriber.transcribe(url: URL(fileURLWithPath: recordingPath))
print(result.text)
for word in result.words { print(word.start, word.end, word.text) }
```

Every word carries its start and end time in seconds. Audio is mono 16 kHz.

## Python

```bash
pip install coremltools numpy soundfile
python python/phonon_coreml.py <Phonon-2-CoreML folder> recording.wav
```

## How audio is read

An utterance of up to 35 seconds is read in one pass by the smallest encoder function that holds it. Longer audio is cut at pauses
into windows of up to 15 seconds and the words are joined, so no word is split at a boundary. The first run prepares each function
for the Neural Engine once and keeps the compiled copy under `~/Library/Caches/phonon-coreml/`; later loads take under a second. Setting
`Transcriber.Options.backgroundLoad` prepares the remaining functions in the background while the first one is already in use.

## Licence

Apache License 2.0 for everything in this repository. The model weights are CC-BY-4.0 and are not in this repository.
