# phonon-coreml

Swift package, command-line tool and Python example for running Phonon-2 on the Apple Neural Engine.

The model folder (one Core ML package, a decoder table and a manifest) is published with the model at
[FermionResearch/Phonon-2-CoreML](https://huggingface.co/FermionResearch/Phonon-2-CoreML).

## Quickstart

Requires a Mac with Apple silicon on macOS 15 or later, and Swift 6 from Xcode 16 or later or from the Command Line Tools
(`xcode-select --install`).

```bash
# 1. The package: clone the 1.1.2 release (or download it from the Releases page) and build it
git clone --branch 1.1.2 https://github.com/fermionresearch/phonon-coreml
cd phonon-coreml
swift build -c release

# 2. The model folder (330 MB), with the Hugging Face command line in a virtual environment
python3 -m venv .hf && .hf/bin/pip install -q huggingface_hub
.hf/bin/hf download FermionResearch/Phonon-2-CoreML --local-dir Phonon-2-CoreML

# 3. Transcribe a recording, with word timings
.build/release/phonon-coreml-cli Phonon-2-CoreML recording.m4a --words
```

The model folder includes three short clips to try first:

```bash
.build/release/phonon-coreml-cli Phonon-2-CoreML Phonon-2-CoreML/ci/clips/1089-134686-0002.flac --words
```

```text
After early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels.
0.40 0.56 After
0.72 1.12 early
1.12 1.68 nightfall
...
```

The first line is the text; with `--words` each word follows on its own line as start and end in seconds, then the word. `--json out.json`
writes the same as JSON. The tool reads wav, m4a, mp3, aiff, caf and flac at any sample rate; channels are averaged.

The first run on a Mac prepares the model for the Neural Engine, which takes a minute or two and happens once; the tool says so
when it starts. Later runs load in under a second.

## Swift package

```swift
// Package.swift
.package(url: "https://github.com/fermionresearch/phonon-coreml", from: "1.1.2")
```

```swift
import PhononCoreML

let transcriber = try Transcriber(bundle: URL(fileURLWithPath: bundlePath))
let result = try transcriber.transcribe(url: URL(fileURLWithPath: recordingPath))
print(result.text)
for word in result.words { print(word.start, word.end, word.text) }
```

Every word carries its start and end time in seconds. `transcribe(url:)` takes any audio file AVFoundation reads, at any sample rate;
`transcribe(_:)` takes mono 16 kHz samples. The library also builds for iOS 18.

## Python

Python 3.10 to 3.13 (coremltools has no wheels for Python 3.14 yet), in a virtual environment:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install coremltools numpy soundfile scipy huggingface_hub
hf download FermionResearch/Phonon-2-CoreML --local-dir Phonon-2-CoreML
python python/phonon_coreml.py Phonon-2-CoreML recording.wav
```

The Python runner reads wav and flac at any sample rate.

## How audio is read

An utterance of up to 35 seconds is read in one pass by the smallest encoder function that holds it. Longer audio is cut at pauses
into windows of up to 15 seconds and the words are joined, so no word is split at a boundary. The first run prepares each function
for the Neural Engine once and keeps the compiled copy under `~/Library/Caches/phonon-coreml/`; later loads take under a second. Setting
`Transcriber.Options.backgroundLoad` prepares the remaining functions in the background while the first one is already in use.

## Licence

Apache License 2.0 for everything in this repository. The model weights are CC-BY-4.0 and are not in this repository.
