# AudioSumma

Record your local audio and summarize it with whisper.cpp and llama.cpp!  Open source, local on-prem transcription and summarization!

![Main UI](screenshot.png)

## Installation

### macOS Requirements
For macOS users, you'll need to have [Homebrew](https://brew.sh/) installed first. Then install the required dependencies:

```bash
brew install ffmpeg portaudio
```

### All Platforms
```
pip install -r requirements.txt
```

## Configuration

Use the Settings button in the app to configure the endpoints for your local Whisper.cpp and OLLAMA/llama.cpp servers.

## Running

Run either meetings.bat or meetings.sh to start app.


## Usage

Hit record to record your global audio, hit stop to save the wav file.  Hit transcribe to transcribe all wav files collected into a single summary markdown document (with date stamp).  Hit the Clean button to remove old wav files and transcripts.

