# AudioSumma

Record your local audio and summarize it with whisper.cpp and llama.cpp!  Open source, local on-prem transcription and summarization!

## AudioSumma Installation

```
pip install -r requirements.txt
```

## Configuration

Copy sample.env to .env and point your endpoint URLs for a working llama.cpp and whisper.cpp running in server/api mode.

## Running

Run either meetings.bat or meetings.sh to start app.


## Usage

Hit record to record your global audio, hit stop to save the wav file.  Hit transcribe to transcribe all wav files collected into a single summary markdown document (with date stamp).  Hit the Clean button to remove old wav files and transcripts.

