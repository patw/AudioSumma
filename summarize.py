import os
import requests
import datetime
import tempfile
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load settings from environment
WHISPERCPP_URL = os.getenv("WHISPERCPP_URL")
LLAMACPP_URL = os.getenv("LLAMACPP_URL")
SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE")
SUMMARY_PROMPT = os.getenv("SUMMARY_PROMPT")
FACT_PROMPT = os.getenv("FACT_PROMPT")
SENTIMENT_PROMPT = os.getenv("SENTIMENT_PROMPT")
PROMPT_FORMAT = os.getenv("PROMPT_FORMAT")
STOP_TOKEN = os.getenv("STOP_TOKEN")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
TEMPERATURE = float(os.getenv("TEMPERATURE"))

def whisper_api(file):
    """Transcribe audio file using Whisper API."""
    files = {"file": file}
    api_data = {
        "temperature": "0.0",
        "response_format": "json"
    }
    response = requests.post(WHISPERCPP_URL, data=api_data, files=files)
    return response.json()["text"]

def llama_api(prompt):
    """Generate response using llama.cpp server API."""
    formatted_prompt = PROMPT_FORMAT.format(system=SYSTEM_MESSAGE, prompt=prompt)
    api_data = {
        "prompt": formatted_prompt,
        "n_predict": -1,
        "temperature": TEMPERATURE,
        "stop": [STOP_TOKEN],
        "tokens_cached": 0
    }
    response = requests.post(LLAMACPP_URL, headers={"Content-Type": "application/json"}, json=api_data)
    json_output = response.json()
    return json_output['content']

def trim_silence(filename):
    """Trim silence from audio file using FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name

    ffmpeg_command = [
        "ffmpeg",
        "-i", filename,
        "-af", "silenceremove=stop_threshold=-40dB:stop_duration=1:stop_periods=-1",
        "-y",  # Overwrite output file if it exists
        temp_filename
    ]

    result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
    os.replace(temp_filename, filename)

def process_wav_files():
    """Process WAV files: trim silence and transcribe."""
    wav_files = [f for f in os.listdir(".") if f.endswith(".wav")]
    for wav_file in wav_files:
        print("Trimming silence: " + wav_file)
        trim_silence(wav_file)

        with open(wav_file, "rb") as file:
            print("Transcribing: " + wav_file)
            output_text = whisper_api(file)
            output_file = os.path.splitext(wav_file)[0] + ".tns"
            with open(output_file, "w") as output:
                output.write(output_text)

def chunk_transcript(string, chunk_size):
    """Chunk the transcript to fit in the context window."""
    chunks = []
    lines = string.split("\n")
    current_chunk = ""
    for line in lines:
        current_chunk += line
        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def summarize_transcripts():
    """Summarize transcript files."""
    today = datetime.datetime.now().strftime('%Y%m%d')
    summary_filename = "summary-" + today + ".md"
    transcript_files = [f for f in os.listdir(".") if f.endswith(".tns")]

    for transcript in transcript_files:
        print("Summarizing: " + transcript)
        with open(transcript, "r") as file:
            transcript_data = file.read()
            chunked_data = chunk_transcript(transcript_data, CHUNK_SIZE)

            with open(summary_filename, "a") as md_file:
                for i, chunk in enumerate(chunked_data):
                    summary = llama_api(SUMMARY_PROMPT.format(chunk=chunk))
                    facts = llama_api(FACT_PROMPT.format(chunk=chunk))
                    sentiment = llama_api(SENTIMENT_PROMPT.format(chunk=chunk))

                    md_file.write(f"# Call Transcript - {transcript} - Part {i + 1}\n\nSummary: {summary}\n\nFacts:\n{facts}\n\nSentiment: {sentiment}\n\n---\n")

    print("Summarizing complete")

if __name__ == "__main__":
    process_wav_files()
    summarize_transcripts()