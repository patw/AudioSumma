import os
import requests
import datetime
import tempfile
import subprocess
from dotenv import load_dotenv

# Use local models with the OpenAI library and a custom baseurl
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Load settings from environment
WHISPERCPP_URL = os.getenv("WHISPERCPP_URL")
LLAMACPP_URL = os.getenv("LLAMACPP_URL")
SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE")
SUMMARY_PROMPT = os.getenv("SUMMARY_PROMPT")
FACT_PROMPT = os.getenv("FACT_PROMPT")
SENTIMENT_PROMPT = os.getenv("SENTIMENT_PROMPT")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
TEMPERATURE = float(os.getenv("TEMPERATURE"))
TOP_P = float(os.getenv("TOP_P"))
MAX_TOKENS = float(os.getenv("MAX_TOKENS"))

def whisper_api(file):
    """Transcribe audio file using Whisper API."""
    files = {"file": file}
    api_data = {
        "temperature": "0.0",
        "response_format": "json"
    }
    response = requests.post(WHISPERCPP_URL, data=api_data, files=files)
    return response.json()["text"]

def llm_local(prompt):
    client = OpenAI(api_key="doesntmatter", base_url=LLAMACPP_URL)
    messages=[{"role": "system", "content": SYSTEM_MESSAGE},{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="whatever", max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, messages=messages)
    return response.choices[0].message.content

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
        # Generate the expected transcript filename
        transcript_file = os.path.splitext(wav_file)[0] + ".tns"

        # Check if transcript already exists
        if os.path.exists(transcript_file):
            print(f"Transcript already exists for {wav_file}, skipping transcription")
            continue

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
                    print("Processing part " + str(i))
                    summary = llm_local(SUMMARY_PROMPT.format(chunk=chunk))
                    facts = llm_local(FACT_PROMPT.format(chunk=chunk))
                    sentiment = llm_local(SENTIMENT_PROMPT.format(chunk=chunk))

                    md_file.write(f"# Call Transcript - {transcript} - Part {i + 1}\n\nSummary: {summary}\n\nFacts:\n{facts}\n\nSentiment: {sentiment}\n\n---\n")

    print("Summarizing complete")

if __name__ == "__main__":
    process_wav_files()
    summarize_transcripts()