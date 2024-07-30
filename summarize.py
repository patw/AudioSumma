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
    # Whisper supports multiple files, but we're sending one
    files = {"file": file}
    
    # Required API call data
    api_data = {
        "temperature": "0.0",
        "response_format": "json"
    }

    # Call API and return text
    response = requests.post(WHISPERCPP_URL, data=api_data, files=files)
    return response.json()["text"]

def llama_api(prompt):
    # Format prompt before sending
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

# Use ffmpeg to trim silence in wav files, to prevent issues with 
# whisper.cpp stopping the transcode if it detects a large amount of silence
def trim_silence(filename):
    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name

    # Construct the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", filename,
        "-af", "silenceremove=stop_threshold=-40dB:stop_duration=1:stop_periods=-1",
        "-y",  # Overwrite output file if it exists
        temp_filename
    ]

    # Run the FFmpeg command
    result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)

    # If FFmpeg command was successful, replace the original file
    os.replace(temp_filename, filename)

# Iterate over each WAV file and transcode with whisper API
wav_files = [f for f in os.listdir(".") if f.endswith(".wav")]
for wav_file in wav_files:

    # Trim silence on the wav file first
    print("Trimming silence: " + wav_file)
    trim_silence(wav_file)
    
    # Open the WAV file for sending to whisper REST API
    with open(wav_file, "rb") as file:
        print("Transcribing: " + wav_file)
        # Call whisper API to transcode file
        output_text = whisper_api(file)

        # Generate the output file name by replacing the extension with .tns
        output_file = os.path.splitext(wav_file)[0] + ".tns"

        # Write the output text to the file
        with open(output_file, "w") as output:
            output.write(output_text)
        
# Chunk the full transcript into multiple parts to fit in the context window
# and allow for better reasoning capability
def chunk_transcript(string, chunk_size):
    chunks = []
    lines = string.split("\n")  # Split the string on newline characters
    current_chunk = ""
    for line in lines:
        current_chunk += line  # Build up the string until the chunk size is reached
        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append(current_chunk)
    return chunks

# Get the current date in yyyymmdd format
today = datetime.datetime.now().strftime('%Y%m%d')

# Modify the filename by appending the current date
summary_filename = "summary-" + today + ".md"

# Get the list of transcript files in the current directory
transcript_files = [f for f in os.listdir(".") if f.endswith(".tns")]

# Iterate over each WAV file
for transcript in transcript_files: 
    print("Summarizing: " + transcript)

    # Open the WAV file
    with open(transcript, "r") as file:
        transcript_data = file.read()

        # chunk the transcript so we don't blow out the context window
        chunked_data = chunk_transcript(transcript_data, CHUNK_SIZE)

        # Iterate through the chunks, and summarize them
        for i, chunk in enumerate(chunked_data):
            with open(summary_filename, "a") as md_file:
                # Generate call summary
                summary_prompt = SUMMARY_PROMPT.format(chunk=chunk)
                summary = llama_api(summary_prompt)

                # Generate fact summary
                fact_prompt = FACT_PROMPT.format(chunk=chunk)
                facts = llama_api(fact_prompt)

                # Generate call sentiment
                sentiment_prompt = SENTIMENT_PROMPT.format(chunk=chunk)
                sentiment = llama_api(sentiment_prompt)

                # Write the notes
                md_file.write(f"# Call Transcript - {transcript} - Part {i + 1}\n\nSummary: {summary}\n\nFacts:\n{facts}\n\nSentiment: {sentiment}\n\n---\n")

print("Summarizing complete")