import sys
import os
import pyaudio
import wave
import threading
import requests
import datetime
import tempfile
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog, QLabel, QComboBox
from PyQt5.QtGui import QIcon, QPixmap

class RecordingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.recording_pixmap = QPixmap("recording.png")
        self.not_recording_pixmap = QPixmap("notrecording.png")
        self.is_recording = False
        self.audio_thread = None
        self.stream = None
        self.p = None
        self.wf = None
        
        # Load environment variables
        load_dotenv()
        self.WHISPERCPP_URL = os.getenv("WHISPERCPP_URL")
        self.LLAMACPP_URL = os.getenv("LLAMACPP_URL")
        self.SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE")
        self.SUMMARY_PROMPT = os.getenv("SUMMARY_PROMPT")
        self.FACT_PROMPT = os.getenv("FACT_PROMPT")
        self.SENTIMENT_PROMPT = os.getenv("SENTIMENT_PROMPT")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE"))
        self.TOP_P = float(os.getenv("TOP_P"))
        self.MAX_TOKENS = float(os.getenv("MAX_TOKENS"))
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Meeting Recorder')
        self.setGeometry(500, 500, 500, 150)

        # Main vertical layout
        main_layout = QVBoxLayout()

        # Horizontal layout for the status label and device combo box
        top_layout = QHBoxLayout()

        self.status_label = QLabel(self)
        self.status_label.setPixmap(self.not_recording_pixmap)
        top_layout.addWidget(self.status_label)

        self.device_combo = QComboBox(self)
        self.device_combo.addItems(self.get_device_names())
        top_layout.addWidget(self.device_combo)

        main_layout.addLayout(top_layout)

        # Horizontal layout for the buttons
        button_layout = QHBoxLayout()

        self.record_button = QPushButton('Record', self)
        self.record_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_button)

        self.transcribe_button = QPushButton('Transcribe', self)
        self.transcribe_button.clicked.connect(self.transcribe)
        button_layout.addWidget(self.transcribe_button)

        self.clean_button = QPushButton('Clean', self)
        self.clean_button.clicked.connect(self.clean)
        button_layout.addWidget(self.clean_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def get_device_names(self):
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        device_names = []
        for i in range(num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                device_names.append(device_info.get('name'))
        p.terminate()
        return device_names

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        filename, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter filename:')
        if ok and filename:
            self.is_recording = True
            self.record_button.setText('Stop Recording')
            self.status_label.setPixmap(self.recording_pixmap)
            selected_device_index = self.device_combo.currentIndex()
            self.audio_thread = threading.Thread(target=self.record_audio, args=(filename, selected_device_index))
            self.audio_thread.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.setText('Record')
            self.status_label.setPixmap(self.not_recording_pixmap)
            if self.audio_thread:
                self.audio_thread.join()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
            if self.wf:
                self.wf.close()

    def record_audio(self, filename, device_index):
        chunk_size = 1024
        sampling_rate = 16000
        num_channels = 1

        self.p = pyaudio.PyAudio()
        file_path = f"{filename}.wav"

        self.wf = wave.open(file_path, 'wb')
        self.wf.setnchannels(num_channels)
        self.wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        self.wf.setframerate(sampling_rate)

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=num_channels,
            rate=sampling_rate,
            input=True,
            frames_per_buffer=chunk_size,
            input_device_index=device_index
        )

        print(f"Recording to {file_path}. Press 'Stop Recording' to stop...")

        while self.is_recording:
            data = self.stream.read(chunk_size)
            self.wf.writeframes(data)

        print(f"Audio saved to {file_path}")

    def transcribe(self):
        # Run the transcription and summarization in the background
        threading.Thread(target=self.run_transcription_and_summarization).start()

    def whisper_api(self, file):
        """Transcribe audio file using Whisper API."""
        files = {"file": file}
        api_data = {
            "temperature": "0.0",
            "response_format": "json"
        }
        response = requests.post(self.WHISPERCPP_URL, data=api_data, files=files)
        return response.json()["text"]

    def llm_local(self, prompt):
        client = OpenAI(api_key="doesntmatter", base_url=self.LLAMACPP_URL)
        messages=[{"role": "system", "content": self.SYSTEM_MESSAGE},{"role": "user", "content": prompt}]
        response = client.chat.completions.create(model="whatever", max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE, top_p=self.TOP_P, messages=messages)
        return response.choices[0].message.content

    def trim_silence(self, filename):
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

    def process_wav_files(self):
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
            self.trim_silence(wav_file)

            with open(wav_file, "rb") as file:
                print("Transcribing: " + wav_file)
                output_text = self.whisper_api(file)
                output_file = os.path.splitext(wav_file)[0] + ".tns"
                with open(output_file, "w") as output:
                    output.write(output_text)

    def chunk_transcript(self, string, chunk_size):
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

    def summarize_transcripts(self):
        """Summarize transcript files."""
        today = datetime.datetime.now().strftime('%Y%m%d')
        summary_filename = "summary-" + today + ".md"
        transcript_files = [f for f in os.listdir(".") if f.endswith(".tns")]

        for transcript in transcript_files:
            print("Summarizing: " + transcript)
            with open(transcript, "r") as file:
                transcript_data = file.read()
                chunked_data = self.chunk_transcript(transcript_data, self.CHUNK_SIZE)

                with open(summary_filename, "a") as md_file:
                    for i, chunk in enumerate(chunked_data):
                        print("Processing part " + str(i))
                        summary = self.llm_local(self.SUMMARY_PROMPT.format(chunk=chunk))
                        facts = self.llm_local(self.FACT_PROMPT.format(chunk=chunk))
                        sentiment = self.llm_local(self.SENTIMENT_PROMPT.format(chunk=chunk))

                        md_file.write(f"# Call Transcript - {transcript} - Part {i + 1}\n\nSummary: {summary}\n\nFacts:\n{facts}\n\nSentiment: {sentiment}\n\n---\n")

        print("Summarizing complete")

    def run_transcription_and_summarization(self):
        self.process_wav_files()
        self.summarize_transcripts()

    def clean(self):
        print("Cleaning files...")
        try:
            for file in os.listdir('.'):
                if file.endswith(('.wav', '.tns')):
                    os.remove(file)
                    print(f"Deleted: {file}")
            print("Cleaning complete.")
        except Exception as e:
            print(f"An error occurred while cleaning files: {e}")

    def closeEvent(self, event):
        self.stop_recording()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("headphones.png"))
    ex = RecordingApp()
    ex.show()
    sys.exit(app.exec_())
