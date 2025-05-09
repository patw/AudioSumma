import sys
import os
import pyaudio
import wave
import threading
import requests
import datetime
import tempfile
import subprocess
import json
from openai import OpenAI
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QInputDialog, QLabel, QComboBox,
                           QDialog, QFormLayout, QLineEdit, QDoubleSpinBox, 
                           QSpinBox, QTextEdit)
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
        
        # Configuration with defaults
        self.config = {
            "OUTPUT_DIR": os.path.join(os.path.expanduser("~"), "Documents"),
            "WHISPERCPP_URL": "http://localhost:8081/inference",
            "LLAMACPP_URL": "http://localhost:8080/v1",
            "SYSTEM_MESSAGE": "You are a friendly chatbot that summarizes call transcripts",
            "SUMMARY_PROMPT": "Call Transcript: {chunk}\n\nInstruction: Summarize the above call transcript but DO NOT MENTION THE TRANSCRIPT",
            "FACT_PROMPT": "Call Transcript: {chunk}\n\nInstruction: Summarize all the facts in the transcript, one per line bullet point",
            "SENTIMENT_PROMPT": "Call Transcript: {chunk}\n\nInstruction: Summarize the sentiment for topics in the above call transcript but DO NOT MENTION THE TRANSCRIPT",
            "CHUNK_SIZE": 12288,
            "TEMPERATURE": 0.6,
            "TOP_P": 0.9,
            "MAX_TOKENS": 2000
        }
        
        # Try to load saved config
        self.load_config()
        
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

        self.settings_button = QPushButton('Settings', self)
        self.settings_button.clicked.connect(self.show_settings)
        button_layout.addWidget(self.settings_button)

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
        os.makedirs(self.config["OUTPUT_DIR"], exist_ok=True)
        file_path = os.path.join(self.config["OUTPUT_DIR"], f"{filename}.wav")

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
        response = requests.post(self.config["WHISPERCPP_URL"], data=api_data, files=files)
        return response.json()["text"]

    def llm_local(self, prompt):
        client = OpenAI(api_key="doesntmatter", base_url=self.config["LLAMACPP_URL"])
        messages=[{"role": "system", "content": self.config["SYSTEM_MESSAGE"]},{"role": "user", "content": prompt}]
        response = client.chat.completions.create(model="whatever", 
                                               max_tokens=self.config["MAX_TOKENS"], 
                                               temperature=self.config["TEMPERATURE"], 
                                               top_p=self.config["TOP_P"], 
                                               messages=messages)
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
        wav_files = [f for f in os.listdir(self.config["OUTPUT_DIR"]) if f.endswith(".wav")]
        for wav_file in wav_files:
            # Generate the expected transcript filename
            transcript_file = os.path.splitext(wav_file)[0] + ".tns"

            # Check if transcript already exists
            full_transcript_path = os.path.join(self.config["OUTPUT_DIR"], transcript_file)
            if os.path.exists(full_transcript_path):
                print(f"Transcript already exists for {wav_file}, skipping transcription")
                continue

            full_wav_path = os.path.join(self.config["OUTPUT_DIR"], wav_file)
            print("Trimming silence: " + full_wav_path)
            self.trim_silence(full_wav_path)

            with open(full_wav_path, "rb") as file:
                print("Transcribing: " + full_wav_path)
                output_text = self.whisper_api(file)
                output_file = os.path.join(self.config["OUTPUT_DIR"], os.path.splitext(wav_file)[0] + ".tns")
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
        summary_filename = os.path.join(self.config["OUTPUT_DIR"], "summary-" + today + ".md")
        transcript_files = [f for f in os.listdir(self.config["OUTPUT_DIR"]) if f.endswith(".tns")]

        for transcript in transcript_files:
            full_path = os.path.join(self.config["OUTPUT_DIR"], transcript)
            print("Summarizing: " + full_path)
            with open(full_path, "r") as file:
                transcript_data = file.read()
                chunked_data = self.chunk_transcript(transcript_data, self.config["CHUNK_SIZE"])

                with open(summary_filename, "a") as md_file:
                    for i, chunk in enumerate(chunked_data):
                        print("Processing part " + str(i))
                        summary = self.llm_local(self.config["SUMMARY_PROMPT"].format(chunk=chunk))
                        facts = self.llm_local(self.config["FACT_PROMPT"].format(chunk=chunk))
                        sentiment = self.llm_local(self.config["SENTIMENT_PROMPT"].format(chunk=chunk))

                        md_file.write(f"# Call Transcript - {transcript} - Part {i + 1}\n\nSummary: {summary}\n\nFacts:\n{facts}\n\nSentiment: {sentiment}\n\n---\n")

        print("Summarizing complete")

    def run_transcription_and_summarization(self):
        self.process_wav_files()
        self.summarize_transcripts()

    def clean(self):
        print("Cleaning files...")
        try:
            for file in os.listdir(self.config["OUTPUT_DIR"]):
                if file.endswith(('.wav', '.tns')):
                    os.remove(file)
                    print(f"Deleted: {file}")
            print("Cleaning complete.")
        except Exception as e:
            print(f"An error occurred while cleaning files: {e}")

    def load_config(self):
        """Load configuration from config.json if it exists."""
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self):
        """Save configuration to config.json."""
        try:
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def show_settings(self):
        """Show configuration dialog."""
        dialog = ConfigDialog(self.config, self)
        dialog.resize(600, 400)  # Make dialog larger
        if dialog.exec_():
            self.config.update(dialog.get_values())
            self.save_config()

    def closeEvent(self, event):
        self.stop_recording()
        event.accept()

class ConfigDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        
        layout = QFormLayout()
        layout.setSpacing(5)  # Add more spacing between rows
        
        # Use QTextEdit for multi-line text fields
        self.whisper_url = QTextEdit(config["WHISPERCPP_URL"])
        self.llama_url = QTextEdit(config["LLAMACPP_URL"])
        self.system_msg = QTextEdit(config["SYSTEM_MESSAGE"])
        self.summary_prompt = QTextEdit(config["SUMMARY_PROMPT"])
        self.fact_prompt = QTextEdit(config["FACT_PROMPT"])
        self.sentiment_prompt = QTextEdit(config["SENTIMENT_PROMPT"])
        
        # Set minimum sizes for text areas
        for text_edit in [self.whisper_url, self.llama_url, self.system_msg,
                         self.summary_prompt, self.fact_prompt, self.sentiment_prompt]:
            text_edit.setMinimumSize(500, 80)  # Width, Height
            text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1000, 32000)
        self.chunk_size.setValue(config["CHUNK_SIZE"])
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 1.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(config["TEMPERATURE"])
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.1, 1.0)
        self.top_p.setSingleStep(0.1)
        self.top_p.setValue(config["TOP_P"])
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(512, 4096)
        self.max_tokens.setValue(config["MAX_TOKENS"])
        
        layout.addRow("Whisper URL:", self.whisper_url)
        layout.addRow("LLaMA URL:", self.llama_url)
        layout.addRow("System Message:", self.system_msg)
        layout.addRow("Summary Prompt:", self.summary_prompt)
        layout.addRow("Fact Prompt:", self.fact_prompt)
        layout.addRow("Sentiment Prompt:", self.sentiment_prompt)
        layout.addRow("Chunk Size:", self.chunk_size)
        layout.addRow("Temperature:", self.temperature)
        layout.addRow("Top P:", self.top_p)
        self.output_dir = QTextEdit(config["OUTPUT_DIR"])
        self.output_dir.setMinimumSize(500, 80)
        self.output_dir.setLineWrapMode(QTextEdit.WidgetWidth)
        
        layout.addRow("Max Tokens:", self.max_tokens)
        layout.addRow("Output Directory:", self.output_dir)
        
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        
        layout.addRow(buttons)
        self.setLayout(layout)
    
    def get_values(self):
        return {
            "WHISPERCPP_URL": self.whisper_url.toPlainText(),
            "LLAMACPP_URL": self.llama_url.toPlainText(),
            "SYSTEM_MESSAGE": self.system_msg.toPlainText(),
            "SUMMARY_PROMPT": self.summary_prompt.toPlainText(),
            "FACT_PROMPT": self.fact_prompt.toPlainText(),
            "SENTIMENT_PROMPT": self.sentiment_prompt.toPlainText(),
            "CHUNK_SIZE": self.chunk_size.value(),
            "TEMPERATURE": self.temperature.value(),
            "TOP_P": self.top_p.value(),
            "MAX_TOKENS": self.max_tokens.value(),
            "OUTPUT_DIR": self.output_dir.toPlainText()
        }

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("headphones.png"))
    ex = RecordingApp()
    ex.show()
    sys.exit(app.exec_())
