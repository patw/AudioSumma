import sys
import os
import subprocess
import pyaudio
import wave
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog, QLabel
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
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Meeting Recorder')
        self.setGeometry(500, 500, 500, 150)

        layout = QHBoxLayout()

        self.status_label = QLabel(self)
        self.status_label.setPixmap(self.not_recording_pixmap)
        layout.addWidget(self.status_label)

        self.record_button = QPushButton('Record', self)
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)

        self.transcribe_button = QPushButton('Transcribe', self)
        self.transcribe_button.clicked.connect(self.transcribe)
        layout.addWidget(self.transcribe_button)

        self.clean_button = QPushButton('Clean', self)
        self.clean_button.clicked.connect(self.clean)
        layout.addWidget(self.clean_button)

        self.setLayout(layout)

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
            self.audio_thread = threading.Thread(target=self.record_audio, args=(filename,))
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

    def record_audio(self, filename):
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
            input_device_index=0
        )

        print(f"Recording to {file_path}. Press 'Stop Recording' to stop...")

        while self.is_recording:
            data = self.stream.read(chunk_size)
            self.wf.writeframes(data)

        print(f"Audio saved to {file_path}")

    def transcribe(self):
        subprocess.Popen(['python', 'summarize.py'])

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