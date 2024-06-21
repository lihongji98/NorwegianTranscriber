import numpy as np
import speech_recognition as sr
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import threading
from datetime import datetime, time
import time


def transcriber(model_size):
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    
    processor = AutoProcessor.from_pretrained(f"NbAiLab/nb-whisper-" + model_size + "-beta")
    audio_model = AutoModelForSpeechSeq2Seq.from_pretrained("NbAiLab/nb-whisper-" + model_size + "-beta")

    input("Press <Enter> to start recording, (better shorter than 30 seconds)")

    audio_np = record_audio()
    if audio_np is not None:
        print("Transcribing...")
        inputs = processor(audio_np, return_tensors="pt", sampling_rate=16000, language="no")
        input_features = inputs.input_features
        generated_ids = audio_model.generate(input_features=input_features, language="no")
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # print("\nTranscription:")
        # print(text)

        return text
    else:
        print("No audio is detected...")
        return

def record_audio():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    stop_recording = threading.Event()

    def wait_for_stop():
        input()
        stop_recording.set()
    
    def show_timer():
        start_time = datetime.now()
        while not stop_recording.is_set():
            elapsed_time = datetime.now() - start_time
            print(f"\rElapsed time: {elapsed_time.seconds} seconds", end="")
            time.sleep(1)

    with sr.Microphone(sample_rate=16000) as source:
        recorder.adjust_for_ambient_noise(source)
        print("Recording... Press <Enter> to stop.")
        
        audio_chunks = []
        stop_thread = threading.Thread(target=wait_for_stop)
        timer_thread = threading.Thread(target=show_timer)

        stop_thread.start()
        timer_thread.start()

        while not stop_recording.is_set():
            try:
                audio = recorder.listen(source, timeout=1, phrase_time_limit=None)
                audio_chunks.append(audio)
            except sr.WaitTimeoutError:
                pass
    
    timer_thread.join()
    
    combined_audio = b''.join(chunk.get_raw_data() for chunk in audio_chunks)
    audio_np = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0
    print("Recording stopped.")

    return audio_np