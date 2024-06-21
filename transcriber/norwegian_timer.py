import numpy as np
import speech_recognition as sr
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import threading


def main():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    
    processor = AutoProcessor.from_pretrained("NbAiLab/nb-whisper-base-beta")
    audio_model = AutoModelForSpeechSeq2Seq.from_pretrained("NbAiLab/nb-whisper-base-beta")

    print("Model loaded. Press Enter to start recording. Press Enter again to stop recording.")
    input("Press Enter to start recording...")

    audio = record_audio()
    if audio is not None:
        print("Transcribing...")
        audio_np = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
        inputs = processor(audio_np, return_tensors="pt", sampling_rate=16000, language="no")
        input_features = inputs.input_features
        generated_ids = audio_model.generate(input_features=input_features, language="no")
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("\nTranscription:")
        print(text)
    else:
        print("No audio is detected...")

def record_audio():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    stop_recording = threading.Event()

    def wait_for_stop():
        input()
        stop_recording.set()

    with sr.Microphone(sample_rate=16000) as source:
        recorder.adjust_for_ambient_noise(source)
        print("Recording... Press Enter to stop.")
        
        audio = None

        stop_thread = threading.Thread(target=wait_for_stop)
        stop_thread.start()
        
        while not stop_recording.is_set():
            try:
                audio = recorder.listen(source, timeout=1, phrase_time_limit=None)
            except sr.WaitTimeoutError:
                pass
        
    print("Recording stopped.")
    return audio

if __name__ == "__main__":
    main()