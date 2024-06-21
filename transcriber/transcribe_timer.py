import argparse
import numpy as np
import speech_recognition as sr
import whisper
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    args = parser.parse_args()

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False
    
    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    print("Model loaded. Press Enter to start recording (max 30 seconds).")
    input("press to continue...")

    with sr.Microphone(sample_rate=16000) as source:
        print("Recording... (max 30 seconds)")
        try:
            audio = recorder.record(source, duration=10)
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        else:
            print("Recording complete.")

    print("Transcribing...")
    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

    result = audio_model.transcribe(audio_data, fp16=torch.cuda.is_available())
    text = result['text'].strip()

    print("\nTranscription:")
    print(text)

if __name__ == "__main__":
    main()