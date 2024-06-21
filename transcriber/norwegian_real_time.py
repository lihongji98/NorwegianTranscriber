import numpy as np
import speech_recognition as sr

from queue import Queue
from time import sleep

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def main():
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)

    # model = "tiny"
    # audio_model = whisper.load_model(model)
    processor = AutoProcessor.from_pretrained("NbAiLab/nb-whisper-tiny-beta")
    audio_model = AutoModelForSpeechSeq2Seq.from_pretrained("NbAiLab/nb-whisper-tiny-beta")

    record_timeout = 3.5

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)
    
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    transcription = [""]

    print("Model loaded.\n")

    while True:
        try:
            if not data_queue.empty():                
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                inputs = processor(audio_np, return_tensors="pt", sampling_rate=16000, language="no")
                input_features = inputs.input_features
                generated_ids = audio_model.generate(input_features=input_features, language="no")
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                # result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                # text = result['text'].strip()
                print(text)

                # os.system('cls' if os.name=='nt' else 'clear')
                transcription.append(text)

                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                sleep(0.1)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
