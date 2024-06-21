import speech_recognition as sr
from queue import Queue
from time import sleep
import numpy as np
from datetime import datetime

source = sr.Microphone(sample_rate=16000)
recorder = sr.Recognizer()
recorder.energy_threshold = 1000
recorder.dynamic_energy_threshold = False

record_timeout = 2
phrase_timeout = 3

data_queue = Queue()
phrase_time = None

with source:
    recorder.adjust_for_ambient_noise (source)

def record_callback(_, audio:sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

while True:
    try:
        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            audio_data = b''.join(data_queue.queue)

            data_queue.queue.clear()

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            print(audio_np.shape)

        else:
            sleep(0.25)
    except KeyboardInterrupt:
            break