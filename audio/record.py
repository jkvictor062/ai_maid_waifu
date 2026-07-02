import queue
import time
from collections import deque
from tarfile import BLOCKSIZE

import numpy as np
import sounddevice as sd


class VoiceRecorder:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCKSIZE = 512

    START_THRESHOLD = 0.015
    STOP_THRESHOLD = 0.008

    END_SILENCE = 1.2
    MAX_WAIT = 10

    def __init__(self) -> None:
        self.queue = queue.Queue()

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        self.queue.put(indata.copy())

    def record(self):
        frames = []
        preroll = deque(maxlen=10)
        speaking = False
        silence_time = 0
        start = time.monotonic()
        rms_window = deque(maxlen=8)

        while not self.queue.empty():
            self.queue.get_nowait()
        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            blocksize=self.BLOCKSIZE,
            dtype="float32",
            callback=self.callback,
        ):
            while True:
                chunk = self.queue.get()

                preroll.append(chunk)

                current_rms = np.sqrt(np.mean(chunk**2))

                rms_window.append(current_rms)

                rms = np.mean(rms_window)

                if not speaking and time.monotonic() - start > self.MAX_WAIT:
                    print("Timeout")
                    return None

                if not speaking:
                    if rms >= self.START_THRESHOLD:
                        print("Speech detected")

                        speaking = True

                        frames.extend(preroll)
                        frames.append(chunk)

                        silence_time = 0

                else:
                    frames.append(chunk)

                    if rms < self.STOP_THRESHOLD:
                        silence_time += self.BLOCKSIZE / self.SAMPLE_RATE

                        if silence_time >= self.END_SILENCE:
                            print("End of speech")

                            break

                    else:
                        silence_time = 0

        audio = np.concatenate(frames, axis=0).squeeze().astype(np.float32)
        print(audio.shape)
        print(audio.dtype)
        return audio
