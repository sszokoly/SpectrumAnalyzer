#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pyaudio
import struct
from collections import deque
from threading import Thread


class Mic(object):
    """Microphone streaming source."""
    DEPTH_TO_DTYPE = {
        8: (pyaudio.paInt8, 'b'),
        16: (pyaudio.paInt16, 'h'),
        24: (pyaudio.paInt24, 'i'),
        32: (pyaudio.paInt32, 'i'),
    }

    def __init__(self, buffer_len_in_chunks=8):
        """Initializes a Mic instance."""
        self.buffer_len_in_chunks = buffer_len_in_chunks
        self._deque = deque(maxlen=buffer_len_in_chunks)
        self.end_thread = False
        self._stream = None
        self._th = None

    def run(self, rate=8000, channels=1, chunk=1024, sampling_bit_depth=16):
        """Initializes and starts pyaudio stream using the mic as source.

        Args:
            rate (int): sampling rate, Defaults to 8000.
            channels (int): number of microphone sources, Defaults to 1.
            chunk (int): size of internal buffer in number of frames,
                Defaults to 1024.
            sampling_bit_depth (int): sampling bit depth, Defaults to 16.

        Returns:
            None.
        """
        if self.is_alive() and self.is_active():
            self.stop()

        self._th = Thread(
            target=self._worker,
            daemon=True,
            args=(rate, channels, chunk, sampling_bit_depth)
        )
        self._th.start()
        self.end_thread = False

    def stop(self):
        """Stops terminates thread and pyaudio stream."""
        self.end_thread = True
        self._th.join()

    def is_alive(self):
        """bool: Returns True if thread is alive."""
        return self._th and self._th.is_alive()

    def is_active(self):
        """bool: Returns True if pyaudio stream is active."""
        return self._stream and self._stream.is_active()

    def _worker(self, rate, channels, chunk, sampling_bit_depth):
        """Thread worker process which reads one chunk and appends
        it to the internal deque.

        Args:
            rate (int): sampling rate.
            channels (int): number of microphone sources.
            chunk (int): size of internal buffer in number of frames.
            sampling_bit_depth (int): sampling bit depth.

        Returns:
            None.
        """
        pya = pyaudio.PyAudio()
        format, cformat = self.DEPTH_TO_DTYPE.get(sampling_bit_depth, 16)
        self._stream = pya.open(
            rate=rate,
            channels=channels,
            format=format,
            input=True,
            frames_per_buffer=chunk,
        )

        while self._stream.is_active() and not self.end_thread:
            self._deque.appendleft(
                struct.unpack(str(chunk) + cformat, self._stream.read(chunk))
            )

        self._stream.stop_stream()
        self._stream.close()
        pya.terminate()

    def get(self):
        """Returns one chunk from the internal deque buffer.

        Returns:
            tuple: tuple of integers forming one chunk.
        """
        try:
            return self._deque.pop()
        except IndexError:
            return None


if __name__ == "__main__":
    import time
    chunks = 0
    mic = Mic()
    mic.run(rate=48000, chunk=1024, sampling_bit_depth=16)
    while True:
        try:
            data = mic.get()
            if data:
                chunks += 1
                print(data)
            else:
                time.sleep(0.01)
        except KeyboardInterrupt:
            break
    print("Number of chunks: {}".format(chunks))
    mic.stop()
