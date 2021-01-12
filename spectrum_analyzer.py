#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mic import Mic


RATE = 8000
CHUNK = 2048
BIT_DEPTH = 16
AXIS_COLOR = "#dddddd"
MAXY = 2 ** (BIT_DEPTH - 1) - 1
MAXX = round(1000 * CHUNK / RATE)

chunks = 0
mic = Mic()
mic.run(rate=RATE, chunk=CHUNK, sampling_bit_depth=BIT_DEPTH)
fig, (ax_wave, ax_pds) = plt.subplots(2, figsize=(12, 8))
fig.patch.set_facecolor("black")
matplotlib.rc("axes", edgecolor="white")


def init():
    """Initializes the plots and texts used by Matplotlib Animation."""
    global waveline, psdline, chunk_count, energy
    for ax in ax_wave, ax_pds:
        ax.set_facecolor("black")
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.xaxis.label.set_color(AXIS_COLOR)
        ax.tick_params(axis="both", colors=AXIS_COLOR)

    ax_wave.set_title("AUDIO WAVEFORM ({} Hz)".format(RATE), color=AXIS_COLOR)
    ax_wave.set_xlabel("Time (ms)"),
    ax_wave.set_ylabel("Applitude (V)", color=AXIS_COLOR),
    ax_wave.set(xlim=(0, MAXX), ylim=(-1, 1))
    wave_x = np.arange(0, MAXX, MAXX / CHUNK)
    wave_y = np.zeros(CHUNK)
    waveline, = ax_wave.plot(wave_x, wave_y, "-", c="cyan", lw=1)
    chunk_count = ax_wave.text(0.02, 1, "", color=AXIS_COLOR, fontsize=12,
                               transform=ax_wave.transAxes)

    ax_pds.set_title("POWER SPECTRUM", color=AXIS_COLOR)
    ax_pds.set_xlabel("Frequency (Hz)"),
    ax_pds.set_ylabel("Power (dBm)", color=AXIS_COLOR),
    ax_pds.set(xlim=(0, RATE//2), ylim=(-100, 10))
    ax_pds.set_facecolor("black")
    pds_x = np.arange(0, RATE + RATE/CHUNK, RATE/CHUNK)[:CHUNK//2+1]
    pds_y = np.zeros(CHUNK//2+1)
    psdline, = ax_pds.plot(pds_x, pds_y, "-", c="cyan", lw=1)
    energy = ax_pds.text(0.02, 1, "", color=AXIS_COLOR, fontsize=12,
                         transform=ax_pds.transAxes)

    return waveline, psdline, chunk_count, energy


def animate(i):
    """Updates the plots and texts."""
    global chunks
    chunk = None
    while chunk is None:
        chunk = mic.get()
    chunks += 1
    signal = np.array(chunk) / MAXY
    S = np.fft.fft(signal) / (np.size(signal))
    psd = np.abs(S * np.conj(S))
    psd_log = 10 * np.log(psd * 1000)
    mjoules_per_sec = round(np.sum(psd) / (CHUNK/RATE) * 1000, 3)
    psdline.set_ydata(psd_log[:np.size(S)//2+1])
    waveline.set_ydata(signal)
    chunk_count.set_text("Num. of Chunks: {0}".format(chunks))
    energy.set_text("Energy (mJ/s): {0}".format(mjoules_per_sec))
    plt.tight_layout()


if __name__ == "__main__":
    anim = FuncAnimation(
        fig, animate,
        interval=MAXX//2,
        init_func=init
    )
    plt.draw()
    plt.show()
