from pydub import AudioSegment
import soundfile as sfile
import numpy as np
from pathlib import Path
import pandas as pd

from .maths import *
from .globals import *

# import librosa
# import librosa.display


def load_crepe_keyframes(filename):
    df = pd.read_csv(filename)
    freq = to_keyframes(df['frequency'], len(df['frequency']) / df['time'].values[-1])
    confidence = to_keyframes(df['confidence'], len(df['frequency']) / df['time'].values[-1])
    return freq, confidence


def load_dbnorm_keyframes(filename, window=None, caching=True):
    window = window if window is None else window * fps
    return norm(load_db_keyframes(filename, caching), window=window)


def load_db_keyframes(filename, caching=True):
    audio, dbs = load_db(filename, caching)
    return to_keyframes(dbs, audio.frame_rate)


def load_db(filename, caching=True):
    audio = AudioSegment.from_mp3(filename)

    # Load the cache if enabled
    cachepath = Path(filename).with_suffix(".npy")
    if caching and cachepath.exists():
        print(f"Restoring cached decibels: {cachepath}")
        return audio, np.load(cachepath.as_posix())

    # Convert samples to decibels
    signal, sr = sfile.read(filename)
    samples = audio.get_array_of_samples()
    samples_sf = 0
    try:
        samples_sf = signal[:, 0]  # use the first channel for dual
    except:
        samples_sf = signal  # for mono
    decibels = [convert_to_decibel(i) for i in samples_sf]  # idk how to vectorize this so it's a bit slow, I'm implementing caching for now

    # Write the cache if enabled
    if caching:
        print(f"Caching decibels: {cachepath}")
        np.save(cachepath.as_posix(), decibels)

    return audio, decibels


def to_keyframes(dbs, original_sps):
    start = 0
    total_seconds = len(dbs) / original_sps
    # print(len(dbs), original_sps, total_seconds)
    # start=0
    # total_seconds=5

    frames = int(fps * total_seconds)

    dt = np.zeros(frames)
    for i in range(frames):
        # frame --> seconds
        t = (i) / fps + start
        t1 = (i + 1) / fps + start
        # print(t, t1)

        d = dbs[int(t * original_sps):int((t1) * original_sps)]
        dt[i] = np.mean(d)

    return smooth_1euro(dt)


def convert_to_decibel(arr):
    ref = 1
    if arr != 0:
        return 20 * np.log10(np.abs(arr) / ref)

    else:
        return -60


def play_wav(seg):
    import simpleaudio
    return simpleaudio.play_buffer(
            seg.raw_data,
            num_channels=seg.channels,
            bytes_per_sample=seg.sample_width,
            sample_rate=seg.frame_rate
    )
