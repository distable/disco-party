from pydub import AudioSegment
import soundfile as sfile
from pathlib import Path
import pandas as pd

from classes.printlib import trace_decorator
from . import constants
from .maths import *

# TODO melodic speed
# TODO harmonic chg speed


@trace_decorator
def load_crepe_keyframes(filename):
    df = pd.read_csv(filename)
    freq = to_keyframes(df['frequency'], len(df['frequency']) / df['time'].values[-1])
    confidence = to_keyframes(df['confidence'], len(df['frequency']) / df['time'].values[-1])
    return freq, confidence

@trace_decorator
def load_harmonics(filename):
    import librosa
    from src_plugins.disco_party.keyfinder import Tonal_Fragment

    # This audio takes a long time to load because it has a very high sampling rate; be patient.
    # the load function generates a tuple consisting of an audio object y and its sampling rate sr
    y, sr = librosa.load(filename)

    # This function filters out the harmonic part of the sound file from the percussive part, allowing for
    # more accurate harmonic analysis
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    unebarque_fsharp_min = Tonal_Fragment(y_harmonic, sr, tend=22)
    unebarque_fsharp_min.print_chroma()
    # ipd.Audio(filename)

    unebarque_fsharp_min.print_key()
    unebarque_fsharp_min.corr_table()
    unebarque_fsharp_min.chromagram("Une Barque sur l\'Ocean")


@trace_decorator
def load_dbnorm(filename, window=None, caching=True):
    window = window if window is None else window * constants.fps
    return norm(load_db_keyframes(filename, caching), window=window)


@trace_decorator
def load_db_keyframes(filename, caching=True):
    audio, dbs = load_db(filename, caching)
    return to_keyframes(dbs, audio.frame_rate)


@trace_decorator
def load_db(filename, caching=True):
    audio = AudioSegment.from_file(filename)

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


@trace_decorator
def to_keyframes(dbs, original_sps):
    start = 0
    total_seconds = len(dbs) / original_sps
    # print(len(dbs), original_sps, total_seconds)
    # start=0
    # total_seconds=5

    frames = int(constants.fps * total_seconds)

    dt = np.zeros(frames)
    for i in range(frames):
        # frame --> seconds
        t = (i) / constants.fps + start
        t1 = (i + 1) / constants.fps + start
        # print(t, t1)

        d = dbs[int(t * original_sps):int((t1) * original_sps)]
        dt[i] = np.mean(d)

    return smooth_1euro(dt)


@trace_decorator
def convert_to_decibel(arr):
    ref = 1
    if arr != 0:
        return 20 * np.log10(np.abs(arr) / ref)

    else:
        return -60


@trace_decorator
def play_wav(audioseg, t):
    import simpleaudio
    if t is not None:
        audioseg = audioseg.get_sample_slice(int(t * audioseg.frame_rate))

    return simpleaudio.play_buffer(
            audioseg.raw_data,
            num_channels=audioseg.channels,
            bytes_per_sample=audioseg.sample_width,
            sample_rate=audioseg.frame_rate
    )
