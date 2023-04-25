import librosa
from pydub import AudioSegment
import soundfile as sfile
from pathlib import Path
import pandas as pd

from audio_processing.modules.loudness import lufs_meter
from audio_processing.modules import filter
from src_core.classes.printlib import trace_decorator
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
    return load_db_keyframes(filename, caching)


@trace_decorator
def load_db_keyframes(filename, caching=True):
    dbs = load_db(filename, caching)
    return dbs
    # return to_keyframes(dbs, audio.frame_rate)


@trace_decorator
def load_db(filename, caching=True):
    print(f"Loading {filename}...")

    # Load the cache if enabled
    cachepath = Path(filename).with_suffix(".npy")
    if caching and cachepath.exists():
        print(f"Restoring cached decibels: {cachepath}")
        return np.load(cachepath.as_posix())

    # PYLOUDNORM
    # ----------------------------------------

    import soundfile as sf
    # import pyloudnorm as pyln

    y, sr = sf.read(filename)  # load audio (with shape (samples, channels))
    y = filter.butter(y, sr, 'highpass', 1, 400)

    meter = lufs_meter(sr, 1/constants.fps, overlap=0)
    loudness = meter.get_mlufs(y)
    # loudness = meter.integrated_loudness(y)  # measure loudness


    loudness[np.isinf(loudness)] = 0  # Replace infinities and nans with zero
    loudness = norm(loudness)  # Normalize to 0-1 in a 12 second window

    # ROSA
    # ----------------------------------------

    # y, sr = librosa.load(filename, sr=24)
    #
    # # Compute the spectrogram (magnitude)
    # n_fft = 2048
    # hop_length = 1024
    # spec_mag = abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    #
    # # Convert the spectrogram into dB
    #
    # # Compute A-weighting values
    # freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # a_weights = librosa.A_weighting(freqs)
    # a_weights = np.expand_dims(a_weights, axis=1)
    #
    # # Apply the A-weghting to the spectrogram in dB
    # spec_dba = spec_mag + a_weights
    #
    # # Compute the "loudness" value
    # loudness = librosa.feature.rms(S=spec_dba)

    # SHITTY
    # ----------------------------------------

    # # Convert samples to decibels
    # signal, sr = sfile.read(filename)
    # samples = audio.get_array_of_samples()
    # samples_sf = 0
    # try:
    #     samples_sf = signal[:, 0]  # use the first channel for dual
    # except:
    #     samples_sf = signal  # for mono
    #
    # print(f"Converting {filename} to decibels...")
    # # decibels = [convert_to_decibel(i) for i in samples_sf]  # idk how to vectorize this so it's a bit slow, I'm implementing caching for now
    # decibels = convert_to_decibel(samples_sf)


    # Write the cache if enabled
    if caching:
        np.save(cachepath.as_posix(), loudness)

    return loudness
    # return audio, decibels


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

        # remove infinities and nans
        if np.isinf(dt[i]) or np.isnan(dt[i]):
            dt[i] = dt[i - 1]

    return dt
    # return smooth_1euro(dt)


# @trace_decorator
# def convert_to_decibel(arr):
#     ref = 1
#     if arr != 0:
#         return 20 * np.log10(np.abs(arr) / ref)
#     else:
#         return -60

@trace_decorator
def convert_to_decibel(arr):
    ref = 1
    decibel = np.where(arr != 0, 20 * np.log10(np.abs(arr) / ref), -60)
    return decibel

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
