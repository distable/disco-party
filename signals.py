import json
from typing import Union

import numpy as np
from bunch import Bunch

from collections.abc import Iterable

# Assign these
from maths import *
from globals import *
import mido


def load_pose_frames(path, joint: Union[int, tuple], original_fps=29.89):
    # Average out multiple keypoints for overall movement
    if isinstance(joint, tuple):
        xs = None
        ys = None
        count = len(joint)
        for j in joint:
            x, y = load_pose_frames(path, j)
            if xs is None:
                xs = x
                ys = y
            else:
                xs += x
                ys += y

        return xs / count, ys / count

    with open(path, 'r') as file:
        o = json.load(file)

    framecount = len(o)
    stretchcount = int(framecount / original_fps * fps)

    x = np.zeros(framecount)
    y = np.zeros(framecount)

    for i in range(framecount):
        kp = o[i]['keypoints'][joint]
        x[i] = kp[0]
        y[i] = kp[1]

    x = stretch(x, stretchcount)
    y = stretch(y, stretchcount)

    return x, y


def load_dpose_frames(path, joint, original_fps=29.89):
    x, y = load_pose_frames(path, joint)
    return np.diff(x), np.diff(y)


def load_dheadrot_frames(path, original_fps=29.89):
    rx, ry, rz = load_headrot_frames(path, original_fps)
    return np.diff(rx), np.diff(ry), np.diff(rz)


def load_headrot_frames(path, original_fps=29.89):
    with open(path, 'r') as file:
        o = json.load(file)

    framecount = len(o)
    stretchcount = int(framecount / original_fps * fps)

    rx = np.zeros(framecount)
    ry = np.zeros(framecount)
    rz = np.zeros(framecount)

    for i in range(framecount):
        frame = o[i]
        rx[i] = frame[0]
        ry[i] = frame[1]
        rz[i] = frame[2]

    # rx = wavg(rx, 0.02)
    # ry = wavg(ry, 0.02)
    # rz = wavg(rz, 0.02)

    rx -= rx[0]
    ry -= ry[0]
    rz -= rz[0]

    rx = stretch(rx, stretchcount)
    ry = stretch(ry, stretchcount)
    rz = stretch(rz, stretchcount)

    return rx, ry, rz


# think I'm gonna use head pose estimation instead... it falls apart past 45deg but at least it's darn accurate under that
# def load_rxface_frames(path):
#     lx,ly = load_pose_frames(path, 1)
#     rx,ry = load_pose_frames(path, 2)
#
#     # zero = rx[0] - lx[0]
#     zero = 0
#
#     return (wavg(rx - lx - zero, 0.035))
#
# def load_rzface_frames(path):
#     lx,ly = load_pose_frames(path, 1)
#     rx,ry = load_pose_frames(path, 2)
#
#     xdist = rx - lx
#     ydist = ry - ly
#     angle = np.arctan(ydist / xdist)
#     degree = angle * 180 / np.pi
#
#     return (wavg(degree, 0.035))

def load_midibucket_frames(path, as_numpy=False):
    mid = mido.MidiFile(path)

    # Gather information
    duration = 0
    notes = set()

    for msg in mid:
        duration += msg.time
        if msg.type == 'note_on' or msg.type == 'note_off':
            notes.add(msg.note)

    # Build the frame data
    framecount = int(duration * fps)
    frames = dict()
    lasttimes = dict()
    for n in notes:
        lasttimes[n] = 0
        frames[n] = np.zeros(framecount)

    time = 0
    timef = 0

    def write(note, v):
        start = lasttimes[n]
        elapsed = timef - lasttimes[n]
        for i in range(elapsed):
            frames[note][start + i] = v
            lasttimes[n] = timef

    for msg in mid:
        time += msg.time
        timef = int(time * fps)

        if msg.type == 'note_on':
            notes.add(msg.note)
            write(msg.note, 0)
        elif msg.type == 'note_off':
            notes.remove(msg.note)
            write(msg.note, 1)

    ret = [x for x in frames.values()]
    if as_numpy:
        return np.vstack(ret)

    return ret


def load_midi_frames(path):
    mid = mido.MidiFile(path)

    # Gather information
    duration = 0
    for msg in mid:
        duration += msg.time

    # Build the frame data
    frames = np.zeros(int(duration * fps))
    time = 0
    lastf = 0
    lastn = 0
    for msg in mid:
        time += msg.time
        timef = int(time * fps)

        if msg.type == 'note_on':
            for i in range(lastf, timef):
                frames[i] = lastn

            lastf = timef
            lastn = msg.note

    return frames


def make_temporal_markers(like, measure_duration, start=0.0, end=np.inf):
    s = np.zeros_like(like)
    for ts in np.arange(start * fps, np.min((end, s.shape[0])), measure_duration * fps):
        s[int(ts)] = 1
    return s


def make_msec(like, sections, filter=None, step=None, coded=True, sustain=False):
    s = np.zeros_like(like)
    labels = list(set([x[0] for x in sections]))

    for isec in range(len(sections) - 1):  # we expect an end marker at the very end
        now = sections[isec]
        next = sections[isec + 1]
        label, time = now[0], now[1]
        label1, time1 = next[0], next[1]

        if filter is None or label == filter:
            code = 1
            if coded:
                code += labels.index(label)

            if step is None:
                if not sustain:
                    s[int(time * fps)] = code
                else:
                    s[int(time * fps):int(time1 * fps)] = code
            else:
                i = 0
                t = time
                while t < time1:
                    s[int(t * fps)] = code
                    if isinstance(step, float) or isinstance(step, int):
                        t += step
                    elif isinstance(step, tuple) or isinstance(step, list):
                        t += step[i % len(step)]
                i += 1
    return s


def sectionate(sections, *values):
    ret = np.copy(values[0][1])
    for vpair in values:
        filter = vpair[0]
        v = vpair[1]
        for isec in range(len(sections) - 1):  # we expect an end marker at the very end
            now = sections[isec]
            next = sections[isec + 1]
            label, time = now[0], now[1]
            label1, time1 = next[0], next[1]

            if label.startswith(filter):
                lo = int(fps * time)
                hi = int(fps * time1)
                if isinstance(v, Iterable):
                    ret[lo:hi] = v[lo:hi]
                elif isinstance(v, float) or isinstance(v, int):
                    ret[lo:hi] = v

    return ret


def make_linsec01(like, sections, filter=None, step=None):
    return make_section_lin(like, sections, filter, step)


def make_linsec10(like, sections, filter=None, step=None):
    return make_section_lin(like, sections, filter, step, lambda v: 1 - v)


def make_linsec010(like, sections, filter=None, step=None):
    return make_section_lin(like, sections, filter, step, lambda v: 1 - (np.abs(v - 0.5) * 2))


def make_linsec101(like, sections, filter=None, step=None):
    return make_section_lin(like, sections, filter, step, lambda v: np.abs(v - 0.5) * 2)


def make_section_lin(like, sections, filter=None, step=None, processor=None):
    s = np.zeros_like(like)
    for isec in range(len(sections) - 1):  # we expect an end marker at the very end
        now = sections[isec]
        next = sections[isec + 1]

        label, time = now[0], now[1]
        label1, time1 = next[0], next[1]

        if filter is None or label == filter:
            if step is None:
                s[int(time * fps)] = 1
            else:
                i = 0
                t = time
                l = time
                while t < time1:
                    if isinstance(step, float) or isinstance(step, int):
                        t += step
                    elif isinstance(step, tuple) or isinstance(step, list):
                        t += step[i % len(step)]

                    v0 = 0
                    v1 = 1
                    if t > time1:
                        d = (t - time1) / step
                        v1 -= d
                        t = np.clip(t, time, time1)

                    lo = int(l * fps)
                    hi = int(t * fps)

                    l = np.round(hi - lo)
                    lin = np.linspace(v0, v1, int(l))
                    if processor is not None:
                        lin = processor(lin)

                    s[lo:hi] = lin
                    l = t
                    i += 1
    return s
