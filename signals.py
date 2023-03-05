import json
import os
from typing import Union
import mido

from collections.abc import Iterable

import resampy
from scipy.signal import decimate

from . import constants
# Assign these
from .maths import *
# from .constants import *


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
    stretchcount = int(framecount / original_fps * constants.fps)

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
    stretchcount = int(framecount / original_fps * constants.fps)

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


class VRRecording:
    def __init__(self, samples_per_second):
        self.samples_per_second = samples_per_second
        self.inputs = []
        self.nodes = {}

    def __getattr__(self, name):
        if name in self.nodes:
            return self.nodes[name]
        else:
            return getattr(self, name)


class VRInput:
    def __init__(self, input, modality, d1, d2, d3):
        self.input = input
        self.modality = modality
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.dd1 = pdiff(d1)
        self.dd2 = pdiff(d2)
        self.dd3 = pdiff(d3)


VRNodes = {
    'None'   : 0,
    'A'      : 1,
    'B'      : 2,
    'X'      : 3,
    'Y'      : 4,
    'LMenu'  : 5,
    'RMenu'  : 6,
    'LStick' : 7,
    'RStick' : 8,
    'LGrip'  : 9,
    'RGrip'  : 10,
    'LTrig'  : 11,
    'RTrig'  : 12,
    'LPos'   : 13,
    'RPos'   : 14,
    'LRot'   : 15,
    'RRot'   : 16,
    'HeadPos': 17,
    'HeadRot': 18
}

VRNodes_ID = dict(reversed(it) for it in VRNodes.items())

def load_vr(path):
    if not os.path.exists(path):
        raise Exception(f"VR recording does not exist at {path}")

    with open(path, 'r') as file:
        o = json.load(file)
        rec = VRRecording(o['SamplesPerSecond'])

        inputs = o['Inputs']
        for i in range(len(inputs)):
            rec_input = inputs[i]

            itype = int(rec_input['Type'])
            modality = int(rec_input['Modality'])
            d1 = np.array(rec_input['D1'])
            d2 = np.array(rec_input['D2'])
            d3 = np.array(rec_input['D3'])

            if d2.shape[0] == 0: d2 = np.zeros(d1.shape[0])
            if d3.shape[0] == 0: d3 = np.zeros(d1.shape[0])

            # Stretch d2 and d3 to match d1 in case they're different lengths
            d1 = resampy.resample(d1, rec.samples_per_second, constants.fps)
            d2 = resampy.resample(d2, rec.samples_per_second, constants.fps)
            d3 = resampy.resample(d3, rec.samples_per_second, constants.fps)

            vrinput = VRInput(VRNodes_ID[itype], modality, d1, d2, d3)

            rec.inputs.append(vrinput)
            rec.nodes[VRNodes_ID[itype]] = vrinput

        return rec


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
    framecount = int(duration * constants.fps)
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
        timef = int(time * constants.fps)

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
    print(path)
    mid = mido.MidiFile(path)

    # Gather information
    duration = 0
    for msg in mid:
        duration += msg.time

    # Build the frame data
    frames = np.zeros(int(duration * constants.fps))
    time = 0
    lastf = 0
    lastn = 0
    for msg in mid:
        time += msg.time
        timef = int(time * constants.fps)

        if msg.type == 'note_on':
            for i in range(lastf, timef):
                frames[i] = lastn

            lastf = timef
            lastn = msg.note

    return frames


def make_temporal_markers(like, measure_duration, start=0.0, end=np.inf):
    s = np.zeros_like(like)
    for ts in np.arange(start * constants.fps, np.min((end, s.shape[0])), measure_duration * constants.fps):
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
                    s[int(time * constants.fps)] = code
                else:
                    s[int(time * constants.fps):int(time1 * constants.fps)] = code
            else:
                i = 0
                t = time
                while t < time1:
                    s[int(t * constants.fps)] = code
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
                lo = int(constants.fps * time)
                hi = int(constants.fps * time1)
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
                s[int(time * constants.fps)] = 1
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

                    lo = int(l * constants.fps)
                    hi = int(t * constants.fps)

                    l = np.round(hi - lo)
                    lin = np.linspace(v0, v1, int(l))
                    if processor is not None:
                        lin = processor(lin)

                    s[lo:hi] = lin
                    l = t
                    i += 1
    return s
