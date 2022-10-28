import re
import time
from colorsys import *

from globals import *


class Timer:
    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start


def kprint(*kargs):
    s = ""
    for v in kargs:
        s += f"{v}  "

    print(s)


def kprint(*kargs):
    s = ""
    for v in kargs:
        s += f"{v}  "

    print(s)


def kwprint(**kwargs):
    s = ""
    for k, v in kwargs.items():
        # idk why we have to check for bools or if its even required, not gonna question it I have better things to do like actually getting stuff done
        if isinstance(v, (float, complex)) and not isinstance(v, bool):
            s += f"{k}={v:.2f}  "
        elif isinstance(v, int) and not isinstance(v, bool):
            s += f"{k}={v}  "
        else:
            s += f"{k}={v}  "

    print(s)


def flat(pool):
    res = []
    for v in pool:
        if isinstance(v, list):
            res += flat(v)
        else:
            if isinstance(v, int):
                res.append(v)
    return res


def mod2dic(module):
    return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}


rgb_to_hex = lambda tuple: f"#{int(tuple[0] * 255):02x}{int(tuple[1] * 255):02x}{int(tuple[2] * 255):02x}"
hex_to_rgb = lambda hx: (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


def generate_colors(n):
    golden_ratio_conjugate = 0.618033988749895
    h = 0
    ret = []
    for i in range(n):
        h += golden_ratio_conjugate
        ret.append(rgb_to_hex(hsv_to_rgb(h, 0.825, 0.915)))

    return ret


import numpy as np
import matplotlib.pyplot as plt


class InteractiveLegend(object):
    def __init__(self, legend=None):
        if legend == None:
            legend = plt.gca().get_legend()
        self.legend = legend
        self.fig = legend.axes.figure
        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()
        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10)  # 10 points tolerance
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))
        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist
        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))
        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return
        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()


def split_and_keep(seperator, s):
    return re.split(';', re.sub(seperator, lambda match: match.group() + ';', s))
