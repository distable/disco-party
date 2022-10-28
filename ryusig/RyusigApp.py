from load import export_frames
from ryusig.RyuPlotWindow import RyuPlotWindow
from ryusig.Highlighter import Highlighter
from ryusig.RyuPlotWidget import RyuPlotWidget
from ryusig.RyuTextEdit import RyuTextEdit

pyexec = exec  # this gets replaced when you import * from pyqt

import datetime

from enum import Enum

import maths
from audio import *
from partyutils import *
from globals import *

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from pyqtgraph import *
import pyqtgraph as pg

pg.setConfigOptions(antialias=True, useOpenGL=False)

bar_width = 16
bar_height = 96

plot_colors = generate_colors(20)

SP_WAVS = '__wavs__'
SP_WAVNAMES = '__wavnames__'
SP_INIT_SIGNALS = ['chg', 'drum', 'bass', 'other']


# Plot -------------------------------------
# TODO text popup to jump/center to a specific X value
# Textbox ----------------------------------
# TODO support plotting static numbers by wrapping in ndarray and padding to framecount
# TODO select text to display subsection
# TODO grow selection to parentheses
# TODO jump through numbers
# TODO cycle numbers with ctrl-up/down


class TimeMode(Enum):
    Seconds = 0
    Frames = 1


class RyusigApp:
    def __init__(self, project_path, export_path, export_props):
        self.project_path = Path(os.getcwd()) / project_path
        self.export_path = export_path
        self.export_props = export_props

        self.mousepos = Point(0, 0)
        self.time_array = None
        self.max_nframes = None

        self.playback = None
        self.playback_wavs = []
        self.playback_wavnames = []
        self.playback_iwav = None
        self.playback_lasttime = 0
        self.playback_idx = 0
        self.playback_midx = 0
        self.playback_markers = []
        self.playback_startclock = datetime.datetime.now()
        self.playback_mousex = 0
        self.playback_signal = 0

        self.clines = []
        self.cnames = []
        self.csignals = []
        self.csignals_min = []
        self.csignals_max = []
        self.cbars = []

        self.time_mode = TimeMode.Seconds
        self.datadict = globals() | mod2dic(maths) | mod2dic(np)
        self.datakeys = globals().keys() | mod2dic(maths).keys() | mod2dic(np).keys()

        self.win = None
        self.winput = None

        self.exec_project(True)

        self.playback_wavs = self.datadict[SP_WAVS]
        self.playback_wavnames = self.datadict[SP_WAVNAMES]

        self.init_qapp()

        # self.eval_project(False)

    def init_qapp(self):
        # Main window
        pg.mkQApp("Ryusig")

        self.win = RyuPlotWindow()
        self.win.resize(1280, 720)

        vstack = QVBoxLayout()
        vstack.setSpacing(0)

        wmain = QWidget()
        wmain.setLayout(vstack)
        self.win.setCentralWidget(wmain)
        self.win.setContentsMargins(0, 0, 0, 0)
        self.win.setStyleSheet('background-color: black')
        vstack.setContentsMargins(0, 0, 0, 0)

        QShortcut(QtGui.QKeySequence("Ctrl+s"), wmain, self.on_shortcut_save)
        QShortcut(QtGui.QKeySequence("Ctrl+r"), wmain, self.on_shortcut_reload)

        # Evaluation Input
        # self.winput = QLineEdit()
        self.winput = RyuTextEdit()
        self.winput.setStyleSheet("background-color: #1b1818; color: #8a8585; font: 9pt 'Input'")
        self.winput.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.winput.completer_enabled = False
        self.winput.completer.qlist.setStringList(self.datakeys)
        self.winput.completer.setCompletionMode(QCompleter.InlineCompletion)

        # https://atelierbram.github.io/syntax-highlighting/atelier-schemes/plateau/
        self.highlighter = Highlighter()
        self.highlighter.addForeground(r'[\*\#\+\-\\/]', '#8a8585')  # Operators
        self.highlighter.addForeground(r"[\(\)\[\]]", '#8a8585')  # Symbols
        self.highlighter.addForeground(r"[+-]?[0-9]+[.][0-9]*([e][+-]?[0-9]+)?", '#b45a3c')  # Numbers
        self.highlighter.addForeground(";", '#7e7777')
        self.highlighter.functions = '#5485b6'
        self.highlighter.ndarrays = '#b45a3c'
        self.highlighter.addToDocument(self.winput.document())
        self.highlighter.datadict = self.datadict
        self.vplot = RyuPlotWidget()
        self.wplot = self.vplot.getPlotItem()
        self.wplot.showGrid(True, True, 0.75)
        self.vbplot = self.wplot.vb
        vstack.addWidget(self.winput)
        vstack.addWidget(self.vplot)

        # Main coord
        self.wcoord_main = QLabel()
        self.wcoord_main.setStyleSheet("background-color: none; color: white")
        self.wcoord_main.setParent(self.vplot)
        self.wcoord_main.setText("x=0\ny=0")
        self.wcoord_main.setGeometry(0, 0, 300, 200)
        self.wcoord_main.adjustSize()
        self.wcoord_main.setAlignment(QtCore.Qt.AlignTop)
        self.wcoord_main.show()
        self.wcoord_main.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        # Bar Signal
        wbars = QWidget(self.vplot)
        wbars.setStyleSheet("background-color: none")
        wbars.setFixedHeight(bar_height + 24)
        wbars.setFixedWidth(999)
        self.lbars = QHBoxLayout()
        self.lbars.setAlignment(QtCore.Qt.AlignHCenter)
        wbars.setLayout(self.lbars)
        wbars.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        wbars.show()
        wbars.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        QApplication.clipboard().dataChanged.connect(self.on_clipboard_changed)
        QApplication.clipboard().selectionChanged.connect(self.on_clipboard_sel_changed)

        def on_win_resize(ev):
            wbars.setFixedWidth(self.win.size().width())

            text_size = self.wcoord_main.fontMetrics().boundingRect(self.wcoord_main.text())
            self.wcoord_main.setGeometry(self.win.size().width() / 2 - text_size.width() / 2, bar_height + 24, text_size.width() + 4, text_size.height() * 2 + 4)
            # self.wcoord_main.adjustSize()

        # t-line
        self.tline = InfiniteLine(5, pen=self.mkTlinePen())
        self.wplot.vb.addItem(self.tline, ignoreBounds=True)
        # Ploot coord
        self.wcoord = TextItem('y=0')
        # self.wtargetu = TargetItem(movable=False, size=10)
        # self.wtargetu.setPen(mkPen(width=2, color='black'))
        # self.wtargetu.hide()
        self.wtarget = TargetItem(movable=False, size=7)
        self.wtarget.hide()
        self.wcoord.hide()
        self.vbplot.addItem(self.wcoord, ignoreBounds=True)
        # self.vbplot.addItem(self.wtargetu, ignoreBounds=True)
        self.vbplot.addItem(self.wtarget, ignoreBounds=True)
        # Events
        self.winput.textChanged.connect(self.on_text_changed)
        self.vplot.sigKeyPress.connect(self.on_plot_key_pressed)
        self.vplot.sceneObj.sigMouseMoved.connect(self.on_plot_mouse_move)
        self.vplot.sceneObj.sigMouseClicked.connect(self.on_plot_mouse_clicked)
        self.win.sigResize.connect(on_win_resize)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_timer_timeout)
        self.timer.start(int(1000 / 60))  # 60 fps timer

        self.winput.setText(' '.join([varname for varname in SP_INIT_SIGNALS if varname in self.datadict]))
        # self.update_plot_data(signals)

        # Setup the main window
        self.win.show()
        # self.vplot.scale()
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()

    def on_clipboard_sel_changed(self):
        if QApplication.focusWindow() is not None: return
        txt = QApplication.clipboard().text(QtGui.QClipboard.Mode.Selection)
        try:
            eval(txt, self.datadict)
            self.winput.setText(txt)
        except:
            pass

    def on_clipboard_changed(self):
        if QApplication.focusWindow() is not None: return
        txt = QApplication.clipboard().text(QtGui.QClipboard.Mode.Clipboard)
        try:
            eval(txt, self.datadict)
            self.winput.setText(txt)
        except:
            pass

    def to_seconds(self, frame):
        return np.clip(frame, 0, self.max_nframes - 1) / fps

    def to_frame(self, t):
        return int(np.clip(t * fps, 0, self.max_nframes - 1))

    def create_bar(self, color):
        b = QWidget()
        b.setFixedSize(bar_width, bar_height)
        b.setStyleSheet(f'background-color: {color};')
        b.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        return b

    def update_max_nframes(self, *signals):
        self.max_nframes = max(np.array([x.shape[0] for x in signals]))
        pass

    def update_time_frames(self):
        n = self.max_nframes - 1
        if self.time_mode == TimeMode.Seconds:
            n /= fps

        self.time_array = np.linspace(0, n, self.max_nframes)
        pass

    def find_nearest_signal(self, signals, x, y):
        f = self.x_to_frame(x)
        idx = -1
        mindist = sys.maxsize
        for i, s in enumerate(reversed(signals)):
            v = s[f]

            d = abs(v - y)
            if d < mindist:
                idx = i
                mindist = d

        return len(signals) - idx - 1

    def update_plot_data(self, ss):
        self.update_max_nframes(*ss)
        self.update_time_frames()

        # Add new ones
        self.wplot.clear()
        self.csignals.clear()
        self.csignals_min.clear()
        self.csignals_max.clear()
        self.clines.clear()

        for b in self.cbars:
            self.lbars.removeWidget(b)
        self.cbars.clear()

        # self.lbars.addStretch(1)

        for i, s in enumerate(ss):
            s = np.nan_to_num(s)
            s = np.pad(s, (0, self.time_array.shape[0] - s.shape[0]), 'edge')

            c = plot_colors[i % len(ss)]
            l = self.wplot.plot(self.time_array, s, label=f"Signal {i}", pen=pg.mkPen(c, width=1.5))
            b = self.create_bar(c)

            self.lbars.addWidget(b, alignment=QtCore.Qt.AlignBottom)

            self.cbars.append(b)
            self.csignals.append(s)
            self.csignals_min.append(np.min(s))
            self.csignals_max.append(np.max(s))
            self.clines.append(l)

        # self.lbars.addStretch(1)
        self.playback_signal = np.clip(self.playback_signal, 0, len(ss) - 1)

        self.vbplot.removeItem(self.wtarget)
        self.vbplot.addItem(self.wtarget, ignoreBounds=True)

    def set_mouse_pos(self, x, y, tracked_signal=None):
        x = self.clip_x(x)
        x = self.snap_x_to_frame(x)
        f = self.x_to_frame(x)
        t = self.x_to_seconds(x)

        self.playback_mousex = t
        self.tline.setPos(x)

        # Update the label and target
        inear = tracked_signal
        if inear is None:
            inear = self.find_nearest_signal(self.csignals, x, y)

        if inear > -1:
            s = self.csignals[inear]
            v = s[f]

            self.wcoord.show()
            self.wcoord.setText(f"y={v:.2f}")
            self.wcoord.setPos(x, v)

            self.wcoord_main.show()
            if self.time_mode == TimeMode.Seconds:
                self.wcoord_main.setText(f"x={x:.2f}\ny={v:.2f}")
            elif self.time_mode == TimeMode.Frames:
                self.wcoord_main.setText(f"x={x:.0f}\ny={v:.2f}")

            # self.wtargetu.setPen(mkPen(color=plot_colors[inear]))
            # self.wtargetu.setPos(t, v)
            # self.wtargetu.show()
            # self.wtarget.setPen(mkPen(color='white', brush=mkBrush(color='black')))
            self.wtarget.setPos(x, v)
            self.wtarget.show()

            self.playback_signal = inear

        # Update the bars
        for i, (bar, s) in enumerate(zip(self.cbars, self.csignals)):
            v = s[f]
            lo = self.csignals_min[i]
            hi = self.csignals_max[i]
            vt = ilerp(lo, hi, v)

            bar.setMinimumSize(bar_width, bar_height * vt)

        # self.lbars.parentWidget().adjustSize()
        # self.lbars.parentWidget().repaint()

    def x_to_seconds(self, x):
        if self.time_mode == TimeMode.Seconds:
            return x
        elif self.time_mode == TimeMode.Frames:
            return x / fps

    def x_to_frame(self, x):
        f = 0
        if self.time_mode == TimeMode.Seconds:
            f = int(x * fps)
        elif self.time_mode == TimeMode.Frames:
            f = int(x)

        return clamp(f, 0, self.max_nframes - 1)

    def seconds_to_x(self, t):
        if self.time_mode == TimeMode.Seconds:
            return t
        elif self.time_mode == TimeMode.Frames:
            return int(t * fps)

    def clip_x(self, x):
        return np.clip(x, 0, self.time_array[-1])

    def snap_x_to_frame(self, x):
        if self.time_mode == TimeMode.Seconds:
            f = int(x * fps)
            return self.to_seconds(f)
        else:
            return int(x)

    def stop_playback(self):
        if self.is_playing():
            self.playback.stop()
            self.playback = None
            self.tline.setPos(self.playback_lasttime)
            self.tline.setPen(self.mkTlinePen())

    def play_marker(self):
        if self.playback_markers is None or not self.playback_markers:
            print("No playback markers to play.")
            return

        threshold = 0.1
        self.stop_playback()

        self.playback_midx = np.clip(self.playback_midx, 0, len(self.playback_markers) - 1)
        markers = self.playback_markers[self.playback_midx]

        playback_idx = np.clip(self.playback_idx, 0, len(markers[markers > threshold]) - 1)

        spent = 0
        for i in range(markers.shape[0]):
            v = markers[i]
            if v > threshold:
                spent += 1
                if spent == playback_idx + 1:
                    print(self.playback_midx, playback_idx, i / fps)
                    self.start_playback(i / fps)
                    break

    def start_playback(self, t):
        if t is None: return
        if not self.playback_wavs: return

        iwav = self.playback_iwav
        if iwav is None:
            def get_wavname_index(cname):
                for i, n in enumerate(self.playback_wavnames):
                    if n in cname:
                        return i

            if len(self.cnames) == 1:
                # Auto match to the only cname
                iwav = get_wavname_index(self.cnames[0])
            else:
                # Auto-match from mouse
                i = self.playback_signal
                if i in range(len(self.cnames)):  # Should always be true
                    iwav = get_wavname_index(self.cnames[i])

        if iwav is None:
            iwav = 0

        # global playback, playback_lasttime, playback_startclock
        wav = self.playback_wavs[iwav]
        self.playback = play_wav(wav.get_sample_slice(int(t * wav.frame_rate)))
        self.playback_startclock = datetime.datetime.now()
        self.playback_lasttime = t
        self.tline.setPos(self.playback_lasttime)
        self.tline.setPen(self.mkTlinePen())

    def is_playing(self):
        return self.playback is not None

    def on_plot_key_pressed(self, ev):
        # Proxy to text input
        def init_iwav():
            self.playback_iwav = self.playback_iwav if self.playback_iwav is not None else 0

        if ev.key() == QtCore.Qt.Key.Key_Left:
            init_iwav()
            self.playback_iwav -= 1
            self.play_marker()
        elif ev.key() == QtCore.Qt.Key.Key_Right:
            init_iwav()
            self.playback_iwav += 1
            self.play_marker()
        elif ev.key() == QtCore.Qt.Key.Key_Up:
            init_iwav()
            self.playback_iwav = np.clip(self.playback_iwav - 1, 0, len(self.playback_wavs) - 1)
            self.refresh_playback()
            # self.playback_midx -= 1
            # self.play_marker()
        elif ev.key() == QtCore.Qt.Key.Key_Down:
            init_iwav()
            self.playback_iwav = np.clip(self.playback_iwav + 1, 0, len(self.playback_wavs) - 1)
            self.refresh_playback()
            # self.playback_midx += 1
            # self.play_marker()
        elif ev.key() == QtCore.Qt.Key.Key_F1:
            self.time_mode = TimeMode((self.time_mode.value + 1) % len(TimeMode))
            self.update_plot_data([*self.csignals])
        elif ev.key() == QtCore.Qt.Key.Key_Space:
            if self.playback is not None:
                self.stop_playback()
            else:
                self.start_playback(self.playback_mousex)

            self.set_mouse_pos(self.mousepos.x(), self.mousepos.y())
        else:
            self.winput.setFocus()
            self.winput.keyPressEvent(ev)

    def on_plot_mouse_move(self, ev):
        mp = self.vbplot.mapSceneToView(ev)
        x = mp.x()
        y = mp.y()
        self.mousepos = mp

        if ev is not None and not self.is_playing():
            self.set_mouse_pos(x, y)

    def on_text_changed(self):
        text = self.winput.toPlainText()

        # Try eval mode
        strings = []
        ss = []
        formulas = text.split(';')
        try:
            for f in formulas:
                self.datadict['t'] = self.time_array
                v = eval(f, self.datadict)
                if isinstance(v, np.ndarray):
                    strings.append(f)
                    ss.append(v)

        except:
            pass

        # Try list mode
        if not len(ss):
            text = text.replace(',', ' ')
            words = text.split(' ')
            ss = [self.datadict[w] for w in words if w in self.datadict and isinstance(self.datadict[w], np.ndarray)]
            strings.extend(words)

        if not len(ss):
            return

        self.playback_iwav = None
        self.cnames = strings
        self.update_plot_data(ss)

    def on_plot_mouse_clicked(self, ev):
        if ev.button() == 4:
            self.wplot.autoRange()

    def on_timer_timeout(self):
        if self.winput.alignment() != QtCore.Qt.AlignmentFlag.AlignCenter:
            self.winput.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        if self.is_playing():
            t = self.get_playback_t()
            x = self.seconds_to_x(t)
            self.set_mouse_pos(x, 0, tracked_signal=self.playback_signal)

    def get_playback_t(self):
        elapsed = (datetime.datetime.now() - self.playback_startclock).total_seconds()
        t = self.playback_lasttime + elapsed
        return t

    def refresh_playback(self):
        if self.is_playing():
            self.stop_playback()
            self.start_playback(self.get_playback_t())

    def mkTlinePen(self):
        if self.is_playing():
            return mkPen('white', width=1, style=QtCore.Qt.PenStyle.SolidLine)
        else:
            return mkPen('white', width=1, style=QtCore.Qt.PenStyle.DotLine)

    def on_shortcut_save(self):
        self.export_project()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Successfully exported!\n\n{len(self.export_props)} tracks\n{self.max_nframes / fps:.2f} sec\n{self.max_nframes} frames\n{fps} fps")
        msg.setWindowTitle("Export")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def on_shortcut_reload(self):
        self.exec_project()
        self.winput.setText(self.winput.toPlainText())

    def exec_project(self, init=False):
        with open(self.project_path) as f:
            s = f.read()
            self.datadict['__ryuinit__'] = init
            pyexec(s, self.datadict)

    def export_project(self):
        export_frames(self.export_path, self.datadict, *self.export_props)

# def on_tb_key(ev):
#     if ev.key == 'escape':
#         tb.set_val("")
#     if ev.key == 'ctrl+v':
#         tb.set_val(pyclip.paste(text=True))
#         tb.cursor_index = len(tb.text)
#         tb._rendercursor()
#     if ev.key == 'ctrl+c':
#         pyclip.copy(tb.text)
#     if ev.key == 'ctrl+delete':
#         txt = tb.text
#         if len(txt) == 0: return
#         if tb.cursor_index == 0: return
#
#         hi, lo = find_text_stops(txt, tb.cursor_index, -1)
#         txt = txt[0:lo:] + txt[hi:]
#
#         tb.set_val(txt)
#         tb.cursor_index = lo
#         tb._rendercursor()
#     if ev.key == 'ctrl+backspace':
#         txt = tb.text
#         if len(txt) == 0: return
#         if tb.cursor_index == 0: return
#
#         hi, lo = find_text_stops(txt, tb.cursor_index, -1)
#         txt = txt[0:lo:] + txt[hi:]
#
#         tb.set_val(txt)
#         tb.cursor_index = lo
#         tb._rendercursor()


# def find_text_stops(txt, index, dir=1):
#     lo = 0
#     hi = tb.cursor_index
#
#     r = None
#     if dir == -1:
#         r = range(hi - 2, -1, -1)
#     else:
#         r = range(hi, len(txt), dir)
#
#     for i in r:
#         c = txt[i]
#         if c == ' ': continue
#         if c in ' ,./|\\#@!$%^&*()[]<>-+=\'\"!?':
#             # if c == ' ' and i > 0 and txt[i - 1] == ' ':
#             #     continue
#             lo = i + 1
#             break
#     if lo == hi:
#         lo -= 1
#     return hi, lo


# def plot_c(x, y, z):
#     colorline(x, y, z, cmap="magma")
#     plt.xlim(x.min(), x.max())
#     plt.ylim(y.min(), y.max())
#     plt.show(block=True)
#
#
# def plot_s(*signals, color=None, on_press=None):
#     fig = plt.figure()
#     fig.canvas.mpl_connect('button_press_event', on_press)
#
#     plt.subplot(1, 1, 1)
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Signal')
#
#     t = get_indices_seconds(signals)
#     for s in signals:
#         plt.plot(t, np.pad(s, t.shape[0] - s.shape[0], 'edge'))
#
#     plt.show(block=True)
#
#
# def plot_mmotifs(signal, results, on_press=None):
#     # motif_idx = np.argsort(mp[:, 0])[0]
#     # nearest_neighbor_idx = mp[motif_idx, 1]
#
#     fig = plt.figure()
#
#     ax = plt.subplot(1, 1, 1)
#     plt.title('Pattern Mining')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Signal')
#     plt.plot(get_indices_seconds(signal), signal)
#
#     smax = np.max(signal)
#     smin = np.min(signal)
#     motif_height = (smax - smin) / len(results)
#
#     for ires, res in enumerate(results):
#         m = res['m']
#         midx = res['idx']
#         mdist = res['dist']
#         for iidx in range(midx.shape[1]):
#             idx = midx[0][iidx]
#             dist = mdist[0][iidx]
#             distnorm = (dist / np.max(mdist[0]))
#             rect = Rectangle(
#                     (idx / fps, smin - ires * motif_height - motif_height - ires * motif_height * 0.2),
#                     m / fps,
#                     motif_height,
#                     facecolor=(0, 1 - distnorm, 0, 0.1),
#                     edgecolor='black',
#                     linewidth=0.1
#             )
#             ax.add_patch(rect)
#
#     fig.canvas.mpl_connect('button_press_event', on_press)
#     plt.show(block=True)
#
#
# def plot_motifs(signal, midx, mdist, m, on_press=None):
#     # motif_idx = np.argsort(mp[:, 0])[0]
#     # nearest_neighbor_idx = mp[motif_idx, 1]
#
#     fig = plt.figure()
#
#     ax = plt.subplot(1, 1, 1)
#     plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
#
#     ax.plot(get_indices_seconds(signal), signal)
#     ax.set_ylabel('Signal', fontsize='20')
#
#     signalmax = np.max(signal)
#
#     for i in range(midx.shape[1]):
#         idx = midx[0][i]
#         dist = mdist[0][i]
#         distnorm = (dist / np.max(mdist[0]))
#         # print(idx)
#
#         ax.add_patch(Rectangle((idx / fps, 0), m / fps, signalmax, facecolor=(0, 1 - distnorm, 0, 0.1)))
#         # axs[0].add_patch(Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey'))
#
#     # axs[1].set_xlabel('Time (seconds)', fontsize ='20')
#     # axs[1].set_ylabel('Matrix Profile', fontsize='20')
#     # axs[1].axvline(x=motif_idx, linestyle="dashed")
#     # axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
#     # axs[1].plot(mp[:, 0])
#
#     fig.canvas.mpl_connect('button_press_event', on_press)
#     plt.show(block=True)


# def plot_crepe():
#     pass
#     # import matplotlib.cm
#     # from imageio import imwrite
#     #
#     # plot_file = output_path(file, ".activation.png", output)
#     # # to draw the low pitches in the bottom
#     # salience = np.flip(activation, axis=1)
#     # inferno = matplotlib.cm.get_cmap('inferno')
#     # image = inferno(salience.transpose())
#     #
#     # if plot_voicing:
#     #     # attach a soft and hard voicing detection result under the
#     #     # salience plot
#     #     image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
#     #     image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
#     #     image[-10:, :, :] = (
#     #         inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :])
#     #
#     # imwrite(plot_file, (255 * image).astype(np.uint8))
#     # if verbose:
#     #     print("CREPE: Saved the salience plot at {}".format(plot_file))


# def plot_motifs(signal, mp, m):
#     motif_idx = np.argsort(mp[:, 0])[0]
#     nearest_neighbor_idx = mp[motif_idx, 1]
#
#     fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
#     plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
#
#     axs[0].plot(signal)
#     axs[0].set_ylabel('Signal', fontsize='20')
#     axs[0].add_patch(Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey'))
#     axs[0].add_patch(Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey'))
#
#     axs[1].set_xlabel('Time (seconds)', fontsize ='20')
#     axs[1].set_ylabel('Matrix Profile', fontsize='20')
#     axs[1].axvline(x=motif_idx, linestyle="dashed")
#     axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
#     axs[1].plot(mp[:, 0])
#
#     plt.show(block=True)
