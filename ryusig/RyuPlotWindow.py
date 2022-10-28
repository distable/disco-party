from PyQt5.QtWidgets import QMainWindow
from pyqtgraph.Qt import QtCore


class RyuPlotWindow(QMainWindow):
    sigResize = QtCore.Signal(object)
    sigKeyPress = QtCore.Signal(object)

    def __init__(self):
        super(RyuPlotWindow, self).__init__()

    def resizeEvent(self, ev):
        super(RyuPlotWindow, self).resizeEvent(ev)
        self.sigResize.emit(ev)

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)