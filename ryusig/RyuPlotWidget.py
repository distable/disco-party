from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtCore


class RyuPlotWidget(PlotWidget):
    sigKeyPress = QtCore.Signal(object)

    def __init__(self):
        super(RyuPlotWidget, self).__init__()

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)