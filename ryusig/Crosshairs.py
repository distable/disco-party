import pyqtgraph as pg
from PyQt5 import QtCore
from pyqtgraph.Qt import QtCore


class Crosshairs(QtCore.QObject):
    """ Attaches crosshairs to the a plot and provides a signal with the
    x and y graph coordinates
    """

    coordinates = QtCore.Signal(float, float)

    def __init__(self, plot, pen=None):
        """ Initiates the crosshars onto a plot given the pen style.

        Example pen:
        pen=pg.mkPen(color='#AAAAAA', style=QtCore.Qt.DashLine)
        """
        QtCore.QObject.__init__(self)
        self.vertical = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.horizontal = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        plot.vb.addItem(self.vertical, ignoreBounds=True)
        plot.vb.addItem(self.horizontal, ignoreBounds=True)

        self.position = None
        self.proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.plot = plot

    def hide(self):
        self.vertical.hide()
        self.horizontal.hide()

    def show(self):
        self.vertical.show()
        self.horizontal.show()

    def update(self):
        """ Updates the mouse position based on the data in the plot. For
        dynamic plots, this is called each time the data changes to ensure
        the x and y values correspond to those on the display.
        """
        if self.position is not None:
            mousePoint = self.plot.vb.mapSceneToView(self.position)
            self.coordinates.emit(mousePoint.x(), mousePoint.y())
            self.vertical.setPos(mousePoint.x())
            self.horizontal.setPos(mousePoint.y())

    def mouseMoved(self, event=None):
        """ Updates the mouse position upon mouse movement """
        if event is not None:
            self.position = event[0]
            self.update()
        else:
            raise Exception("Mouse location not known")