import re

import numpy as np
from PyQt5 import QtCore, QtGui
from pyqtgraph.Qt import QtCore


## Highligher Class written by igor-bogomolov
class Highlighter(QtCore.QObject):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.datadict = dict()
        self.mappings = {}
        self.ndarrays = 'white'
        self.functions = 'white'

    def addToDocument(self, doc):
        doc.contentsChange.connect(self.highlight)
        # self.connect(doc, QtCore.Signal('contentsChange(int, int, int)'), self.highlight)

    def addForeground(self, pattern, color):
        self.addMapping(pattern, self.mkForeground(color))

    def mkForeground(self, color):
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(color))
        return fmt

    def addMapping(self, pattern, format):
        self.mappings[pattern] = format

    def highlight(self, position, removed, added):
        doc = self.sender()

        block = doc.findBlock(position)
        if not block.isValid():
            return

        if added > removed:
            endBlock = doc.findBlock(position + added)
        else:
            endBlock = block

        while block.isValid() and not (endBlock < block):
            self.highlightBlock(block)
            block = block.next()

    def highlightBlock(self, block):
        layout = block.layout()
        text = block.text()

        overrides = []

        def highlight(m, format):
            range = QtGui.QTextLayout.FormatRange()
            s, e = m.span()
            range.start = s
            range.length = e - s
            range.format = format
            return range

        for pattern in self.mappings:
            for m in re.finditer(pattern, text):
                overrides.append(highlight(m, self.mappings[pattern]))

        for m in re.finditer(r'\b[\w_][\w\d_u]+\b', text):
            mtxt = m.group(0)
            o = None
            if mtxt in self.datadict:
                symbol = self.datadict[mtxt]
                if callable(symbol):
                    o = highlight(m, self.mkForeground(self.functions))  # Functions
                elif isinstance(symbol, np.ndarray):
                    o = highlight(m, self.mkForeground(self.ndarrays))  # ndarrays

            if o is not None:
                overrides.append(o)

        layout.setAdditionalFormats(overrides)
        block.document().markContentsDirty(block.position(), block.length())