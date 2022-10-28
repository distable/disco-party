# Assign these
dbar = 0
dbeat = 0
offset = 0


def bar(n=1.0): return (n - 1) * dbar + (dbar - offset)


def beat(n=1.0): return (n - 1) * dbeat


def hbeat(n=1.0): return n * dbeat / 2


def coord(nbar, nbeat=1.0):
    return bar(nbar - 1) + beat(nbeat)
