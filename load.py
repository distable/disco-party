import json
import numpy as np
from bunch import Bunch


def export_frames(filename, g, *names):
    o = dict()

    # gather into dict
    for n in names:
        if n in g:
            arr = g[n]
            o[n] = np.around(arr, decimals=2).tolist()

    # save to json file
    with open(filename, 'w') as f:
        json.dump(o, f)


def load_frames(filename):
    """
    Note: this is intended to be used in colab, hence a specific filename
    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        o = json.load(f)
        ret = Bunch()
        for pair in o.items():
            ret[pair[0]] = np.array(pair[1])

        return ret
