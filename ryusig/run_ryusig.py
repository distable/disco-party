import numpy as np
from ryusig.RyusigApp import RyusigApp

np.set_printoptions(precision=2, suppress=True)

RyusigApp('project.py', '/home/nuck/mount/gdrive/AI/Disco_Diffusion/export.json', [
    'cam_x',
    'cam_y',
    'cam_z',
    'cam_rx',
    'cam_ry',
    'cam_rz',
    'cfg',
    'chg',
    'hue',
    'mprompt',
    'msectionsus',
    'brightness'
])
