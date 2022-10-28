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


# playback_markers = [msection, msignature, linkeyboard]
# plot(
#         # prompt a --> b
#         # mkeyboard,
#
#         # frame restore
#         # msection,
#
#         # synth spike
#         # msignature,
#
#         # synth spike
#         # drums,
#         # bass,
#         # other,
#
#         chg,
#         cam_x,
#         cam_y,
#         cam_z,
#         cam_rx,
#         cam_ry,
#         cam_rz,
#         cfg,
#         hue,
#         msection,
#
#         norm(bassmid),
#
#         title="Measures",
#         on_click=on_click,
#         on_key=on_key)
#
# plot(
#         # prompt a --> b
#         # mkeyboard,
#
#         # frame restore
#         # msection,
#
#         # synth spike
#         # msignature,
#
#         # synth spike
#         # drums,
#         # bass,
#         # other,
#
#         # z speed
#         cam_z,
#
#         # camera translate
#         # tx, ty,
#
#         # camera rotation
#         # rx, ry, rz,
#         norm(bassmid),
#
#         title="Measures",
#         on_click=on_click,
#         on_key=on_key)
#
# # plot(linkeyboard, title="Measures", on_click=on_click, on_key=on_key)
# # plot(rx, ry, rz, on_click=on_click, title="Camera Angle")
# #
# # bass = wavg(bass, 0.08)
# # other = wavg(other, 0.08)
# # drums = wavg(drums, 0.08)
# #
# # plot(norm(pose), drums, on_click=on_click)
# #
# # min = np.min([bass, other, drums])
# # max = np.max([bass, other, drums])
# #
# # bass = (bass - min) / (max - min)
# # other = (other - min) / (max - min)
# # drums = (drums - min) / (max - min)
# #
# # plot(bass, other, drums, on_click=on_click)
# #
# # stumpy.config.STUMPY_EXCL_ZONE_DENOM = 1.25  # Reset the denominator to its default value
# #
# # mine_motifs(bass, 1, 16)
