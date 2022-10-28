__ryuinit__ = True

if __ryuinit__:
    from typing import Union

    from audio import *
    from maths import *
    from music import *
    from signals import *
    from load import *
    from partyutils import *
    from ryusig.RyusigApp import *

    import music
    import matplotlib.pyplot as plt

    # from motifs import *
    from pydub import AudioSegment

    from globals import *

    from collections.abc import Iterable

projdir = Path('/home/nuck/ai-art/projects/flutation/')

# Loading
if __ryuinit__:
    # bassmid = load_midi_frames('/home/nuck/ai-art/projects/flutation/bass.mid', True)
    bassmid = load_midi_frames(projdir / 'bass.mid')

    __wavs__ = [
        AudioSegment.from_wav(projdir / 'drums.wav'),
        AudioSegment.from_wav(projdir / 'bass.wav'),
        AudioSegment.from_wav(projdir / 'other.wav'),
    ]
    __wavnames__ = ['drum', 'bass', 'other']

    rx, ry, rz = load_dheadrot_frames(projdir / 'headrot.json')
    tx, ty = load_dpose_frames(projdir / 'pose.json', 0)
    rbass = load_db_keyframes(projdir / 'bass.wav')
    rother = load_db_keyframes(projdir / 'other.wav')
    rdrum = load_db_keyframes(projdir / 'drums.wav')

    framecount = rdrum.shape[0]
    t = np.linspace(0, framecount)

# Definitions
bass = norm(rbass)
other = norm(rother)
drum = norm(rdrum)

tx = tx[0:bass.shape[0]]
ty = ty[0:bass.shape[0]]
rx = rx[0:bass.shape[0]]
ry = ry[0:bass.shape[0]]
rz = rz[0:bass.shape[0]]

wbass = wavg(bass, 0.02)
wother = wavg(other, 0.02)
wdrum = wavg(drum, 5 * fps)

energy = norm(np.maximum(drum, other, bass))
ndrum = norm(drum)

# region Music
# Found with Melodyne & FL Studio
offset = 4.05
dbar = 5.94
dbeat = 0.54  # 11/4 time signature

music.offset = offset
music.dbar = dbar
music.dbeat = dbeat  # 11/4 time signature

sections = [
    ('start', 0),
    ('stress-0', dbeat * 3.5),

    # idk why these are offset by .5 beat, these musicians are on crack
    ('float', coord(3, 1)),
    ('stress-0', coord(8, 4.5)),
    # ('stress-v', coord(9, 8)),
    # ('stress-0', coord(10, 1)),
    # ('stress-v', coord(10, 8.5)),

    ('float', coord(11, 1)),
    ('stress-0', coord(19, 2)),
    # ('stress-v', coord(20, 5)),
    # ('stress-0', coord(20, 9.5)),
    # ('stress-v', coord(21, 6)),

    ('end', (other.shape[0] - 4) / fps),
]

msignature = make_temporal_markers(other, dbar, dbeat * 3.50)
msection = make_msec(other, sections)
msectionsus = make_msec(other, sections, sustain=True)
msectionspk = make_msec(other, sections, coded=False)
msectionspkmask = blur(make_msec(other, sections, coded=False), 5)
linkeyboard = make_linsec10(other, sections, 'float', dbar / 3)
linkeyboard010 = make_linsec010(other, sections, 'float', dbar / 3)
mkeyboard = make_msec(other, sections, 'float', dbar / 3, coded=False)
# endregion

move_scale = 0.35
rot_scale = 1

# chg = np.maximum(sectionate(
#         sections,
#         ('stress', lerp(0.1, 0.72, jcurve(other, 0.6) + 0.1 * drum + 0.2 * bass)),
#         ('float', lerp(0.1, 0.72, jcurve(other, 0.6) + 0.1 * drum + 0.2 * bass))
#
# ), blur(jcurve(linkeyboard, 0.85), 3) * 0.25)
# chg = lerp(0, 0.725, norm(np.maximum(rother+7, rbass+2.5, rdrum+4)))
# chg = slerp(0, 0.8, np.convolve(norm(jcurve(blur(other, 1), 0.85, 2*fps), dbeat*5.5*fps), [-2, 5, -2]))
# chg = np.convolve(lerp(0, 0.6, norm(jcurve(blur(other, 1), 0.5, 2*fps), dbeat*5.5*fps)) + 0.2*drum, [-1, 3, -1])
chg = jlerp(0, 0.8, pconvolve(jcurve(blur(other, 1), 0.4, 5*fps), [-1, 3, -1])) + 0.35*drum

# cam_x = jcurve(linkeyboard, 0.6) * (8 * norm(wavg(drum, 0.0118)) * npperlin(framecount, 0.05) + 1.5 * jcurve(drum, 0.25) * npperlin(framecount, 0.75))
# cam_y = jcurve(linkeyboard, 0.6) * (norm(wavg(drum, 0.0118)) * npperlin(framecount, 0.08) * blur(drum, 20) * jcurve(drum, 0.5) * npperlin(framecount, 0.85, 20))
cam_x = blur(linkeyboard, 5) * npperlin(framecount, 0.6)
cam_y = blur(linkeyboard, 5) * npperlin(framecount, 0.6, 20)
cam_z = 2 + 8 * blur(linkeyboard, 5)

# cam_z = energy * 0.25 + 3.5 * ndrum
# cam_z += sectionate(
#         sections,
#         ('stress', 6 * norm(blur(wavg(other, 0.01) + wavg(other, 0.005), 5)) + 4 * pdiff(drum)),
#         ('float', 8 * norm(blur(wavg(other, 0.01) + wavg(other, 0.005), 5)) + 2 * pdiff(drum)),
#         # ('float', 9.875 * blur(jcurve(scurve(other, 0.75), 0.3), 50, msectionspkmask) + 1.75 * ndrum),
# )

# Low frequency noise
cam_x += 0.65 * norm(np.clip(norm(pdiff(pdiff(rdrum))), 0, 0.55)) * npperlin(framecount, 1)
cam_y += 0.75 * norm(np.clip(norm(pdiff(pdiff(rdrum))), 0, 0.55)) * npperlin(framecount, 1, 20)
cam_z += 1.00 * norm(np.clip(norm(pdiff(pdiff(rdrum))), 0, 0.55)) * npperlin(framecount, 1, 20)

cam_rx = rot_scale * rx
cam_ry = rot_scale * ry
cam_rz = sectionate(
        sections,
        ('stress-v', 4.5 * jcurve(scurve(blur(ndrum, 2), 0.5), 0.85, 5 * fps)),
        ('float', 2.5 * scurve(linkeyboard010, 0.5))
)

cfg = 19 * scurve(1 + 0.5 * npperlin(framecount, 64), 0.25)
hue = jlerp(2,60,norm(np.convolve(jcurve(blur(drum, 2), 0.5)*blur(norm(bass,1*fps), 50), [-1,3,-1])), 0.6)

# lerp(0.08, 0.67, np.maximum(jcurve(scurve(other, 0.4), 0.6, 0.5 * fps), 1.35 * drum * 0.75))),
brightness = blur(msectionsus, 20) * -1.5

# Soften transitions
cam_z = blur(cam_z, 75, blur(msectionspk, 25))
# chg = blur(chg, 75, blur(msectionspk, 25))
# chg = jcurve(chg, 0.525, 5 * fps)

mprompt = np.maximum(msectionspk, mkeyboard)  # prompt markers
