# Very useful maths for disco partying,
# along with a parametric eval and global math env to use for evaluation

import random
import time

import torch
import numpy as np
from bunch import Bunch
from math import *
from opensimplex import OpenSimplex
import sys
import numpy as np

# import pytti

simplex = OpenSimplex(random.randint(-999999, 9999999))
math_env = {}


def prepare_math_env(*args):
    math_env.clear()
    for a in args:
        for k, v in a.items():
            math_env[k] = v


def update_math_env(k, v):
    math_env[k] = v
    globals()[k] = v
    # print(math_env[k])
    # print(globals()[k])
    # print(globals())
    # math_env = {'abs': abs, 'max': max, 'min': min, 'pow': pow, 'round': round, '__builtins__': None}
    # math_env.update({key: getattr(math, key) for key in dir(math) if '_' not in key})


def set_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    global simplex
    global current_seed

    current_seed = seed
    simplex = OpenSimplex(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def parametric_eval(string, **kwargs):
    if isinstance(string, str):
        try:
            # print(len(math_env))
            # print(math_env['cosb'])
            output = eval(string)
        except SyntaxError as e:
            raise RuntimeError(f'Error in parametric value: {string}')

        return output
    elif isinstance(string, list):
        return val_or_range(string)
    elif isinstance(string, tuple):
        return val_or_range(string)
    else:
        return string


def choose_or(l: list, default, p=None):
    if len(l) == 1:
        return l[0]
    elif len(l) == 0:
        return default
    else:
        ret = random.choices(l, p)
        if isinstance(ret, list):
            return ret[0]
        return ret


def choose(l: list, p=None):
    ret = random.choices(l, p)
    if isinstance(ret, list):
        return ret[0]
    return ret


def choose_or(l: list, default, p=None):
    if len(l) == 1:
        return l[0]
    elif len(l) == 0:
        return default
    else:
        ret = random.choices(l, p)
        if isinstance(ret, list):
            return ret[0]
        return ret


def wchoose(l: list, weights):
    return random.choices(l, weights)


def rng(min=None, max=None):
    if not min and not max:
        return random.random()
    elif not min and max:
        return random.uniform(0, max)
    else:
        return random.uniform(min, max)


def rngi(min=None, max=None):
    if not min and not max:
        return random.random(-sys.maxsize, sys.maxsize)
    elif not min and max:
        return random.randint(0, max)
    else:
        return random.randint(min, max)


def val_or_range(v, max=None):
    if isinstance(v, list) or isinstance(v, tuple):
        return random.uniform(v[0], v[1])
    elif isinstance(v, float) or isinstance(v, int):
        return float(v)
    else:
        raise ValueError(f"maths.val_or_range: Bad argument={v}, type={type(v)}")


def lerp(a, b, t):
    return a + (b - a) * clamp01(t)


def ilerp(min, max, v):
    return clamp01((v - min) / (max - min))


def sign(v):
    return copysign(1, v)


def stsin(t, a, p, w):
    return sign(sin(t / p * pi)) * sin(t / p * pi) ** w * a


def stcos(t, a, p, w):
    return sign(cos(t / p * pi)) * cos(t / p * pi) ** w * a


def swave(t):
    return 1


def cwave(t, p1, p2, a1, a2, o1):
    return 1


def sinb(t, a, p, o=0):
    return sin(((t / p) - o) * pi) * a


def cosb(t, a, p, o=0):
    return cos(((t / p) - o) * pi) * a


def sin1(t, a, p, o=0):
    return (sin(((t / p) - o) * pi) * .5 + .5) * a


def cos1(t, a, p, o=0):
    return (cos(((t / p) - o) * pi) * .5 + .5) * a


def tsigmoid(x, k=0.3):
    return (x - k * x) / (k - 2 * k * abs(x) + 1)


def scurve(x, k=0.3):
    return (1 + tsigmoid(2 * x - 1, -k)) / 2


def jcurve(x, k=0.3):
    return tsigmoid(clamp01(x), k)


def rcurve(x, k=0.3):
    return tsigmoid(clamp01(x), -k)


def noise(t, freq=0.8):
    return simplex.noise2(t * freq, 0)


def clamp(v, min, max):
    if v < min: return min
    if v > max: return max
    return v


def clamp01(v):
    return clamp(v, 0, 1)


def euler_to_quat(pitch, yaw, roll):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return (qw, qx, qy, qz)


def kprint(*kargs):
    s = ""
    for v in kargs:
        s += f"{v}  "

    print(s)


def kwprint(**kwargs):
    s = ""
    for k, v in kwargs.items():
        s += f"{k}={v}  "

    print(s)


# pytti.parametric_eval = parametric_eval

set_seed()
