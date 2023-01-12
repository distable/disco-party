# Very useful maths for disco partying,
# along with a parametric eval and global math env to use for evaluation

import random
import time

import torch
from math import *
from opensimplex import OpenSimplex
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d
from numpy import ndarray
from .globals import *

# import pytti
simplex = OpenSimplex(random.randint(-999999, 9999999))
math_env = {}

f = 0


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

SEED_MAX = 2 ** 32 - 1

def set_seed(seed=None):
    if seed is None:
        seed = int(time.time())
    elif isinstance(seed, str):
        seed = str_to_seed(seed)

    seed = seed % SEED_MAX

    global simplex
    global current_seed

    current_seed = seed
    simplex = OpenSimplex(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def str_to_seed(seed):
    import hashlib
    seed = hashlib.sha1(seed.encode('utf-8'))
    seed = int(seed.hexdigest(), 16)
    seed = seed % SEED_MAX
    return seed


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


def jlerp(a, b, t, k=0.3):
    return lerp(a, b, jcurve(t, k))


def rlerp(a, b, t, k=0.3):
    return lerp(a, b, rcurve(t, k))


def slerp(a, b, t, k=0.3):
    return lerp(a, b, scurve(t, k))


def ilerp(lo, hi, v):
    ret = clamp01((v - lo) / (hi - lo))
    if isnan(ret):
        return 0
    return ret


def remap(v, a, b, x, y):
    t = ilerp(a, b, v)
    return lerp(x, y, t)


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


def sinb(t, a=1, p=1, o=0):
    return sin(((t / p) - o) * pi) * a


def cosb(t, a=1, p=1, o=0):
    return cos(((t / p) - o) * pi) * a


def sin1(t, a=1, p=1, o=0):
    return (sin(((t / p) - o) * pi) * .5 + .5) * a


def cos1(t, a=1, p=1, o=0):
    return (cos(((t / p) - o) * pi) * .5 + .5) * a


def tsigmoid(x, k=0.3, norm_window=None):
    lo, hi, span = 0, 0, 0

    if isinstance(x, ndarray) and norm_window is not None:
        x, lo, hi = exnorm(x, norm_window)

    x = (x - k * x) / (k - 2 * k * np.abs(x) + 1)

    if isinstance(x, ndarray) and norm_window is not None:
        x = x * (hi - lo) + lo

    return x


def scurve(x, k=0.3, norm_window=None):
    return (1 + tsigmoid(2 * x - 1, -k, norm_window)) / 2


def jcurve(x, k=0.3, norm_window=None):
    return tsigmoid(clamp01(x), k, norm_window)


def rcurve(x, k=0.3, norm_window=None):
    return tsigmoid(clamp01(x), -k, norm_window)


def perlin(t, freq=0.8):
    return simplex.noise2(t * freq, 0)


def npperlin(count, freq=1, off=0):
    ret = np.zeros(count)
    for i in range(count):
        ret[i] = perlin(i + off, freq)
    return ret


def clamp(v, min, max):
    return np.clip(v, min, max)


def clamp01(v):
    return np.clip(v, 0, 1)


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


def noisy(noise_typ, image):
    """
    Add perlin to an image
    https://stackoverflow.com/a/30609854/1319727
    """
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def schedule_spk(v, t):
    sum = 0
    l = len(t) if isinstance(t, list) else len(v) if isinstance(v, list) else 0

    for i in range(int(l)):
        tt = (t[i] if isinstance(t, list) else t)
        vv = (v[i] if isinstance(v, list) else v)
        if f % tt == 0:
            sum += vv
    return sum


def schedule(*args):
    return args[f % len(args)]


def pdiff(x):
    v = np.diff(x)
    v = np.append(v, v[-1])

    return v


def pconvolve(x, kernel):
    v = np.convolve(x, kernel)
    return v[:-(len(kernel) - 1)]


def symnorm(x, window=None):
    return norm(x, window, True)


def norm(x, window=None, symmetric=False):
    x, lo, hi = exnorm(x, window, symmetric)
    return x


def exnorm(x, window=None, symmetric=False):
    if window is not None:
        window = int(window)

        lo = np.min(sliding_window_view(x, window_shape=window), axis=1)
        hi = np.max(sliding_window_view(x, window_shape=window), axis=1)

        lo = np.pad(lo, (0, window - 1), 'edge')
        hi = np.pad(hi, (0, window - 1), 'edge')
    else:
        lo = np.min(x)
        hi = np.max(x)

    if symmetric:
        b = np.hi(np.abs(lo), np.abs(hi))
        lo = -b
        hi = b

    return (x - lo) / (hi - lo), lo, hi


def stretch(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)


def wavg(signal, window=0.08):
    size = int(signal.shape[0] * window)
    return uniform_filter1d(signal, size=size)


def blur(arr, n_iter=20, mask=None):
    if mask is None:
        mask = np.ones_like(arr)

    original = arr

    lo = arr.min()
    hi = arr.max()

    arr = norm(arr)

    kernel = np.array([1.0, 2.0, 1.0])  # Here you would insert your actual kernel of any size
    for i in range(n_iter):
        arr = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, arr)

    arr = norm(arr)
    arr = arr * (hi - lo) + lo

    return lerp(original, arr, mask)


def smooth_1euro(x):
    end = 4 * np.pi
    min_cutoff = 0.004
    beta = 0.7

    x_noisy = x
    x_hat = np.zeros_like(x_noisy)
    x_hat[0] = x_noisy[0]
    t = np.linspace(0, end, len(x))
    one_euro_filter = OneEuroFilter(t[0], x_noisy[0], min_cutoff=min_cutoff, beta=beta)

    for i in range(1, len(t)):
        x_hat[i] = one_euro_filter(t[i], x_noisy[i])

    return x_hat


def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


# pytti.parametric_eval = parametric_eval

set_seed()
