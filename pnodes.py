from bunch import Bunch
from typing import List
from random import shuffle


if not 'google.colab' in sys.modules:
  from core.maths import *
  from core.util import *
  from pytti.Perceptor.Prompt import parse_prompt, mask_image
  from pathlib import Path

# Config ----------------------------------------
TOTAL_DURATION = 60 * 5  # 5 minutes, counted in seconds
RESOLUTION = 6  # Number of keyyframes in a second. Linear interpolation in-between

# Assembled sets ----------------------------------------
params = Bunch({})  # Dummy

# Vis ----------------------------------------
vis_font = r"/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
vis_font_size = 9
vis_display_count = -1
vis_display_min_significance = 0.115

font = None


class Keyframe:
  def __init__(self):
    self.time = 0
    self.w = 0

class PromptNode:
  def __init__(self, text=None, wt=None, mask=None):
    self.parent = None
    self.children = []
    self.text = text
    self.weight = wt
    self.stop = None
    self.mask = mask

    self.add = 0
    self.scale = 1

    self.timeline = None
    self.min = 0
    self.max = 0
    self.w = 0 # The current weight, linked to the clip prompt
    self.a = 0
    self.b = 0

    self.nprompt = None  # The actual 'neural prompt' (PyTTI's Prompt class)

  def reset_state(self):
    self.timeline = None
    self.min = 0
    self.max = 0
    self.w = 0
    self.a = 0
    self.b = 0

  def prepare_timeline(self):
    self.reset_state()
    self.timeline = []

  def to_torch(self, tti):
    self.nprompt = parse_prompt(tti.embedder, self.to_pytti())
    self.nprompt.node = self
    if self.mask is not None:
      pilmask = gpil(self.mask)
      print(f"set_mask ({pilmask})")
      self.nprompt.set_mask(pilmask)

    return self.nprompt

  def update_torch(self, tti):
    global t, parametric_eval
    global i

    if self.nprompt is not None:
      # Aggregate parent scales
      scale = self.scale
      n = self.parent
      while n is not None:
        scale *= parametric_eval(n.scale)
        n = n.parent

      self.nprompt.text = self.text
      self.nprompt.weight = str(parametric_eval(self.weight) * scale)
      if self.timeline is not None:
        self.nprompt.text = self.text
        self.nprompt.weight = str(parametric_eval(self.weight) * self.evaluate(t) * scale)

    for n in self.children:
      n.update_torch(tti)

  def to_pytti(self):
    if self.stop is None:
      return f"{self.text}:{self.weight}"
    else:
      return f"{self.text}:{self.weight}:{self.stop}"

  def bake(self):
    for n in self.children:
      n.bake()

  def evaluate(self, t):
    if self.timeline is None:
      raise RuntimeError(f"Cannot evaluate a PromptNode without timeline ({self.text})")

    # Interpolate smoothly between keyframes
    idx = int(t * RESOLUTION)
    idx = clamp(idx, 0, TOTAL_DURATION * RESOLUTION)

    last = self.timeline[idx]
    next = self.timeline[idx + 1]  #

    frameStart = idx / RESOLUTION
    interframe = (t - frameStart) / (1 / RESOLUTION)

    return lerp(last, next, interframe)

  def print(self,
            t_length=10,
            t_timestep=0.5,
            depth=0):
    s = ""
    for i in range(depth): s += "   "
    s += self.to_debug_string(t_length, t_timestep)

    print(s)

    for n in self.children:
      n.print(t_length, t_timestep, depth + 1)

  def to_debug_string(self,
                      t_length=10,
                      t_timestep=0.5):
    if self.timeline is not None:
      t_values = []
      for i in range(t_length):
        t = i * t_timestep
        t_values.append(self.evaluate(t))

      str_timeline = ["{:.1f}".format(v) for v in t_values]
      return f"{', '.join(str_timeline)}"
    else:
      return f"{self.weight}"


class PromptSet(PromptNode):
  def __init__(self, *children):
    super(PromptSet, self).__init__()
    self.add_children(children)

  def __iter__(self): return self.children.__iter__()
  def __next__(self): return self.children.__next__()
  # def __len__(self): return self.children.__len__()
  # def __getitem__(self, k): return self.children.__getitem__(k)
  # def __setitem__(self, k, v): return self.children.__setitem__(k, v)
  # def __delitem__(self, k): return self.children.__delitem__(k)
  # def __getslice__(self, i, j): return self.children.__getslice__(i, j)
  # def __setslice__(self, i, j, s): return self.children.__setslice__(i, j, s)
  # def __delslice__(self, i, j): return self.children.__delslice__(i, j)
  # def __contains__(self, obj): return self.children.__contains__(obj)

  def to_debug_string(self,
                      t_length=10,
                      t_timestep=0.5):
    return type(self).__name__

  def evaluate_prompt(self, id, t):
    return self.children[id].evaluate(t)

  def to_torch(self, tti):
    return None

  def to_pytti(self):
    text = ""
    for pmt in self.children:
      text += pmt.to_pytti()
      text += "|"

    return text

  def add_child(self, node):
    self.children.append(node)
    node.parent = self

  def add_children(self, add):
    if isinstance(add, str):
      for p in PromptList(add):
        self.add_child(p)
    elif isinstance(add, list) or isinstance(add, tuple):
      for p in add:
        self.add_child(p)
    elif isinstance(add, PromptNode):
      self.add_child(add)
    else:
      raise RuntimeError(f"PromptSet.__init__: Invalid input children nodes ({add})")


class PromptList(PromptSet):
  def __init__(self, promptstr: str, scale=1, add=0, prefix='', suffix=''):
    super(PromptList, self).__init__()
    self.scale = scale
    self.add = 0

    if len(prefix) > 0: prefix = prefix + ' '
    if len(suffix) > 0: suffix = ' ' + suffix

    w_max = 0

    for text in promptstr.splitlines():
      if not text or not text.strip():
        continue

      text = text.strip()

      # [weight] [text]
      parts = text.split(' ')
      weight = float(parts[0])
      text = prefix + ' '.join(parts[1:]) + suffix
      pmt = PromptNode(text, weight)

      w_max = max(w_max, weight)

      self.add_child(pmt)

    for p in self.children:
      p.weight /= w_max

class GlobalSet(PromptSet):
  def __init__(self, scale=1, add=0, prefix="", suffix="", **kwargs):
    super(GlobalSet, self).__init__(*kwargs.values())

    self.scale = scale
    self.add = add
    self.prefix = prefix
    self.suffix = suffix

    for k, v in kwargs.items():
      globals()[k] = v
    pass


class ProportionSet(PromptSet):
  def __init__(self,
               children,
               period=None,
               proportion=None,
               drift=None,
               interpolation=0.95,
               scale=1,
               add=0,
               prefix="",
               suffix=""):
    super(ProportionSet, self).__init__(*children)
    self.children = children

    self.scale = scale
    self.add = 0
    self.prefix = prefix
    self.suffix = suffix

    if drift is None: drift = [0, 0.25]
    if proportion is None: proportion = [0.75, 0.95]
    if period is None: period = [15, 20]

    self.period = period
    self.proportion = proportion
    self.drift = drift
    self.interpolation = interpolation

    self.bake()

  def bake(self):
    totalSteps = TOTAL_DURATION * RESOLUTION

    for child in self.children.children:
      child.prepare_timeline()

    # CREATE THE KEYFRAMES BY STATE -------------------------------------------
    states: List[List[Keyframe]] = []

    # Prepare shuffled order
    shuffled_children = list()
    for i, node in enumerate(self.children.children):
      node.index = i
      shuffled_children.append(node)

    time = 0
    while time < totalSteps:
      period = int(val_or_range(self.period) * RESOLUTION)
      time += period

      shuffle(shuffled_children)

      keyframes:List[Keyframe] = [None] * len(shuffled_children)
      w = 1
      p = val_or_range(self.proportion)

      if p >= 1:
        print("WARNING: bake_proportion is >= 1")

      for node in shuffled_children:
        keyframe = Keyframe()
        keyframe.w = w # jcurve(w, 0.85)  # TODO make this not hardcoded
        keyframe.time = time + int(val_or_range(self.drift) * period)

        w *= p
        keyframes[node.index] = keyframe

      states.append(keyframes)

    # BAKE THE TIMELINES ---------------------------------------------
    lenStates = len(states)

    for ipmt in range(len(self.children.children)):
      for istate in range(lenStates):
        node = self.children.children[ipmt]
        timeline = node.timeline

        if istate == lenStates - 1:
          while len(timeline) < totalSteps:
            timeline.append(node.w)
          break

        kf0:Keyframe = states[istate][ipmt]
        kf1:Keyframe = states[istate + 1][ipmt]

        segment_duration = kf1.time - kf0.time

        lmin = kf1.time - segment_duration * val_or_range(self.interpolation)  # Interpolation frames (100% interpolation otherwise with just np.time)
        lmax = kf1.time

        for i in range(kf0.time, kf1.time):
          t = ilerp(lmin, lmax, i)
          w = lerp(kf0.w, kf1.w, t)  # scurve(t)

          timeline.append(w)


class SequenceSet(PromptSet):
  def __init__(self, children, scale=1, add=0, prefix='', suffix=''):
    super(SequenceSet, self).__init__(*children)
    self.children = children

    self.scale = scale
    self.add = 0

    self.period = 10
    self.interpolation = 0.25
    self.children = children

    if len(prefix) > 0: prefix = prefix + ' '
    if len(suffix) > 0: suffix = ' ' + suffix

    self.bake()

  def bake(self):
    # bake the timelines
    totalSteps = TOTAL_DURATION * RESOLUTION

    for child in self.children.children:
      child.prepare_timeline()

    # CREATE THE KEYFRAMES BY STATE -------------------------------------------

    period = self.period  # Current period
    index = -1  # Current prompt index
    end = 0  # Current prompt end

    for i in range(totalSteps):
      # Advance the interpolations
      for j, node in enumerate(self.children.children):
        if i == end and j == index:  # End of the current scene, tween out
          node.min = i
          node.max = i + int(period * val_or_range(self.interpolation))
          node.a = node.w
          node.b = 0
          node.k = rng(0.4, 0.6)

        # If we're still interpolating
        if i <= node.max and node.max > 0:
          t = (i - node.min) / (node.max - node.min)
          w = lerp(node.a, node.b, t)

          node.w = w
          node.timeline.append(w)
        else:
          node.timeline.append(node.w)  # Copy last w

      # Advance the scene, tween in
      if i == end:
        period = int(val_or_range(self.period) * RESOLUTION)

        index = (index + 1) % len(self.children.children)
        end += period

        node = self.children.children[index]
        node.min = i
        node.max = i + int(period * val_or_range(self.interpolation))
        node.a = node.w
        node.b = 1
        node.k = rng(0.4, 0.6)


class CurveSet(PromptSet):
  pass

# scenes = PromptList("""
# 1 A lamborghini on the road leaving a trail of psychedelic fumes and smoke
# 1 Huge megalithic rocks in the shape of human nose and ears
# 1 A large tree next to a megalithic rock
# """)
#
# backdrops = PromptList("""
# 1 A beautiful psychelic mural in the style of Van Gogh
# 1 Beautiful snake textures 4k tiling texture
# 1 Intricate bark texture from a tree in nature
# """)
#
# root = GlobalSet(
#   scenes = ProportionSet(scenes, 7.5, 0.5, [0.01, 0.25], 1),
#   backdrops = ProportionSet(backdrops, 7.5, [0.95, 0.985], [0.01, 0.25], 1)
#   eyes = ProportionSet(
# )
#
# root.bake();
# root.print(30, 2)


# region Experimental stuff
# def print_scene(scene):
#   for i in range(TOTAL_DURATION * RESOLUTION):
#     values = map(lambda x: x.timeline[i], scene.prompts)
#     parts = map(lambda x: f'{x: .2f}', values)
#     str = ', '.join(parts)
#
#     vsum = 0
#     sceneCount = len(scene.prompts)
#     for j in range(sceneCount):
#       vsum += scene.prompts[j].timeline[i]
#
#     print(f'{str} || {vsum / sceneCount:.2f}')


# def run_vis():
#   dirpath = get_image_dirpath()
#   if not os.path.isdir(dirpath):  # Doesn't exist!
#     print(f"Couldn't find '{dirpath}'")
#     return
#
#   outputdir = f"{dirpath}visualization/"
#   Path(outputdir).mkdir(parents=True, exist_ok=True)
#
#   print(f"vis dir: {dirpath}")
#
#   i = 1
#   while True:
#     # Get the image path for the index
#     filepath = get_image_filepath(i)
#
#     if not os.path.isfile(filepath):  # Doesn't exist!
#       print(f"vis step: ending ({filepath} not found)")
#       break
#
#     # Draw the visualization onto the image
#     with Image.open(filepath) as im:
#       draw_vis(im, i)
#       visimgpath = f"{outputdir}{i}.png"
#       print(f"vis step: {visimgpath}")
#       im.save(visimgpath)
#
#     i += 1
#
#
# visframe = []
#
#
# def draw_vis(im, i):
#   global font
#   if font is None:
#     font = ImageFont.truetype(vis_font, vis_font_size)
#
#   # im = Image.new(mode="RGB", size=(800, 800), color="#fff")
#   draw = ImageDraw.Draw(im)
#   draw.fontmode = "1"
#
#   pos = (6, 6)
#   pad = 4
#   vistext = ""
#
#   # Collect entries for this frame
#   for j in range(len(prompt_timelines) - 1):
#     s = params.steps_per_frame * i
#     t = s / (params.frames_per_second * params.steps_per_frame)
#     idx = int(t * RESOLUTION)
#
#     t = prompt_strings[j]
#     w = prompt_timelines[j][idx]
#
#     visframe.append(dict(t=t, w=w))
#
#   # Sort the entries in descending w
#   visframe.sort(key=lambda x: x['w'], reverse=True)
#   wmax = visframe[0]['w']
#
#   for entry in visframe[:vis_display_count]:
#     t = entry['t']
#     w = entry['w']
#
#     if w < wmax * vis_display_min_significance:
#       break
#
#     tstr = str(t)
#     limit = int(params.width / vis_font_size - 2)
#
#     wstr = f"{w:.2f}"
#     vistext += f"{wstr.ljust(5)}{tstr[:limit] + '...' * (len(tstr) > limit)}\n"
#
#   draw.text((6, 6), vistext, "#ffff", font)
#   size = draw.textsize(vistext, font)
#   rect = (pos[0] - pad, pos[1] - pad, pos[0] + size[0] + pad, pos[1] + size[1] + pad)
#   draw.rectangle(rect, fill="#00000044")
#   draw.text(pos, vistext, "#cccccc", font)
#
#   visframe.clear()


# def scene_slice(str, min, max):
#   lines = str.splitlines()
#   l = len(lines)
#   a = int(l * min)
#   b = int(l * max)

#   s = '\n'.join(lines[a:b])
#   return scene(s)

# a1 = scene_slice(sa, 0.0, 0.5)
# a2 = scene_slice(sa, 0.5, 1.0)
# a3 = scene_slice(sa, 0.0, 1.0)

# endregion
