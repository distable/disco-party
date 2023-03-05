from random import shuffle
from typing import List

from typing_extensions import override

from . import constants
from .maths import *

if not 'google.colab' in sys.modules:
    pass
    # from core.maths import *
    # from core.util import *
    # from pytti.Perceptor.Prompt import parse_prompt, mask_image

# Config ----------------------------------------

# Vis ----------------------------------------
vis_font = r"/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
vis_font_size = 9
vis_display_count = -1
vis_display_min_significance = 0.115

font = None


# Baked Data ----------------------------------------

class Keyframe:
    def __init__(self):
        self.time = 0
        self.w = 0


class PromptNode:
    def __init__(self, content=None, wt=1.0, mask=None, scale: float = 1.0, add: float = 0.0, prefix: str = '', suffix: str = ''):
        self.parent = None
        self.children = []
        self.weight = wt
        self.stop = None
        self.mask = mask
        self.text = content if isinstance(content, str) else None

        # Modifiers
        self.add = 0
        self.scale = 1

        # Baking
        self.w = 1
        self.timeline = None
        self.min = 0
        self.max = 0
        self.a = 0
        self.b = 0

        # The actual 'neural prompt' (PyTTI's Prompt class)
        self.nprompt = None

        self.scale = scale
        self.add = 0
        self.children = []
        if content is not None:
            self.add_children(content)

    def __str__(self):
        return f"{type(self).__name__}({self.text})"

    def reset_bake(self):
        self.timeline = None
        self.min = 0
        self.max = 0
        self.w = 0
        self.a = 0
        self.b = 0

        self.timeline = []

    def eval_text(self, t):
        if len(self.children) == 0:
            return self.text
        elif len(self.children) >= 1:
            ret = ""
            for n in self.children:
                ret += n.eval_text(t)
            return ret
        elif isinstance(self.text, str):
            return self.text
        elif isinstance(self.text, list):
            ret = ""
            for token in self.text:
                if isinstance(token, str):
                    ret += token
                elif isinstance(token, PromptNode):
                    ret += token.eval_text(t)
            return ret
        raise RuntimeError(f"Invalid PromptNode text: {self.text}")


    def get_lossy_scale(self):
        scale = self.scale

        # Aggregate parent scales
        n = self.parent
        while n is not None:
            scale *= parametric_eval(n.scale)
            n = n.parent

        return scale

    def get_weight_at(self, t):
        scale = self.get_lossy_scale()
        if self.timeline is not None:
            return parametric_eval(self.weight) * self.get_bake_at(t) * scale

        return parametric_eval(self.weight) * scale

    def get_bake_at(self, t):
        if self.timeline is None:
            raise RuntimeError(f"Cannot evaluate a PromptNode without timeline ({self.eval_text(t)})")

        res = constants.resolution
        seconds = constants.max_duration

        # Interpolate smoothly between keyframes
        idx = int(t * res)
        idx = clamp(idx, 0, seconds * res)
        idx = int(idx)

        last = self.timeline[idx]
        next = self.timeline[idx + 1]

        frameStart = idx / res
        interframe = (t - frameStart) / (1 / res)

        return lerp(last, next, interframe)

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        if self.timeline is not None:
            t_values = []
            for i in range(t_length):
                t = i * t_timestep
                t_values.append(self.get_bake_at(t))

            str_timeline = [f"{v:.1f}" for v in t_values]
            return f"{self.text}: {', '.join(str_timeline)}"
        else:
            return f"{self.weight} {self.text}"

    # def get_debug_string(self,
    #                      t_length=10,
    #                      t_timestep=0.5):
    #     return f"PromptNode({len(self.children)} children)"

    def print(self,
              t_length=10,
              t_timestep=0.5,
              depth=0):
        s = ""
        for i in range(depth): s += "   "
        s += self.get_debug_string(t_length, t_timestep)

        print(s)

        for n in self.children:
            n.print(t_length, t_timestep, depth + 1)

    def evaluate(self, t):
        raise RuntimeError(f"Cannot evaluate a PromptNode ({self.eval_text(t)})")

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def add_children(self, add):
        if isinstance(add, str) and '\n' in add:
            # if len(prefix) > 0: prefix = prefix + ' '
            # if len(suffix) > 0: suffix = ' ' + suffix

            children, _ = parse_promptlines(add)
            self.add_children(children)
        elif isinstance(add, list) or isinstance(add, tuple):
            for p in add:
                self.add_child(p)
        elif isinstance(add, PromptNode):
            self.add_child(add)
        else:
            # raise RuntimeError(f"PromptNode.add_children: Invalid input children nodes type {type(add)}")
            pass

    def swap(self, a, b):
        idx = self.children.index(a)
        self.children.remove(a)
        self.children.insert(idx, b)

    def bake(self):
        for n in self.children:
            n.bake()

    def __len__(self): return self.children.__len__()

    def __getitem__(self, k): return self.children.__getitem__(k)

    def __setitem__(self, k, v): return self.children.__setitem__(k, v)

    def __delitem__(self, k): return self.children.__delitem__(k)

    def __getslice__(self, i, j): return self.children.__getslice__(i, j)

    def __setslice__(self, i, j, s): return self.children.__setslice__(i, j, s)

    def __delslice__(self, i, j): return self.children.__delslice__(i, j)

    def __contains__(self, obj): return self.children.__contains__(obj)

    def __iter__(self): return self.children.__iter__()

    def __next__(self): return self.children.__next__()


class JoinList(PromptNode):
    def __init__(self, children, join_char: str = '', join_num: int = 1, **kwargs):
        super(JoinList, self).__init__(children, **kwargs)
        self.join_char = join_char
        self.join_num = join_num

    def eval_text(self, t):
        if len(self.children) == 0:
            return self.text

        children_by_w = sorted(self.children, key=lambda n: n.get_weight_at(t), reverse=True)
        top = children_by_w[0]

        # TODO workaround, fix the actual problem
        # if self.parent is not None and self.parent.parent is None:
        #     self.join_num = 16
        # print("JOIN_NUM", self.join_num)
        if self.join_num > 1:
            s = ''
            c = self.join_char
            if c == ',': c = ', '

            for i in range(self.join_num):
                s += children_by_w[i].eval_text(t)
                s += c

            s = s[:-len(c)]
            return s

        # print(f"PromptNode.eval_text: top={top.eval_text(t)}", [n.get_weight_at(t) for n in children_by_w])
        return top.eval_text(t)


class GlobalSet(PromptNode):
    def __init__(self, scale: float = 1, add: float = 0, prefix: str = "", suffix: str = "", **kwargs: object):
        super(GlobalSet, self).__init__([x for x in kwargs.values()], add=add, scale=scale, prefix=prefix, suffix=suffix)

        for k, v in kwargs.items():
            globals()[k] = v
        pass


class ProportionSet(JoinList):
    def __init__(self,
                 children,
                 width: float | tuple[float, float] = None,
                 p: float | tuple[float, float] = None,
                 drift: float | tuple[float, float] = None,
                 lerp: float | tuple[float, float] = 0.95,
                 scale: float | tuple[float, float] = 1,
                 add: float | tuple[float, float] = 0,
                 curve=None,
                 prefix: str = "",
                 suffix: str = ""):
        super(ProportionSet, self).__init__(children=children, scale=scale, add=add, prefix=prefix, suffix=suffix)

        if drift is None: drift = [0, 0.25]
        if p is None: p = [0.75, 0.95]
        if width is None: width = [15, 20]

        self.period = width
        self.proportion = p
        self.drift = drift
        self.interpolation = lerp
        self.curve = curve

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"ProportionSet({len(self.children)} children)"

    @override
    def bake(self):
        super().bake()
        totalSteps = constants.max_duration * constants.resolution

        for child in self.children:
            child.reset_bake()

        # CREATE THE KEYFRAMES BY STATE -------------------------------------------
        states: List[List[Keyframe]] = []

        # Prepare shuffled order
        shuffled_children = list()
        for i, node in enumerate(self.children):
            node.iwav = i
            shuffled_children.append(node)

        time = 0
        while time < totalSteps:
            period = int(val_or_range(self.period) * constants.resolution)
            time += period

            shuffle(shuffled_children)

            keyframes: List[Keyframe] = [None] * len(shuffled_children)
            w = 1
            p = val_or_range(self.proportion)

            if p >= 1:
                print("WARNING: bake_proportion is >= 1")

            for node in shuffled_children:
                keyframe = Keyframe()
                keyframe.w = w  # jcurve(w, 0.85)  # TODO make this not hardcoded
                keyframe.time = time + int(val_or_range(self.drift) * period)

                w *= p
                keyframes[node.iwav] = keyframe

            states.append(keyframes)

        # BAKE THE TIMELINES ---------------------------------------------
        lenStates = len(states)

        for ipmt in range(len(self.children)):
            for istate in range(lenStates):
                node = self.children[ipmt]
                timeline = node.timeline

                if istate == lenStates - 1:
                    while len(timeline) < totalSteps:
                        timeline.append(node.w)
                    break

                kf0: Keyframe = states[istate][ipmt]
                kf1: Keyframe = states[istate + 1][ipmt]

                segment_duration = kf1.time - kf0.time

                lmin = kf1.time - segment_duration * val_or_range(self.interpolation)  # Interpolation frames (100% interpolation otherwise with just np.time)
                lmax = kf1.time

                for i in range(kf0.time, kf1.time):
                    t = ilerp(lmin, lmax, i)

                    curve = self.curve or lerp
                    w = curve(kf0.w, kf1.w, t)

                    timeline.append(w)


class SequenceSet(JoinList):
    def __init__(self, children, width=10, lerp=0.25, scale=1, add=0, prefix='', suffix=''):
        super(SequenceSet, self).__init__(children, scale=scale, add=add, prefix=prefix, suffix=suffix)
        self.children = children

        self.width = width
        self.lerp = lerp

        if len(prefix) > 0: prefix = prefix + ' '
        if len(suffix) > 0: suffix = ' ' + suffix

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"SequenceSet({len(self.children)} children)"

    @override
    def bake(self):
        super().bake()

        # bake the timelines
        totalSteps = constants.max_duration * constants.resolution

        for child in self.children:
            child.reset_bake()

        # CREATE THE KEYFRAMES BY STATE -------------------------------------------

        period = self.width  # Current period
        index = -1  # Current prompt index
        end = 0  # Current prompt end

        for i in range(totalSteps):
            # Advance the interpolations
            for j, node in enumerate(self.children):
                if i == end and j == index:  # End of the current scene, tween out
                    node.min = i
                    node.max = i + int(period * val_or_range(self.lerp))
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
                from src_plugins.disco_party.constants import resolution
                period = int(val_or_range(self.width) * resolution)

                index = (index + 1) % len(self.children)
                end += period

                node = self.children[index]
                node.min = i
                node.max = i + int(period * val_or_range(self.lerp))
                node.a = node.w
                node.b = 1
                node.k = rng(0.4, 0.6)


class CurveSet(PromptNode):
    pass


class PromptWords(PromptNode):
    def __init__(self, words, scale=1, add=0, prefix='', suffix=''):
        if ',' in words:
            words = words.split(',')
        else:
            words = words.split(' ')

        words = [f'1.0 {prefix}{word.strip()}{suffix}' for word in words]
        s = '\n'.join(words)

        super(PromptWords, self).__init__(s, scale=scale, add=add)


    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PromptWords({len(self.children)} children)"


class PromptPhrases(PromptNode):
    def __init__(self, phrases, scale=1, add=0, prefix='', suffix=''):
        phrases = phrases.strip().split('\n')
        phrases = [f'1.0 {prefix}{phrase.strip()}{suffix}' for phrase in phrases]
        s = '\n'.join(phrases)
        super(PromptPhrases, self).__init__(s, scale=scale, add=add)

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PromptPhrases({len(self.children)} children)"


# region Visualization
# scenes = PromptNode("""
# 1 A lamborghini on the road leaving a trail of psychedelic fumes and smoke
# 1 Huge megalithic rocks in the shape of human nose and ears
# 1 A large tree next to a megalithic rock
# """)
#
# backdrops = PromptNode("""
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

# def print_scene(scene):
#   for i in range(max_duration * resolution):
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
#     idx = int(t * resolution)
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

def bake(root):
    ret = []

    # root.bake()
    for v in dfs(root):
        v.bake()
        ret.append(v)

    return ret


def dfs(node):
    stack = list(node)

    while len(stack) > 0:
        n = stack.pop()

        yield n

        for child in n.children:
            stack.append(child)


def parse_promptlines(promptstr, prefix='', suffix=''):
    w_max = 0
    ret = []

    for text in promptstr.splitlines():
        if not text or not text.strip():
            continue

        text = text.strip()

        # [weight] [text]
        parts = text.split(' ')
        weight = float(parts[0])
        text = f"{prefix}{' '.join(parts[1:])}{suffix}"

        # Split the text and interpret each token
        # Example text: "Aerial shot of __________ mountains by Dan Hillier, drawn with psychedelic white ink on black paper"
        tokens = text.split(' ')
        pmt = PromptNode(text, weight)
        w_max = max(w_max, weight)
        ret.append(pmt)

    # Normalize weights
    for p in ret:
        p.weight /= w_max

    return ret, w_max


# bake_prompt(f"{scene} with black <a_penumbra> distance fog. Everything is ultra detailed, has 3D overlapping depth effect, into the center, painted with neon reflective/metallic/glowing ink, covered in <a_gem> <a_gem2> gemstone / <n_gem> ornaments, <a_light> light diffraction, (vibrant and colorful colors), {style}, painted with (acrylic)", settypes, setmap, locals())

wc_regex = r'(\<(\w+)(:([\w|\d\/,]*))?(:([\w|\d\/,]+))?\>)'


class WildcardNode(PromptNode):
    def __init__(self, text, wcconfs, globals):
        super().__init__()

        import re
        import copy

        self.text = text
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text) # Collapse multiple spaces to 1, caused by multiline strings
        text = text.strip()

        def parse_wc(wctext):
            parts = re.findall(wc_regex, wctext)[0]
            wcname = parts[1]
            wcconf = parts[3]
            tiling = parts[5]
            wc = globals.get(wcname)
            conf = globals.get(wcconf) or wcconfs.get(wcname) or wcconfs.get('*')

            if isinstance(conf, list):
                conf = choose(conf)

            if wc is None: raise Exception(f"Couldn't get set: {wcname}")
            if conf is None: raise Exception(f"Couldn't get conf: {wcname}, {wcconf}")

            v = copy.copy(conf)
            v.children = copy.deepcopy(wc)
            if tiling:
                tile_char = tiling[-1]
                tile_num = int(tiling[:-1])
                v.join_char = tile_char
                v.join_num = int(tile_num)
                print(v)

            return v

        # Match map \<\w+\>
        children = []
        i = 0
        last_i = 0

        def append_wc():
            nonlocal last_i
            s = text[last_i:i + 1]
            if not s: return
            node = parse_wc(s)
            children.append(node)
            last_i = i + 1

        def append_text():
            nonlocal last_i
            s = text[last_i:i]
            if not s: return
            node = WildcardNode(s, wcconfs, globals)
            children.append(node)
            last_i = i + 1

        while i < len(text):
            c = text[i]
            if c == '<':
                append_text()
                last_i = i
                i = text.find('>', i)
                append_wc()

            i += 1

        if children:
            append_text()
            self.add_children(children)
            # self.children = children

            for child in dfs(self):
                if child.text and '<' in child.text and '>' in child.text:
                    wn = WildcardNode(child.text, wcconfs, globals)
                    child.parent.swap(child, wn)


    def bake(self):
        self.index = -1

    def get_weight_at(self, t):
        return 0

    def __str__(self):
        return self.text


def bake_prompt(prompt: str, wcconfs, globals):
    print("")
    print(f"Baking prompt:\n{prompt.strip()}")

    root = WildcardNode(prompt, wcconfs, globals)

    bake(root)
    root.print()
    print("")

    return root


def eval_prompt(root, t):
    in_range = True
    for node in dfs(root):
        # print("DFS", node)
        if node.timeline:
            try:
                # TODO there is a glitch in ProportionSet where the last values are zero for ALL children, so for now we will buffer a few seconds in advance to avoid this
                # print("CHECK+5")
                node.get_bake_at(t+5)
            except:
                # print("FAILED")
                in_range = False
                break

    if not in_range:
        # Extend the duration and rebake
        constants.max_duration = t + 50
        print(f"pnodes: extending max duration to {constants.max_duration:.02f}s and rebaking")

        bake(root)

        # Try again
        return eval_prompt(root, t)

    return root.eval_text(t)


# region Disco

#   {
#    0:"Aerial shot of scenic meadows and mountains by Dan Hillier, drawn with psychedelic white ink on black paper:evaluate_node:i"
#   }
def bake_disco(root: PromptNode):
    dd_prompts = []
    for v in bake(root):
        dd_prompts.append(f'{v.eval_text()}:evaluate_node:{v.iwav}')

    return {0: dd_prompts}


# endregion


# region PyTTI


# def to_pytti_object(self, tti):
#     self.nprompt = parse_prompt(tti.embedder, self.to_pytti())
#     self.nprompt.node = self
#     if self.mask is not None:
#         pilmask = gpil(self.mask)
#         print(f"set_mask ({pilmask})")
#         self.nprompt.set_mask(pilmask)
#
#     return self.nprompt
#
#
# def update_pytti(self, t):
#     if self.nprompt is not None:
#         self.nprompt.text = self.eval_text()
#         self.nprompt.weight = str(self.get_weight_at(t))
#
#
# def to_pytti(self):
#     if self.stop is None:
#         return f"{self.eval_text()}:{self.weight}"
#     else:
#         return f"{self.eval_text()}:{self.weight}:{self.stop}"
#

# endregion
