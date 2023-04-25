from datetime import datetime, timedelta
from random import shuffle
from typing import List

from typing_extensions import override

from .maths import *
from .maths import choose

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


class PNode:
    def __init__(self, text_or_children=None, wt=None, mask=None, scale: float = 1.0, add: float = 0.0, prefix: str = '', suffix: str = '', join_char: str = '', join_num: int = 999,):
        # def __init__(self, children,  **kwargs):
        self.parent = None
        self.children = []
        self.weight = wt
        self.stop = None
        self.mask = mask
        self.text = text_or_children if isinstance(text_or_children, str) else None
        self.join_char = join_char
        self.join_num = join_num

        # Modifiers
        self.add = 0
        self.scale = 1

        # Baking
        self.transformed = False
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
        if isinstance(text_or_children, str):
            self.text = text_or_children
        elif text_or_children is not None:
            self.add_children(text_or_children)

    def eval_text(self, t):
        if len(self.children) == 0:
            return self.text
        elif len(self.children) >= 1:
            sorted_children = self.children
            if any([c.has_weight() for c in sorted_children]):
                sorted_children = sorted(self.children, key=lambda n: n.get_weight_at(t), reverse=True)

            top = sorted_children[0]

            # TODO workaround, fix the actual problem
            # if self.parent is not None and self.parent.parent is None:
            #     self.join_num = 16
            # print("JOIN_NUM", self.join_num)
            join_num = min(self.join_num, len(sorted_children))
            if join_num > 1:
                s = ''
                c = self.join_char
                if c == ',': c = ', '

                for i in range(join_num):
                    s += sorted_children[i].eval_text(t)
                    s += c

                if c: # It would be very bad if we did it with len(c) == 0
                    s = s[:-len(c)]
                return s

            # print(f"PromptNode.eval_text: top={top.eval_text(t)}", [n.get_weight_at(t) for n in children_by_w])
            return top.eval_text(t)
        elif isinstance(self.text, str):
            return self.text
        elif isinstance(self.text, list):
            ret = ""
            for token in self.text:
                if isinstance(token, str):
                    ret += token
                elif isinstance(token, PNode):
                    ret += token.eval_text(t)
            return ret
        raise RuntimeError(f"Invalid PromptNode text: {self.text}")

    def transform(self, wcconfs, globals):
        # Promote
        if len(self.children) == 1 and self.children[0].can_promote():
            self.children = self.children[0].children

        import re
        import copy

        if not self.text:
            return False

        text_or_children = self.text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text_or_children = re.sub(r'\s+', ' ', text_or_children)  # Collapse multiple spaces to 1, caused by multiline strings
        text_or_children = text_or_children.strip()

        # Match map \<\w+\>
        children = []
        i = 0
        last_i = 0

        def append_wc():
            nonlocal last_i
            s = text_or_children[last_i:i + 1]
            if not s: return
            parts = re.findall(wc_regex, s)[0]
            wcname = parts[1]
            wcconf = parts[3]
            tiling = parts[5]
            wc = globals.get(wcname)
            conf = globals.get(wcconf) or wcconfs.get(wcname) or wcconfs.get('*')
            if isinstance(conf, list):
                conf = choose(conf)
            if wc is None: raise Exception(f"Couldn't get set: {wcname}")
            if conf is None:
                conf = IntervalNode(interval=1)
                # raise Exception(f"Couldn't get conf: {wcname}, {wcconf}")

            replacement_node = copy.copy(conf)
            replacement_node.children = [copy.deepcopy(wc)]
            replacement_node.join_num = 1
            if tiling:
                tile_char = tiling[-1]
                tile_num = int(tiling[:-1])
                replacement_node.join_char = tile_char
                replacement_node.join_num = int(tile_num)

            children.append(replacement_node)
            last_i = i + 1

        def append_text():
            nonlocal last_i
            s = text_or_children[last_i:i]
            if not s: return
            node = PNode(s)
            children.append(node)
            last_i = i + 1

        while i < len(text_or_children):
            c = text_or_children[i]
            if c == '<':
                append_text()
                last_i = i
                i = text_or_children.find('>', i)
                append_wc()

            i += 1

        if children:
            append_text()
            self.add_children(children)

            return True

        self.transformed = True
        return False

    def __str__(self):
        return f"{type(self).__name__}({self.text})"


    def bake(self):
        self.index = -1

    # def get_weight_at(self, t):
    #     return 0
    #
    # def __str__(self):
    #     return self.text

    def reset_bake(self):
        self.timeline = None
        self.min = 0
        self.max = 0
        self.w = 0
        self.a = 0
        self.b = 0

        self.timeline = []

    def get_lossy_scale(self):
        scale = self.scale

        # Aggregate parent scales
        n = self.parent
        while n is not None:
            scale *= parametric_eval(n.scale)
            n = n.parent

        return scale

    def has_weight(self):
        return self.weight is not None

    def get_weight_at(self, t):
        if self.weight is None:
            return 0

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

    def can_promote(self):
        return True

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        if self.timeline is not None:
            t_values = []
            for i in range(t_length):
                t = i * t_timestep
                t_values.append(self.get_bake_at(t))

            str_timeline = [f"{v:.1f}" for v in t_values]
            return f"PNode({self.text}: {', '.join(str_timeline)})"
        elif self.weight is not None:
            return f"PNode({self.weight} \"{self.text}\")"
        else:
            return f"PNode(\"{self.text}\")"

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
        # if isinstance(add, str):
        #     # if len(prefix) > 0: prefix = prefix + ' '
        #     # if len(suffix) > 0: suffix = ' ' + suffix
        #
        #     children, _ = parse_promptlines(add)
        #     self.add_children(children)
        # el
        if isinstance(add, list) or isinstance(add, tuple):
            for p in add:
                self.add_child(p)
        elif isinstance(add, PNode):
            self.add_child(add)
        else:
            raise RuntimeError(f"PromptNode.add_children: Invalid input children nodes type {type(add)}")

    def bake(self):
        for n in self.children:
            n.bake()

    # def __len__(self): return self.children.__len__()

    def __getitem__(self, k): return self.children.__getitem__(k)

    def __setitem__(self, k, v): return self.children.__setitem__(k, v)

    def __delitem__(self, k): return self.children.__delitem__(k)

    def __getslice__(self, i, j): return self.children.__getslice__(i, j)

    def __setslice__(self, i, j, s): return self.children.__setslice__(i, j, s)

    def __delslice__(self, i, j): return self.children.__delslice__(i, j)

    def __contains__(self, obj): return self.children.__contains__(obj)

    def __iter__(self): return self.children.__iter__()

    def __next__(self): return self.children.__next__()

class IntervalNode(PNode):
    def __init__(self, children=None, interval: float = 1, **kwargs):
        super(IntervalNode, self).__init__(children, **kwargs)
        self.interval = interval

    def eval_text(self, t):
        loop = int(t / self.interval)
        return self.children[loop % len(self.children)].eval_text(t)


def parse_time_to_seconds(time_str):
    # Parse time string to datetime object
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    # Convert datetime object to timedelta object
    time_delta = timedelta(hours=time_obj.hour,
                           minutes=time_obj.minute,
                           seconds=time_obj.second,
                           microseconds=time_obj.microsecond)
    # Return total seconds
    return time_delta.total_seconds()


class PTiming(PNode):
    # A node with a list of child, eval_text returns the text of the child for the current time
    # Example usage
    #
    # # Change the artist every second, then hang on the last one
    # TimingNode("""
    # 00:00 Salvador Dali
    # 00:01 Van Gogh
    # 00:02 Picasso
    # """)

    def __init__(self, children, **kwargs):
        super(PTiming, self).__init__(**kwargs)
        # Check every line, get the time for each line and text, and then create a prompt node and add to children with add_child
        self.times = {}
        for line in children.split('\n'):
            if len(line) == 0: continue
            t, text = line.split(' ', 1)

            time_seconds = parse_time_to_seconds(t)
            node = PNode(text, t)
            self.times[node] = time_seconds

            self.add_child(node)

    def can_promote(self):
        return False

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PTiming(children={len(self.children)})"

    def eval_text(self, t):
        # Find the node with the closest time
        closest_node = None
        closest_time = 999

        for node in self.children:
            node_t = self.times[node]
            if abs(node_t - t) < closest_time and node_t <= t:
                closest_node = node
                closest_time = abs(node_t - t)

        if closest_node is None:
            return ''

        return closest_node.eval_text(t)


class PGlobals(PNode):
    def __init__(self, scale: float = 1, add: float = 0, prefix: str = "", suffix: str = "", **kwargs: object):
        super(PGlobals, self).__init__([x for x in kwargs.values()], add=add, scale=scale, prefix=prefix, suffix=suffix)

        for k, v in kwargs.items():
            globals()[k] = v
        pass


class PProp(PNode):
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
        super(PProp, self).__init__(children, scale=scale, add=add, prefix=prefix, suffix=suffix)

        if drift is None: drift = [0, 0.25]
        if p is None: p = [0.75, 0.95]
        if width is None: width = [15, 20]

        self.period = width
        self.proportion = p
        self.drift = drift
        self.interpolation = lerp
        self.curve = curve

    def can_promote(self):
        return False

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PSet(children={len(self.children)})"

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


class PSeq(PNode):
    def __init__(self, children, width=10, lerp=0.25, scale=1, add=0, prefix='', suffix=''):
        super(PSeq, self).__init__(children, scale=scale, add=add, prefix=prefix, suffix=suffix)
        self.children = children

        self.width = width
        self.lerp = lerp

        if len(prefix) > 0: prefix = prefix + ' '
        if len(suffix) > 0: suffix = ' ' + suffix

    def can_promote(self):
        return False

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PSeq(children={len(self.children)})"

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


class PCurves(PNode):
    pass


# class PWords(PromptNode):
#     def __init__(self, words, scale=1, add=0, prefix='', suffix=''):
#         if ',' in words:
#             words = words.split(',')
#         elif ';' in words:
#             words = words.split(';')
#         else:
#             words = words.split(' ')
#
#         words = [f'1.0 {prefix}{word.strip()}{suffix}' for word in words]
#         s = '\n'.join(words)
#
#         super(PWords, self).__init__(s, scale=scale, add=add)
#
#
#     def get_debug_string(self,
#                          t_length=10,
#                          t_timestep=0.5):
#         return f"PWords(children={len(self.children)})"


class PList(PNode):
    def __init__(self, phrases, scale=1, add=0, prefix='', suffix=''):
        import re
        if ';' in phrases or '\n' in phrases or ',' in phrases:
            # phrases split on newlines and semicolons
            phrases = re.split(r'[\n;,]', phrases)
            phrases = [phrase for phrase in phrases if len(phrase.strip()) > 0]
            phrases = [f'1.0 {prefix}{phrase.strip()}{suffix}' for phrase in phrases]
        else:
            # phrases split on spaces
            phrases = phrases.split(' ')
            phrases = [f'1.0 {prefix}{phrase.strip()}{suffix}' for phrase in phrases]

        s = '\n'.join(phrases)
        nodes,max = parse_promptlines(s)
        super(PList, self).__init__(nodes, scale=scale, add=add)

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PPhrases(children={len(self.children)})"


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

def transform(root, wcconfs, globals):
    for v in dfs(root):
        v.transform(wcconfs, globals)


def dfs(node):
    stack = list()
    stack.append(node)

    while len(stack) > 0:
        n = stack.pop()

        yield n

        for child in n.children:
            stack.append(child)


def parse_promptlines(promptstr, prefix='', suffix=''):
    w_max = 0
    ret_nodes = []

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
        pmt = PNode(text, weight)
        w_max = max(w_max, weight)
        ret_nodes.append(pmt)

    # Normalize weights
    for p in ret_nodes:
        p.weight /= w_max

    return ret_nodes, w_max


# bake_prompt(f"{scene} with black <a_penumbra> distance fog. Everything is ultra detailed, has 3D overlapping depth effect, into the center, painted with neon reflective/metallic/glowing ink, covered in <a_gem> <a_gem2> gemstone / <n_gem> ornaments, <a_light> light diffraction, (vibrant and colorful colors), {style}, painted with (acrylic)", settypes, setmap, locals())

wc_regex = r'(\<(\w+)(:([\w|\d\/,]*))?(:([\w|\d\/,]+))?\>)'


def print_prompts(root, min, max, step=1):
    for t in range(min, max, step):
        print(f"prompt(t={t}) ", eval_prompt(root, t))

def bake_prompt(prompt: str, wcconfs, globals):
    print(f"Baking prompt:\n{prompt.strip()}")

    root = PNode(prompt)
    root.print()

    print("")
    print("")
    print("")
    print("")

    transform(root, wcconfs, globals)
    bake(root)
    root.print()
    print("")
    print_prompts(root, 1, 30)

    return root


def eval_prompt(root, t):
    in_range = True
    for node in dfs(root):
        # print("DFS", node)
        if node.timeline:
            try:
                # TODO there is a glitch in ProportionSet where the last values are zero for ALL children, so for now we will buffer a few seconds in advance to avoid this
                # print("CHECK+5")
                node.get_bake_at(t + 5)
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


# root = PromptNode("Testing the <wildcard>")
# wc = PromptPhrases("one;two;three")
#
# root.transform(dict(), {'wildcard': wc, '*': wc})
# root.bake()
# root.print()


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
