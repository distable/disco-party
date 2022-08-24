from nltk.corpus.reader import Synset
from typing import List

from pnodes import *

import nltk

# nltk.download('wordnet')
# nltk.download('wordnet_ic')
# nltk.download('omw-1.4')

import conceptnet_lite as cn
from nltk.corpus import wordnet as wn, wordnet_ic
from gensim.models import Word2Vec
import gensim.downloader as api

model = api.load('glove-wiki-gigaword-50')
print()

# brown_ic = wordnet_ic.ic('ic-brown.dat')

cn.connect("/home/nuck/.local/share/conceptnet5/conceptnet.db")
word = "introvert"

# Config
# ----------------------------------------
themes = [(0.72, 'leaves'),
          (0.72, 'trees'),
          (0.72, 'flowers'),
          (0.73, 'fire')]

blacklist = [
    # 0.63, '36-years-old',
    # (0.53, 'dichagyris'),
    # (0.53, 'iranian-man'),
    # (0.53, 'helcystograma')
    (0.66, "43-42"),
    (0.66, "20-16",
     ),
]


# Functions
# ----------------------------------------

def get_related_cn():
    ss = wn.synset(f'{word}.n.01')
    result = []

    for e in cn.edges_for(cn.Label.get(text=word, language='en').concepts, same_language=True):
        s = wn.synsets(f'{e.end.text}')
        if len(s) == 0:
            continue

        s = s[0]

        try:
            similarity = model.similarity(ss.name().split('.')[0], s.name().split('.')[0])
            # if similarity < 0.735:
            result.append(dict(sim=similarity, name=e.end.text, rname=e.relation.text))
        except:
            pass
    return result


def get_related_w2v(size, luck=None, history_penalty_dist=5, history_penalty=0.1):
    luck = luck if luck is not None else [0.05, 0.072]
    ret = []
    w = word

    l1 = 1 - val_or_range(luck)
    l2 = 1 - val_or_range(luck)

    if l1 > l2:
        # Swap
        tmp = l1
        l1 = l2
        l2 = tmp

    while len(ret) < size:
        canditates = model.most_similar(w, topn=50)  # { name, similarity }
        final_candidates = []
        l = val_or_range(luck)

        for ic in range(len(canditates)):
            c = canditates[ic]
            name = c[0]
            sim = c[1]
            wgt = 1

            # Skip if it's not a real word in wordnet
            if len(wn.synsets(name)) == 0:
                continue

            # Apply luck bonus
            if l1 < sim < l2:
                wgt = 2

            # Apply theme bonus using the word2vec model
            # for theme in themes:
            #     if model.similarity(w, theme[1]) > theme[0]:
            #         wgt += 2

            # Apply blacklist penalty
            for blacklist_entry in blacklist:
                if model.similarity(w, blacklist_entry[1]) > blacklist_entry[0]:
                    wgt -= 3


            # Apply history penalty
            for ih in range(len(ret) - 1, max(0, len(ret) - history_penalty_dist - 1), -1):
                if ret[ih] == c[0]:  # Names matching, it's in history
                    c[1] -= history_penalty

                break

            if wgt < 0:
                continue

            final_candidates.append((name, wgt));

        w = choose(final_candidates, [x[1] for x in final_candidates])[0]
        ret.append(w)

    return ret


result = get_related_w2v(1000)
print(result)


# result.sort(key=lambda x: x[0], reverse=True)
# for d in result:
#     print(d.sim, ":", d.name, "|", e.rname)


def print_wn_chain(size):
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    start = wn.synset('ocean.n.01')
    result = []
    size = 5000

    vertical_probability = 0.2
    vertical_elapsed_penalty = 0.25
    vertical_requirement = 4
    history_effort = 30
    skip_effort = 30

    vertical_elapsed = 0
    skip_elapsed = 0

    def multiplier():
        return 1

    themes = [(x[0], wn.synset(f"{x[1]}.n.01")) for x in themes]

    def get_probs(l: List[Synset]):
        ret = []
        for x in l:
            prob = 0
            sum = 0
            for y in themes:
                prob += wn.lin_similarity(x, y[1], brown_ic)
                sum += y[0]

            ret.append(prob / sum)

        return ret

    skip = False
    saved_node = None
    node = start

    while len(result) < size:
        saved_node = node

        vertical_elapsed += 1
        hypernyms = node.hypernyms()
        hyponyms = node.hyponyms()
        neighbors = choose_or(hypernyms, node, get_probs(hypernyms)).hyponyms()

        # Decision logic
        # ----------------------------------------

        vertical_scale = 1 + lerp(0, 0.25, clamp01(len(neighbors) / 5)) + lerp(0, 0.25, skip_elapsed / skip_effort)

        if rng() < vertical_probability * vertical_scale + vertical_elapsed_penalty * vertical_elapsed or len(neighbors) == 0:
            vertical_elapsed = 0
            if rng() < 0.5 or len(hyponyms) < vertical_requirement:
                node = choose_or(hypernyms, node, p=get_probs(hypernyms))
            else:
                node = choose(hyponyms, p=get_probs(hyponyms))
        else:
            # Get a random neighbor
            node = choose(neighbors)

        # Decision logic
        # ----------------------------------------

        # Can't reuse a node too quickly, use distance penalty

        if not skip:
            w = len(result)
            for i in range(history_effort):
                index = w - 1 - i
                if index < 0:
                    break

                n = result[min(index, w - 1)]
                # print((n, node))
                if n.name == node.name:
                    skip = True
                    break

        adding = not skip or skip_elapsed > skip_effort
        if adding:
            result.append(node)
            skip_elapsed = 0
            skip = False
        else:
            node = saved_node
            skip_elapsed += 1
            skip = False

    print(f"RESULT: {[r.name().split('.')[0] for r in result]}")


# for ss in synsets[:1000]:
#     print((ss, similarity))
#     results.append((ss, similarity))

# results.sort(key=lambda x: x[1], reverse=True)

# print(results[:50])


class WordNode(PromptNode):
    pass


class AdjectiveNode(PromptList):
    def __init__(self, text):
        super(AdjectiveNode, self).__init__()
        pass


class NounNode(PromptNode):
    def __init__(self, text):
        pass

    def cycle(self, reach=0.25):
        pass


class VerbNode(PromptNode):
    pass


class AdverbNode(PromptNode):
    pass
