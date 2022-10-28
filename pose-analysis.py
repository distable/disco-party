# %matplotlib inline
import json

import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from dateutil.parser import parse, parserinfo

plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')

# ---------------------------------------------------------

with open('data.json', 'r') as f:
    frames = json.load(f) # [{kp: ..., delta:... }, {kp:,delta:}, {...}]
    v = []
    for i in range(len(frames)):
        frame = []
        if 'delta' in frames[i]:
            for kp in frames[i]['delta']:
              frame.append(kp[1]) # Etract X only
        v.append(frame)

    print(v)

    k = [f'Joint {v}' for v in range(1,17)]

    df = pd.DataFrame(v, columns=k)

print(df.head())


# ---------------------------------------------------------

fig, axs = plt.subplots(df.shape[1], sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Can You Spot The Multi-dimensional Motif?', fontsize='30')

for i in range(df.shape[1]):
    axs[i].set_ylabel(f'T{i + 1}', fontsize='20')
    axs[i].set_xlabel('Time', fontsize ='20')
    axs[i].plot(df[f'Joint {i + 1}'])

plt.show()

# ---------------------------------------------------------
#
# m = 30
# mps = {}  # Store the 1-dimensional matrix profiles
# motifs_idx = {}  # Store the index locations for each pair of 1-dimensional motifs (i.e., the index location of two smallest matrix profile values within each dimension)
# for dim_name in df.columns:
#     mps[dim_name] = stumpy.stump(df[dim_name], m)
#     motif_distance = np.round(mps[dim_name][:, 0].min(), 1)
#     print(f"The motif pair matrix profile value in {dim_name} is {motif_distance}")
#     motifs_idx[dim_name] = np.argsort(mps[dim_name][:, 0])[:2]


# # ---------------------------------------------------------
#
# fig, axs = plt.subplots(len(mps), sharex=True, gridspec_kw={'hspace': 0})
#
# for i, dim_name in enumerate(list(mps.keys())):
#     axs[i].set_ylabel(dim_name, fontsize='20')
#     axs[i].plot(df[dim_name])
#     axs[i].set_xlabel('Time', fontsize ='20')
#     for idx in motifs_idx[dim_name]:
#         axs[i].plot(df[dim_name].iloc[idx:idx+m], c='red', linewidth=4)
#         axs[i].axvline(x=idx, linestyle="dashed", c='black')
#
# plt.show()




# m = 30
#
# mps, indices = stumpy.mstump(df, m)
# motifs_idx = np.argmin(mps, axis=1)
# nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]
#
#
#
# fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0})
#
# for k, dim_name in enumerate(df.columns):
#     axs[k].set_ylabel(dim_name, fontsize='20')
#     axs[k].plot(df[dim_name])
#     axs[k].set_xlabel('Time', fontsize ='20')
#
#     axs[k + mps.shape[0]].set_ylabel(dim_name.replace('T', 'P'), fontsize='20')
#     axs[k + mps.shape[0]].plot(mps[k], c='orange')
#     axs[k + mps.shape[0]].set_xlabel('Time', fontsize ='20')
#
#     axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#     axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
#     axs[k + mps.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#     axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')
#
#     if dim_name != 'T3':
#         axs[k].plot(range(motifs_idx[k], motifs_idx[k] + m), df[dim_name].iloc[motifs_idx[k] : motifs_idx[k] + m], c='red', linewidth=4)
#         axs[k].plot(range(nn_idx[k], nn_idx[k] + m), df[dim_name].iloc[nn_idx[k] : nn_idx[k] + m], c='red', linewidth=4)
#         axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
#         axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')
#     else:
#         axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='black')
#         axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='black')
#
# plt.show()