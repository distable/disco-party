# y, sr = librosa.load("/home/nuck/tmp/separated/mdx_extra_q/project/bass_short.wav")
# f0,vflag,vprob = librosa.pyin(y, librosa.note_to_hz("C2"), librosa.note_to_hz("C7"))
# times = librosa.times_like(f0)
#
# import matplotlib.pyplot as plt
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
# ax.set(title='pYIN fundamental frequency estimation')
# fig.colorbar(img, ax=ax, format="%+2.f dB")
# ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
# ax.legend(loc='upper right')
# plt.show()


# kf = to_keyframes(df['frequency'], len(df['frequency']) / df['time'].values[-1])
# freq, confidence = load_crepe_keyframes("/home/nuck/tmp/separated/mdx/flutation/bass.f0.csv")
# indices = np.linspace(0, freq.shape[0] / fps, freq.shape[0])
# plot_s(indices, freq, confidence)

# def mine_motifs(signal, mmin=2, mmax=16, step=1):
#     results = []
#
#     for mseconds in range(mmin, mmax, step):
#         m = int(fps * mseconds)
#         with Timer() as t:
#             mp = stumpy.gpu_stump(signal, m)
#         print(f'{t.interval} gpu_stump m={m}')
#
#         set = dict(m=m, mp=mp)
#         for i in range(mp.shape[1]):
#             mdist, midx = stumpy.motifs(signal, mp[:, i], 1, 8.0)
#             if midx.shape[1] == 0:
#                 continue
#             set['idx'] = midx
#             set['dist'] = mdist
#
#             results.append(set)
#
#             # plot_motifs(signal, midx, mdist, m, on_press)
#
#     plot_mmotifs(signal, results)
