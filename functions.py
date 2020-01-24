#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def monauralize(data):
    #モノラル化
    try:
        if data.shape[1] == 2:
            res = 0.5 * (data.T[0] + data.T[1])
    except:
        res = data
    return res
###

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
###

def librosa_chroma(file_path="audios/harmony1.wav", sr=44100):
    #インポート(インストールしないと使えません)
    import librosa

    # 読み込み(sr:サンプリングレート)
    y, sr = librosa.load(file_path, sr=sr)

    # 楽音成分とパーカッシブ成分に分けます
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # クロマグラムを計算します
    C = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

    # プロットします
    plt.figure(figsize=(12,4))
    librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return C
###

# 信号とbpmのマッチ度を計算
# 参考:http://ism1000ch.hatenablog.com/entry/2014/07/08/164124
def calcMatchBPM(data,bpm):
    N       = len(data)
    f_bpm   = bpm / 60
    f_frame = 44100 / 512

    phase_array = np.arange(N) * 2 * np.pi * f_bpm / f_frame
    sin_match   = (1/N) * sum( data * np.sin(phase_array))
    cos_match   = (1/N) * sum( data * np.cos(phase_array))
    return np.sqrt(sin_match ** 2 + cos_match ** 2)

# 各bpmでのマッチ度リストを返す
# 参考:http://ism1000ch.hatenablog.com/entry/2014/07/08/164124
def calcAllMatchBPM(data):
    match_list = []
    bpm_iter   = range(60,300)

    # 各bpmにおいてmatch度を計算する
    for bpm in bpm_iter:
        match = calcMatchBPM(data,bpm)
        match_list.append(match)

    return match_list
###
