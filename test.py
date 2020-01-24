#coding:utf-8
from collections import OrderedDict
import os
import sys
import numpy as np
import pydub as dub
from pydub.playback import play
import matplotlib.pyplot as plt
import shutil
import tempfile as temp
import librosa
import librosa.display
import functions as fn
import soundfile as sf
import sys

# 参考:https://qiita.com/yuuki__/items/4bc16ae439de46cd0d76
class TransToWav:

    def __init__(self,dir_path,_path):
        self.path = os.path.join(dir_path,_path)
        self.split_path = os.path.splitext(self.path)
        self.ext = self.split_path[-1]
        self.file_name = _path
        self.save_file = tmp.name +"/"+ self.file_name + ".wav"

    def trans_wav(self):
        self.music = dub.AudioSegment.from_mp3(self.path)
        self.music.export(self.save_file,format="wav")

    def save_wav(self):
            if self.ext == ".mp3":
                self.trans_wav()
            elif self.ext == ".wav":
                shutil.copyfile(self.path,self.save_file)
            else:
                pass

class WavSaveTmp:

    def __init__(self,path):
        self.path = path
        if os.path.isdir(self.path) is True:
            self.dir_path = self.path
            self.file_name = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path,f))]
        elif os.path.isfile(path) is True:
            self.dir_path = os.path.dirname(self.path)
            self.file_name = os.path.basename(self.path)
        else:
            pass


    def save_tmp(self):
        if type(self.file_name) is list:
            for i in self.file_name:
                handle_wav = TransToWav(self.path,i)
                handle_wav.save_wav()
        else:
            handle_wav = TransToWav(self.dir_path,self.file_name)
            handle_wav.save_wav()

class play_music:

    def __init__(self):
        self.wav_name = os.listdir(tmp.name)
        self.path = [tmp.name + "/" + i for i in self.wav_name]

    def play(self):
        for i in self.path:
            self.sound = dub.AudioSegment.from_wav(i)
            print(type(self.sound))
            play(self.sound)
        self.play()


class BpmAnalyse:
    def __init__(self):
        self.dir_path = tmp.name
        self.file_names = os.listdir(self.dir_path)
        self.file_path = [self.dir_path + "/" + i for i in self.file_names]
        self.bpm = {}

    def analyse_bpm(self):
        for file_name,path in zip(self.file_names,self.file_path):
            if file_name not in self.bpm:
                self.music, self.sr = librosa.load(path)
                # self._bpm =  librosa.beat.tempo(self.music,self.sr)
                # onset_env = librosa.onset.onset_strength(self.music, self.sr)
                self._bpm, self._beatsPosition = librosa.beat.beat_track(self.music,self.sr)
                # 開始拍位置をフレームから秒に変換する
                self._startBeatPosition = librosa.frames_to_time(self._beatsPosition)
                # self._bpm = librosa.beat.tempo(onset_env, self.sr)
                self.bpm[file_name] = (int(self._bpm), self._startBeatPosition[0])
            else:
                continue
        return self.bpm

class calcChroma:
    def librosa_chroma(self, file_path, sr):

        # 読み込み(sr:サンプリングレート)
        # 曲の1:00~1:30を抜き出す(処理が重い)
        # TODO: 曲の中心30秒にする
        y, sr = librosa.load(file_path, sr=sr,offset=0.0, duration=30.0)

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

### BPMと開始拍位置を求める
# 参考:https://qiita.com/yuuki__/items/4bc16ae439de46cd0d76

# path
path = sys.argv[1]
# make temp directory
tmp = temp.TemporaryDirectory()

# mp3,wav save to temp file
save_ = WavSaveTmp(path)
save_.save_tmp()

# bpm analyse
analyser = BpmAnalyse()
bpm_list = analyser.analyse_bpm()

print(bpm_list)

# clean up temp directory
tmp.cleanup()


### クロマグラムを求める
# フレームごとのChromaを求める
file_name = "落魄フード_inst_BPM132.wav"
data, sampling_rate = sf.read(path + file_name)
Chromagram = calcChroma()
filename = path + file_name
allChroma = Chromagram.librosa_chroma(filename,sampling_rate)

# フレームごとのChromaを1次元の12音階に押し込む
Chroma = np.zeros(12)
for i in range(allChroma.shape[0]):
    for j in range(allChroma.shape[1]):
        Chroma[i] += allChroma[i][j]

### キーを求める
# スケールのテンプレートベクトル
# メジャーとマイナーを区別しないダイアトニックスケールのみ(メジャースケールのみの12キー)を考える
# 順番を保ちたいのでOrderdDict
# TODO: 効率化
one_seventh = 1.0/7
scale_dic = OrderedDict()
scale = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
for i in range(len(scale)):
        scale_dic[scale[i]] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        scale_dic[scale[i]][(i+1)%12] = 0
        scale_dic[scale[i]][(i+3)%12] = 0
        scale_dic[scale[i]][(i+6)%12] = 0
        scale_dic[scale[i]][(i+8)%12] = 0
        scale_dic[scale[i]][(i+10)%12] = 0

# Normalize
Normalizer = np.linalg.norm(Chroma)
Chroma /= Normalizer
Normalizer = np.linalg.norm(scale_dic[C])
for i in range(len(scale)):
        scale_dic[scale[i]] /= Normalizer

# Chromaとコサイン類似度が最大になるスケールを調べる
for scale_index, (name, vector) in enumerate(scale_dic.items()):
    similarity = fn.cos_sim(sum_chroma, vector)
    result[chord_index][int(nth_chord - 1)] = similarity
    if similarity > maximum:
        maximum = similarity
        this_chord = name


### 楽曲間類似度を求め曲順を決定する
