#coding:utf-8
from collections import OrderedDict
import pydub as dub
from pydub.playback import play
from pprint import pprint
from glob import glob
import os
import sys
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tempfile as temp
import functions as fn
import soundfile as sf

# 参考:https://qiita.com/yuuki__/items/4bc16ae439de46cd0d76
class TransToWav:

    def __init__(self,dir_path,_path):
        self.path = os.path.join(dir_path,_path)
        self.split_path = os.path.splitext(self.path)
        self.ext = self.split_path[-1]
        self.file_name = _path
        # self.save_file = tmp.name +"/"+ self.file_name + ".wav"
        self.save_file = tmp.name +"/"+ os.path.splitext(self.file_name)[0] + ".wav"

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

class PlayMusic:

    def __init__(self):
        self.wav_name = os.listdir(tmp.name)
        self.path = [tmp.name + "/" + i for i in self.wav_name]

    def play(self):
        for i in self.path:
            self.sound = dub.AudioSegment.from_wav(i)
            print(type(self.sound))
            play(self.sound)
        self.play()

class Analyse:

    def __init__(self):
        self.dir_path = tmp.name
        self.file_names = os.listdir(self.dir_path)
        self.file_path = [self.dir_path + "/" + i for i in self.file_names]
        self.bpm = {}
        self.key = {}
        self.chroma = {}
        # スケールのテンプレートベクトル
        # メジャーとマイナーを区別しないダイアトニックスケールのみ(メジャースケールのみの12キー)を考える
        # 順番を保ちたいのでOrderdDict
        # TODO: 効率化
        self.scale_dic = OrderedDict()
        self.scale = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
        for i in range(len(self.scale)):
                self.scale_dic[self.scale[i]] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                self.scale_dic[self.scale[i]][(i+1)%12] = 0
                self.scale_dic[self.scale[i]][(i+3)%12] = 0
                self.scale_dic[self.scale[i]][(i+6)%12] = 0
                self.scale_dic[self.scale[i]][(i+8)%12] = 0
                self.scale_dic[self.scale[i]][(i+10)%12] = 0

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

    def calc_chroma(self):
        for file_name,path in zip(self.file_names,self.file_path):
            if file_name not in self.chroma:
                # 曲の1:00~1:30を抜き出す(処理が重い)
                # TODO: 曲の中心30秒にする
                self.music, self.sr = librosa.load(path,offset=60.0, duration=30.0)
                # 楽音成分とパーカッシブ成分に分離
                harmonic, percussive = librosa.effects.hpss(self.music)
                # フレームごとのChromaを計算
                self.allChroma = librosa.feature.chroma_cens(y=harmonic)
                self._chroma = np.zeros(12)
                '''
                # プロット
                plt.figure(figsize=(12,4))
                librosa.display.specshow(allChroma, sr=self.sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
                plt.title('Chromagram')
                plt.colorbar()
                plt.tight_layout()
                plt.show()
                '''
                # フレームごとのChromaを1次元12音階に押し込む
                for i in range(self.allChroma.shape[0]):
                    for j in range(self.allChroma.shape[1]):
                        self._chroma[i] += self.allChroma[i][j]
                self.chroma[file_name] = self._chroma
            else:
                continue
        return self.chroma

    def analyse_key(self):
        for file_name,path in zip(self.file_names,self.file_path):
            if file_name not in self.chroma:
                self.music, self.sr = librosa.load(path,offset=60.0, duration=30.0)
                harmonic, percussive = librosa.effects.hpss(self.music)
                self.allChroma = librosa.feature.chroma_cens(y=harmonic)
                self._chroma = np.zeros(12)
                for i in range(self.allChroma.shape[0]):
                    for j in range(self.allChroma.shape[1]):
                        self._chroma[i] += self.allChroma[i][j]
            # Chromaとコサイン類似度が最大になるスケールを調べる
            # TODO : 調を決定せずにコサイン類似度を楽曲間類似度に反映させた方がいいか検討する
            self.maximum = -100000
            self._key = ""
            for scale_index, (name, templateVec) in enumerate(self.scale_dic.items()):
                self.similarity = fn.cos_sim(self._chroma, templateVec)
                if self.similarity > self.maximum:
                    self.maximum = self.similarity
                    self._key = name
                    self.key[file_name] = self._key
                else:
                    continue
            else:
                continue
        return self.key

def keyTest():
    # [曲名 - (求めたキー,正解のキー,Y/Nラベル)]
    list = {}
    actualKeys = {}
    key_list = analyser.analyse_key()
    # actualKeysの生成(+キーの変換)
    for file in glob(path + "/key" + '/*.txt'):
        f = open(file)
        basename = os.path.splitext(os.path.basename(file))[0]+".wav"
        # convertKeyDict = {}
        # actualKeys[basename] = convertKeyDict[f.readline()]
        actualKeys[basename] = f.readline()
        f.close()
    for i,(song_name,analyzedKey) in enumerate(key_list.items()):
        actualKey = actualKeys[song_name]
        if analyzedKey == actualKey:
            result = "Y"
        else:
            result = "N"
        list[song_name] = (analyzedKey,actualKey,result)
        # list[song_name.rjust(25)] = list.pop(song_name)
        list["song"+str(i)] = list.pop(song_name)
    pprint(list)

### 前準備
# path
# audios/GiantSteps+\ EDM\ Key\ Dataset/sample/audio
path = sys.argv[1]
# make temp directory
tmp = temp.TemporaryDirectory()

# mp3,wav save to temp file
save_ = WavSaveTmp(path)
save_.save_tmp()

# instantiation analyser
analyser = Analyse()

keyTest()

### BPMと開始拍位置を求める
# 参考 : https://qiita.com/yuuki__/items/4bc16ae439de46cd0d76
# bpm analyse
bpm_list = analyser.analyse_bpm()
# print
pprint(bpm_list)

'''
### クロマグラムを求める
# 参考 : https://qiita.com/namaozi/items/31ea255ecc6a04320dfc
# chroma calc
chroma_list = analyser.calc_chroma()
# Normalize
pprint(chroma_list)
for chroma in chroma_list:
    Normalizer = np.linalg.norm(chroma)
    chroma /= Normalizer
# print
pprint(chroma_list)

### 調性を求める
# 参考 : https://qiita.com/namaozi/items/31ea255ecc6a04320dfc
# key analyse
key_list = analyser.analyse_key()
# print
pprint(key_list)
'''

### 楽曲間類似度を求め曲順を決定する

### 曲を再生する

### 後処理
# clean up temp directory
tmp.cleanup()
