#coding:utf-8

"""

     * 楽曲間類似度の計算 -> リストを作って再生

Todo:

    * python main.py (オーディオファイルまでの相対パス)
    * オーディオフォルダの中に正解アノテーションのフォルダを入れる
    * オーディオの対応形式: mp3,wav(mp3はwavに変換)

"""

from collections import OrderedDict
import pydub as dub
from pydub.playback import play
from pydub.silence import split_on_silence
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
from typing import NamedTuple
import random
from tqdm import tqdm
import datetime

class TransToWav:
    '''

    参考: https://qiita.com/yuuki__/items/4bc16ae439de46cd0d76

    '''
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
    '''

    参考: https://qiita.com/yuuki__/items/4bc16ae439de46cd0d76

    '''
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

class Mixer:
    """

     音楽再生・ミックスクラス

     Attributes:
        self.wav_name (リスト): tmp直下のwavファイルのリスト.
        self.path (リスト): tmp直下のwavファイル(絶対パス)のリスト.

    """

    def __init__(self, songDict, playList = None):
        """イニシャライザ

         音楽プレーヤーの初期化

        Args:
            songDict (順序つき辞書): [ファイル名 - ((BPM,beats),Key)]
            playList (リスト): 曲名のリスト

        Note:
            playListを指定しない場合はディレクトリ内の曲をos.listdirで取得した順番で再生

        """
        self._songDict = songDict
        self.songDict = OrderedDict()
        for key in self._songDict:
            self.songDict[tmp.name + "/" + key] = self._songDict[key]
        if playList is None:
            self.wav_name = os.listdir(tmp.name)
            self.playList = [tmp.name + "/" + i for i in self.wav_name]
        else:
            self.playList = [tmp.name + "/" + i for i in playList]

    def play(self):
        for i in self.playList:
            self.song = dub.AudioSegment.from_wav(i)
            print(type(self.song))
            play(self.song)
        self.play()

    def MIX(self):
        ### 方針: 適切な長さのサイレンスに楽曲をオーバーレイする (-> 最後に無音部分をカット?)
        self.silenceDuration = 0
        for song in self.playList:
            self.silenceDuration += self.songDict[song].BPM.beats[-16]
        else:
            self.silenceDuration += 60
        self.mixDown = dub.AudioSegment.silent(duration=self.silenceDuration * 1000)
        # 拍位置を合わせて楽曲をオーバーレイする
        # エフェクトを適用する(High Pass and Fade in)
        self.startPosition = 0 # 曲の再生開始位置[sec]
        self.prevSongEndBeatPosition = 0 # 曲の終了拍位置[sec]
        self.fadeInDuration = 0 # 次の曲のフェードインをかける時間[sec]
        self.fadeOutDuration = 0 # 曲のフェードアウトをかける時間[sec]
        self.startPositionDict = {} # 開始位置を記録しておく[sec]
        for i, song in tqdm(enumerate(self.playList)):
            self.song_as = dub.AudioSegment.from_wav(song)
            self.fadeOutDuration = self.song_as.duration_seconds - self.songDict[song].BPM.beats[-16] # 再生時間[sec] - 終了から15拍目の位置
            self.fadeInDuration = self.songDict[song].BPM.beats[15]
            if i is not 0 and i is not len(self.playList)-1: # フェードアウト、フェードインを適用
                self.song_as = self.song_as.fade_in(duration=int(self.fadeInDuration * 1000))
                self.song_as = self.song_as.fade_out(duration=int(self.fadeOutDuration * 1000))
            elif i is 0: # フェードアウトのみ適用
                self.song_as = self.song_as.fade_out(duration=int(self.fadeOutDuration * 1000))
            elif i is len(self.playList)-1: # フェードインのみ適用
                self.song_as = self.song_as.fade_in(duration=int(self.fadeInDuration * 1000))
            if i is not 0: # 最初の曲のみ0[sec]から再生
                self.startPosition = self.prevSongEndBeatPosition - self.songDict[song].BPM.beats[15]
            self.mixDown = self.mixDown.overlay(self.song_as, position=self.startPosition*1000, loop=False, times=1, gain_during_overlay=0)
            self.startPositionDict[song] = self.startPosition
            # self.prevSongEndBeatPosition += (self.songDict[song].BPM.beats[-1] - self.songDict[song].BPM.beats[0])
            self.prevSongEndBeatPosition = self.startPosition + self.songDict[song].BPM.beats[-1]
        else:
            # ミックスを書き出す
            print("\nExporting Mix...")
            chunks = split_on_silence(self.mixDown, min_silence_len=3000, silence_thresh=-40, keep_silence=500)
            self._exPath = "/Users/hmori/ChromagramSample3/MixDown"
            if not os.path.isdir(self._exPath):
                os.makedirs(self._exPath)
            chunks[0].export(self._exPath + "/" + "MixDown😈.mp3", format="mp3")
            # self.mixDown.export(self._exPath + "/" + "MixDown😈.mp3", format="mp3")
            print("\nSuccessful export!🎉🍺 : " + self._exPath + "/" + "MixDown😈.mp3")
            # 曲のリスト、再生位置を書き出す
            print("\n------------------------------- Playlist -------------------------------\n")
            for index, song in enumerate(self.playList):
                songname = os.path.splitext(os.path.basename(song))[0]
                print(str(index+1) + " " + str(songname))
                td = datetime.timedelta(seconds=round(self.startPositionDict[song]))
#                print("  再生位置 | " + str(td) + "\n")
                print("  再生位置 | " + str(self.startPositionDict[song]) + "\n")
            print("------------------------------------------------------------------------\n\n")
        return

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
        # 順番を保ちたいのでOrderedDict
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
        '''

        参考 :
        https://qiita.com/yuuki__/items/4bc16ae439de46cd0d76
        https://www.wizard-notes.com/entry/music-analysis/compute-bpm-with-librosa

        https://librosa.github.io/librosa/generated/librosa.beat.tempo.html#librosa.beat.tempo
        https://librosa.github.io/librosa/generated/librosa.beat.beat_track.html

        '''
        for file_name,path in tqdm(zip(self.file_names,self.file_path)):
            if file_name not in self.bpm:
                self.music, self.sr = librosa.load(path)
                self._bpm, self._beatsPosition = librosa.beat.beat_track(self.music,self.sr)
                self.beatsPosition = librosa.frames_to_time(self._beatsPosition)
                self.bpm[file_name] = BPM(self._bpm, self.beatsPosition)
            else:
                continue
        return self.bpm

    def calc_chroma(self):
        '''

        参考 : https://qiita.com/namaozi/items/31ea255ecc6a04320dfc

        '''
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
        for file_name,path in tqdm(zip(self.file_names,self.file_path)):
            if file_name not in self.chroma:
                self.music, self.sr = librosa.load(path,offset=30.0, duration=30.0)
                harmonic, percussive = librosa.effects.hpss(self.music)
                self.allChroma = librosa.feature.chroma_cens(y=harmonic)
                self._chroma = np.zeros(12)
                for i in range(self.allChroma.shape[0]):
                    for j in range(self.allChroma.shape[1]):
                        self._chroma[i] += self.allChroma[i][j]
            # Chromaとコサイン類似度が最大になるスケールを調べる
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

class Map:
    """

     楽曲間類似度を算出、保持するクラス

    Attributes:
        self.songmMap (2次元配列): 曲をノードと見立てた楽曲間距離の隣接行列.
        self.songList (1次元配列): 再生順に並べられた曲名のリスト.
        self.keyDist (2次元配列): キー間の相性を距離として格納した隣接距離.

    """
    def __init__(self, songDict, param):
        """イニシャライザ

         隣接行列の生成
         最短ハミルトン路の算出、プレイリストの作成

        Args:
            songDict: 順序付きディクショナリ,[ファイル名 - ((BPM,beats),Key)]
            param: パラメーターのタプル,(BPMの重み、Keyの重み)

        Todo:
            最短ハミルトン路の高速計算を実装する

        """
        self.songMap = np.empty((len(songDict), len(songDict)))
        self.keyDist = np.empty((12,12))
        self.songDict = songDict
        self.param = param
        self.songListIndex = np.empty(len(songDict))
        self.playList = []

    def play_list(self):
        for i, s1 in enumerate(self.songDict.values()):
            for j, s2 in enumerate(self.songDict.values()):
                if i is not j:
                    self.songMap[i][j] = (abs(s1.BPM.BPM - s2.BPM.BPM) * 0.04 ) ** 1.2
                    self.songMap[i][j] += (1 - self.key_distance(s1.Key, s2.Key)) ** 1.2
                    #self.songMap[i][j] = random.random() # 完全ランダム
                else :
                    self.songMap[i][j] = 10000

        self.songDict_list = []

        for song in self.songDict.keys():
            self.songDict_list.append(song)

        for idx, songIdx in enumerate(self.songListIndex):
            if idx is 0:
                # self.songListIndex[idx] = random.randrange(len(self.songDict))
                self.songListIndex[idx] = self.songDict_list.index("Jason Sparks - Close My Eyes feat. J. Little (Original Mix).wav") # 起点となる楽曲
            else:
                li = []
                row_num = int(self.songListIndex[idx-1])
                sortedIdxArr = np.argsort(self.songMap[row_num])
                for sortedIdx in sortedIdxArr:
                    if sortedIdx not in self.songListIndex:
                        li.append(sortedIdx)
                    if len(li) >= 1:
                        break
                if len(li) is not 0:
                    self.songListIndex[idx] = random.choice(li)
                li.clear

        for songIdx in self.songListIndex:
            self.playList.append(self.songDict_list[int(songIdx)])

        return self.playList

    def key_distance(self, key1, key2):
        #　共通している音階の数の隣接行列
        self.keyDist = [[7, 2, 5, 4, 3, 6, 2, 6, 3, 4, 5, 2],
                        [2, 7, 2, 5, 4, 3, 6, 2, 6, 3, 4, 5],
                        [5, 2, 7, 2, 5, 4, 3, 6, 2, 6, 3, 4],
                        [4, 5, 2, 7, 2, 5, 4, 3, 6, 2, 6, 3],
                        [3, 4, 5, 2, 7, 2, 5, 4, 3, 6, 2, 6],
                        [6, 3, 4, 5, 2, 7, 2, 5, 4, 3, 6, 2],
                        [2, 6, 3, 4, 5, 2, 7, 2, 5, 4, 3, 6],
                        [6, 2, 6, 3, 4, 5, 2, 7, 2, 5, 4, 3],
                        [3, 6, 2, 6, 3, 4, 5, 2, 7, 2, 5, 4],
                        [4, 3, 6, 2, 6, 3, 4, 5, 2, 7, 2, 5],
                        [5, 4, 3, 6, 2, 6, 3, 4, 5, 2, 7, 2],
                        [2, 5, 4, 3, 6, 2, 6, 3, 4, 5, 2, 7]]
        return self.keyDist[self.scale.index(key1)][self.scale.index(key2)] / 7

    def printMap(self):
        print("songMap : ")
        print(self.songMap)

    def printList(self):
        print(self.playList)

class BPM(NamedTuple):
    BPM: np.float64
    beats: np.ndarray

class BPM_n_Key(NamedTuple):
    BPM: BPM
    Key: str

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

print("\n\nConversioning to wav file...")
# path
path = sys.argv[1]
# make temp directory
tmp = temp.TemporaryDirectory()
# mp3,wav save to temp file
save_ = WavSaveTmp(path)
save_.save_tmp()

# instantiation analyser
analyser = Analyse()

# bpm analyse
print("\nAnalyzing BPM...")
bpm_list = analyser.analyse_bpm()

# key analyse
print("\nAnalyzing Key...")
key_list = analyser.analyse_key()

# song_dict: [ファイル名 - ((BPM,beats),Key)]の順序付き辞書
song_dict = OrderedDict()
for k, tp in bpm_list.items():
    song_dict[k] = BPM_n_Key(tp, key_list[k])

# 楽曲間類似度のマップを作成
print("\nAnalyzing music between similarity...")
Map = Map(song_dict, (1,1))

# 曲順のリストを作成
print("\nDetermining playback order...")
play_list = Map.play_list()
Map.printMap()
print("\nsongDict:")
for key,item in song_dict.items():
    print("\n" + key + " : ")
    print("BPM : " + str(item.BPM.BPM))
    print("beats[0] : " + str(item.BPM.beats[0]))
    print("beats[15] : " + str(item.BPM.beats[15]))
    print("beats[-16] : " + str(item.BPM.beats[-16]))
    print("beats[-1] : " + str(item.BPM.beats[-1]))
    print("Key : " + item.Key)

# instantiation player
mixer = Mixer(song_dict,play_list)
# MIXを作成
print("\nCreating Mix...")
mixer.MIX()

# clean up temp directory
tmp.cleanup()
