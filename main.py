#coding:utf-8

"""

     * 楽曲間類似度の計算 -> リストを作って再生
     * log2(曲数)をコマンドライン引数で指定して解析を行う


"""

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
from typing import NamedTuple
import random
from tqdm import tqdm
import datetime

class Mix:
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
        self.songDict = songDict
        if playList is None:
            self.wav_name = os.listdir(wavAudioPath)
            self.playList = self.wav_name
        else:
            self.playList = playList
        print("\nMixをinitした")
        print("self.playList : ")
        pprint(self.playList)
        print("\n合計" + str(len(self.playList)) + "曲")
        print("2^実行時引数 : " + str(songNum))

    def play(self):
        for i in self.playList:
            self.song = dub.AudioSegment.from_wav(i)
            print(type(self.song))
            play(self.song)
        self.play()

    def MIX(self):
        self.silenceDuration = self.songDict[self.playList[0]].BPM.beats[15] + dub.AudioSegment.from_wav(self.playList[-1]).duration_seconds - self.songDict[self.playList[-1]].BPM.beats[-1]
        for song in self.playList:
            self.silenceDuration += self.songDict[song].BPM.beats[-1]
            self.silenceDuration -= self.songDict[song].BPM.beats[15]
        self.mixDown = dub.AudioSegment.silent(duration=(self.silenceDuration + 1) * 1000)
        # 拍位置を合わせて楽曲をオーバーレイする
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
            self.prevSongEndBeatPosition = self.startPosition + self.songDict[song].BPM.beats[-1]
        return

    def export(self):
        #chunks = split_on_silence(self.mixDown, min_silence_len=3000, silence_thresh=-40, keep_silence=1000)
        self._exPath = "/Users/hmori/ChromagramSample3/MixDown"
        if not os.path.isdir(self._exPath):
            os.makedirs(self._exPath)
        self.mixDown.export(self._exPath + "/" + "MixDown😈.mp3", format="mp3")
        #chunks[0].export(self._exPath + "/" + "MixDown😈.mp3", format="mp3")
        print("\nSuccessful export!🎉🍺 : " + self._exPath + "/" + "MixDown😈.mp3")
        return

class Analyse:

    def __init__(self):
        self.dir_path = wavAudioPath
        self.wav_path = self.dir_path + '/*.wav'
        self.file_names = glob(self.wav_path)
        del self.file_names[songNum:]
        print("解析するファイル")
        pprint(self.file_names)
        self.file_path = self.file_names
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
                else :
                    self.songMap[i][j] = 10000

        self.songDict_list = []

        for song in self.songDict.keys():
            self.songDict_list.append(song)

        for idx, songIdx in enumerate(self.songListIndex):
            if idx is 0:
                self.songListIndex[idx] = random.randrange(len(self.songDict))
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
        self.scale = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
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

if __name__ == "__main__" :

    commandArg = sys.argv[1]

    for index in range(int(commandArg)):

        print("\nConversioning to wav file...")

        # 曲数(指数が実行時パラメータ)
        songNum = 2 ** index

        # make wav file directory
        wavAudioPath = "/Users/hmori/ChromagramSample3/waves"

        import time
        t0 = time.time()

        # instantiation analyser
        analyser = Analyse()

        # bpm analyse
        print("\nAnalyzing BPM...")
        bpm_list = analyser.analyse_bpm()

        t1 = time.time()

        # key analyse
        print("\nAnalyzing Key...")
        key_list = analyser.analyse_key()

        t2 = time.time()

        # song_dict: [ファイル名 - ((BPM,beats),Key)]の順序付き辞書
        song_dict = OrderedDict()
        for k, tp in bpm_list.items():
            song_dict[k] = BPM_n_Key(tp, key_list[k])

        t3 = time.time()

        # 楽曲間類似度のマップを作成
        print("\nAnalyzing music between similarity...")
        Map = Map(song_dict, (1,1))

        t4 = time.time()

        # 曲順のリストを作成
        print("\nDetermining playback order...")
        play_list = Map.play_list()
        print("曲数 : " + str(len(play_list)))

        t5 = time.time()

        # instantiation player
        mixer = Mix(song_dict,play_list)
        # MIXを作成
        print("\nCreating Mix...")
        mixer.MIX()

        t6 = time.time()

        # MIXをエクスポート
        print("\nExporting Mix...")
        mixer.export()

        t7 = time.time()

        # 最終的に出力するリスト
        # [指数][曲数][BPM解析にかかった時間][キー解析にかかった時間][songDict作成にかかった時間][マップ作成にかかった時間]
        # [プレイリスト生成にかかった時間][ミックスの生成にかかった時間][ミックスの書き出しにかかった時間]
        timeList = []
        timeList.append(index)
        timeList.append(songNum)
        timeList.append(t1 - t0)
        timeList.append(t2 - t1)
        timeList.append(t3 - t2)
        timeList.append(t4 - t3)
        timeList.append(t5 - t4)
        timeList.append(t6 - t5)
        timeList.append(t7 - t6)

        print("timeList : ")
        print(timeList)
        dump_str = ",".join(map(str, timeList))

        resultFolderPath = "/Users/hmori/ChromagramSample3/dump"
        if not os.path.isdir(resultFolderPath):
            os.makedirs(resultFolderPath)
        with open(resultFolderPath + "/dump.txt",mode='a') as f:
            f.write("\n" + dump_str)

        print("\nFinish dumping results！🥳: " + resultFolderPath+ "/" + "dump.txt")
