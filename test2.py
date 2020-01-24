import librosa

y, sr = librosa.load(librosa.util.example_audio_file())
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
librosa.frames_to_time(beats[:20], sr=sr)
print(beats[:20])
