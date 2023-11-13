import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_file1 = '/content/drive/MyDrive/大學專題/切割音檔/音檔1/vocals.wav'
audio_file2 = '/content/drive/MyDrive/大學專題/OneDrive_1_2023-8-14/A/音檔1.m4a'
y1, sr1 = librosa.load(audio_file1)
y2, sr2 = librosa.load(audio_file2)
# 生成音訊波形圖

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y1, sr=sr1)
plt.grid()
plt.title('Noise reduction Audio')
plt.xlabel('time(second)')
plt.ylabel('Amplitude')
plt.ylim(-0.6, 0.6)

plt.subplot(2, 1, 2)
librosa.display.waveshow(y2, sr=sr2)
plt.title('Audio')
plt.grid()
plt.xlabel('time(second)')
plt.ylabel('Amplitude')
plt.ylim(-0.6, 0.6)

plt.tight_layout()
plt.show()