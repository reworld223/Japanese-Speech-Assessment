from pydub import AudioSegment
import speech_recognition as sr

# 轉換 .m4a 到 .wav 格式
input_path = "/content/drive/MyDrive/大學專題/OneDrive_1_2023-8-14/A/音檔3.m4a"
output_path = "output_file.wav"
audio = AudioSegment.from_file(input_path, format="m4a")
audio.export(output_path, format="wav")

# 創建一個語音辨識器實例
recognizer = sr.Recognizer()

# 載入 .wav 檔案
with sr.AudioFile(output_path) as source:
    audio = recognizer.record(source)  # 將音訊讀取為語音對象

try:
    # 使用 Google Web Speech API 將語音轉換為文字
    text = recognizer.recognize_google(audio, language="ja-JP")  # 轉換為日文
    print("轉換結果:", text)
except sr.UnknownValueError:
    print("無法識別語音")
except sr.RequestError as e:
    print(f"發生錯誤: {e}")