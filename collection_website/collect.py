# ----- Imports -----
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pyarrow as pa
import librosa
import subprocess
import torchaudio
import pykakasi
from datasets import Dataset
from reazonspeech.nemo.asr import transcribe, audio_from_path, load_model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import difflib
import time
import os
from transformers import AutoTokenizer
# ----- Model Loading -----
kks = pykakasi.kakasi()
model = load_model()
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

# ----- Function Definitions -----

def wait_for_file(file_path, timeout=3):
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"The file {file_path} does not exist after waiting for {timeout} seconds.")
        time.sleep(0.5) 

def convert_to_wav(input_path, output_path, retries=5):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        output_path.replace('.mp3', '.wav')
    ]
    attempt = 0

    while attempt < retries:
        try:
            time.sleep(0.3)
            subprocess.run(command, check=True, capture_output=True)
            print("Conversion succeeded")
            wait_for_file(output_path)
            return
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed on attempt {attempt + 1}:")
            print(e.stderr.decode('utf-8'))
            attempt += 1
            if attempt < retries:
                print("Retrying...")
            else:
                print("All retry attempts failed")
                raise e

def modified_filename(file_path):
    base_name = file_path.rsplit('.', 1)[0]
    extension = 'wav'
    output_file = f"{base_name}1.{extension}"
    return output_file


def acoustic_noise_suppression(input_wav_path, output_wav_path):
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model='damo/speech_frcrn_ans_cirm_16k')
    result = ans(
        input_wav_path,
        output_path=output_wav_path)
    return result

def detect_audio_features(audio_file, energy_threshold=0.1, amplitude_threshold=0.1):
    # Load the audio file once
    y, sr = librosa.load(audio_file, sr=None)

    # Calculate frame energy
    frame_length = int(0.025 * sr)  # 25ms frame length
    hop_length = int(0.010 * sr)    # 10ms hop length
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(np.square(frames), axis=0)
    avg_energy = np.mean(energy)

    # Calculate maximum amplitude
    max_amplitude = np.max(np.abs(y))

    # Print the results
    print("Average Energy:", avg_energy)
    print("Maximum Amplitude:", max_amplitude)

    # Return the checks as a tuple
    return (avg_energy > energy_threshold) and (max_amplitude > amplitude_threshold)

def asr(path):
    audio = audio_from_path(path)
    ret = transcribe(model, audio)
    result = kks.convert(ret.text)
    return result[0]['hira']

def get_most_similar(predicted_word):
    correct_words = ['わたし', 'わたしたち', 'あなた', 'あのひと', 'あのかた', 'みなさん', 
                     'せんせい', 'きょうし', 'がくせい', 'かいしゃいん','しゃいん', 
                     'ぎんこういん', 'いしゃ', 'けんきゅうしゃ', 'エンジニア', 'だいがく', 
                     'びょういん', 'でんき', 'だれ', 'どなた', '～さい', 'なんさい', 'おいくつ']
    # 初始化最高相似度和對應的單字
    highest_similarity = 0.0
    most_similar_word = predicted_word
    
    # 遍歷正確的單字列表，比較相似度
    for word in correct_words:
        # 使用SequenceMatcher計算相似度
        similarity = difflib.SequenceMatcher(None, predicted_word, word).ratio()
        
        # 更新最高相似度和最相似的單字
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_word = word
    print(f'most similar word: ', most_similar_word, 'highest_similarity: ', highest_similarity)
    if highest_similarity < 0.4:
        return '單字未被收納'
    return most_similar_word

def make_dataframe(audio_path):
    row = []
    text = asr(audio_path)
    print('asr text: ', text)
    row.append({'text': text})
    df = pd.DataFrame(row)
    return df

def collect(file_path):
    converted_file_path = modified_filename(file_path) # 新的名稱.wav
    convert_to_wav(input_path=file_path, output_path=converted_file_path) # converted_file_path 給予新的名稱音檔
    
    output_file = modified_filename(converted_file_path) # 新的名稱(降噪).wav
    
    acoustic_noise_suppression(input_wav_path=converted_file_path, output_wav_path=output_file) # output_file 降噪音檔
    
    if(detect_audio_features(output_file)):
        df = make_dataframe(output_file)
        dataset = Dataset.from_pandas(df)
        text = list(df['text'])
        similar_text = get_most_similar(text[0][:-1])
        if similar_text == '單字未被收納':
            return 0, similar_text
        
        # 確認目標資料夾存在，不存在則創建
        target_dir = os.path.join("audio", similar_text)
        os.makedirs(target_dir, exist_ok=True)
        
        # 計算目標資料夾中的文件數量
        num_files = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
        
         # 根據目標資料夾的名字和文件數量命名
        new_output_file = os.path.join(target_dir, f"{similar_text}_{num_files + 1}.wav")
        os.rename(output_file, new_output_file)
        
        os.remove(file_path)
        os.remove(converted_file_path)
        
        return 100, similar_text
    
    return -1, None
