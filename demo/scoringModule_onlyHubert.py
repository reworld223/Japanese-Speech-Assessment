# ----- Imports -----
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pyarrow as pa
import librosa
import subprocess
import torchaudio
import pykakasi
from datasets import Dataset
from transformers import HubertForCTC, Wav2Vec2Processor, AutoTokenizer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks



# ----- Model Loading -----
kks = pykakasi.kakasi()
processor = Wav2Vec2Processor.from_pretrained('TKU410410103/uniTKU-hubert-japanese-asr')
hubert = HubertForCTC.from_pretrained('TKU410410103/uniTKU-hubert-japanese-asr')
hubert.config.output_hidden_states=True
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

# ----- Function Definitions -----

def acoustic_noise_suppression(input_wav_path, output_wav_path):
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model='damo/speech_frcrn_ans_cirm_16k')
    result = ans(
        input_wav_path,
        output_path=output_wav_path)
    return result

def modified_filename(file_path):
    base_name = file_path.rsplit('.', 1)[0]
    extension = file_path.rsplit('.', 1)[1]
    output_file = f"{base_name}1.{extension}"
    return output_file

def convert_to_wav(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:")
        print(e.stderr.decode('utf-8'))
        raise e

def process_waveforms(batch):
    waveform, sample_rate = torchaudio.load(batch['audio_path'])
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    # 如果 waveform 是雙聲道，需要轉單聲道。
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    # 讓 waveform的維度正確
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    batch["speech_array"] = waveform
    return batch

def asr(outputs):
    predicted_ids = torch.argmax(outputs.logits, dim=-1) 

    predicted_ids = predicted_ids.view(-1).tolist()

    decoded_output = processor.decode(predicted_ids)
    return decoded_output

class BLSTMSpeechScoring(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=1, output_size=1, embedding_dim=64, vocab_size=4000):
        super(BLSTMSpeechScoring, self).__init__()

        # 聲學特徵的 BLSTM
        self.acoustic_blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True, bidirectional=True)

        # 語言特徵（字符）的 BLSTM
        self.linguistic_blstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                         num_layers=num_layers, batch_first=True, bidirectional=True)

        # 字符的嵌入層
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 處理 BLSTM 輸出的線性層，以匹配維度
        self.acoustic_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linguistic_linear = nn.Linear(hidden_size * 2, hidden_size)

        # 串接後的最終線性層
        self.final_linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, acoustic_input, linguistic_input):
        # 聲學輸入通過 BLSTM
        acoustic_output, _ = self.acoustic_blstm(acoustic_input)

        # 將語言輸入嵌入並通過 BLSTM
        embedded_chars = self.embedding(linguistic_input)
        linguistic_output, _ = self.linguistic_blstm(embedded_chars)

        # 線性層確保維度匹配
        acoustic_features = self.acoustic_linear(acoustic_output)
        linguistic_features = self.linguistic_linear(linguistic_output)

        # 對兩輸出進行全局平均池化（GAP）
        gap_acoustic = torch.mean(acoustic_features, dim=1)
        gap_linguistic = torch.mean(linguistic_features, dim=1)

        # 串接特徵並最終評分
        concatenated_features = torch.cat((gap_acoustic, gap_linguistic), dim=1)
        
        score = self.final_linear(concatenated_features)

        return score

judge = 0.65
class Trainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def pred(self, acoustic_input, text):
        self.model.eval()
        with torch.no_grad():
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=100)
            linguistic_input = encoded_input['input_ids']
            outputs = self.model(acoustic_input, linguistic_input)
        return outputs

def make_dataframe(audio_path):
    df = pd.DataFrame(columns=['audio_path'])
    row = []
    row.append({'audio_path': audio_path})
    df = pd.DataFrame(row)
    return df

def get_acoustic_feature(batch):
    with torch.no_grad():
        processed_audios = processor(batch['speech_array'],
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=160000)
        outputs = hubert(**processed_audios)

    text = asr(outputs)
    
    transformer_hidden_states = outputs.hidden_states[:]

    # Stack transformer hidden states to have a new dimension for layers
    stacked_hidden_states = torch.stack(transformer_hidden_states)

    # Average across layers dimension (0) while keeping sequence_length
    overall_avg_hidden_state = torch.mean(stacked_hidden_states, dim=0)

    return overall_avg_hidden_state, text

# Model Loading and Initialization
blstm = BLSTMSpeechScoring()
model_save_path = "./BLSTMSpeechScoring_TKU.pth"
blstm.load_state_dict(torch.load(model_save_path))
blstm.eval()
trainer = Trainer(blstm, tokenizer)

def scoring(file_path):
    converted_file_path = modified_filename(file_path)
    convert_to_wav(input_path=file_path, output_path=converted_file_path)
    
    output_file = modified_filename(converted_file_path)
    acoustic_noise_suppression(input_wav_path=converted_file_path, output_wav_path=output_file)
    
    df = make_dataframe(output_file)
    dataset = Dataset.from_pandas(df)
    dataset_array = dataset.map(process_waveforms, remove_columns=['audio_path'])
    acoustic_input, text = get_acoustic_feature(dataset_array)
    score = trainer.pred(acoustic_input, text)

    score = 1 if score > judge else 0

    return score*100
    # return f"{float(score):.2f}"

