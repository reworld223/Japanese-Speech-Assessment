import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, HubertModel, AutoConfig, AutoTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pandas as pd
import librosa
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import torchaudio
import pykakasi

whisperProcessor = WhisperProcessor.from_pretrained("jakeyoo/whisper-medium-ja")
model = WhisperForConditionalGeneration.from_pretrained("jakeyoo/whisper-medium-ja")
model.config.forced_decoder_ids = None


# def speech_file_to_array_fn(batch):
#     # 使用 librosa 載入音頻檔案，並將其轉換為陣列
#     speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
#     batch["array"] = speech_array
#     return batch

def process_waveforms(batch):
    
    waveform, sample_rate = torchaudio.load(batch['audio_path'])

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 如果 waveform 是雙聲道，需要轉單聲道。給 4GE用
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)

    # 讓 waveform的維度正確
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    batch["speech_array"] = waveform

    return batch

def whisper(audio_paths):
    kakasi = pykakasi.kakasi()
    kakasi.setMode("J","H")
    kakasi.setMode("K","H")
    kakasi.setMode("r","Hepburn")
    conv = kakasi.getConverter()
    
    # 建立一個包含音頻路徑的 DataFrame
    test_dataset = {"audio_path":[audio_paths]}
    test_dataset = pd.DataFrame(test_dataset)
    # 將 DataFrame 轉換為 Dataset 對象
    test_dataset = Dataset(pa.Table.from_pandas(test_dataset))
    # 將每個音頻檔案轉換為陣列
    test_dataset = test_dataset.map(process_waveforms)
    # 處理音頻數據以獲取模型的輸入特徵
    input_features = whisperProcessor(test_dataset['speech_array'], sampling_rate=16_000, return_tensors="pt").input_features

    # 使用模型生成預測結果，並關閉梯度計算以加快速度
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # 解碼預測結果以獲得文字轉寫
    transcription = whisperProcessor.batch_decode(predicted_ids, skip_special_tokens=True)

    # 處理轉寫結果以去除不需要的部分
    text = []
    for index in range(len(transcription)):
        transcription_len = len(transcription[index])
        text.append(transcription[index][0:transcription_len-1])

    result = conv.do(text[0])
    # 返回處理後的文字結果
    return result  # 返回文本


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

        # 確保在串接之前批量大小相同，怕音檔和文字的數量不對，取完整的
        if gap_acoustic.size(0) != gap_linguistic.size(0):
            min_batch_size = min(gap_acoustic.size(0), gap_linguistic.size(0))
            gap_acoustic = gap_acoustic[:min_batch_size, :]
            gap_linguistic = gap_linguistic[:min_batch_size, :]

        # 串接特徵並最終評分
        concatenated_features = torch.cat((gap_acoustic, gap_linguistic), dim=1)
        concatenated_features = F.relu(concatenated_features)
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


processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

config = AutoConfig.from_pretrained("rinna/japanese-hubert-base", output_hidden_states=True)
hubert = HubertModel.from_pretrained("rinna/japanese-hubert-base", config=config)


def make_dataframe(audio_path):
    row = []
    text = whisper(audio_path)
    # text = 'さい'
    print(text)
    row.append({'audio_path': audio_path, 'text': text})
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

    # all layers
    transformer_hidden_states = outputs.hidden_states[:]

    # Stack transformer hidden states to have a new dimension for layers
    stacked_hidden_states = torch.stack(transformer_hidden_states)

    # Average across layers dimension (0) while keeping sequence_length
    overall_avg_hidden_state = torch.mean(stacked_hidden_states, dim=0)

    return overall_avg_hidden_state

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")
blstm = BLSTMSpeechScoring()

model_save_path = "./BLSTMSpeechScoring.pth"
blstm.load_state_dict(torch.load(model_save_path))

blstm.eval()

trainer = Trainer(blstm, tokenizer)


def scoring(file_path):
    df = make_dataframe(file_path)
    dataset = Dataset.from_pandas(df)
    print(dataset)
    dataset_array = dataset.map(process_waveforms, remove_columns=['audio_path'])
    acoustic_input = get_acoustic_feature(dataset_array)
    text = list(df['text'])
    score = trainer.pred(acoustic_input, text)

    score = 1 if score > judge else 0

    return score*100
    # return f"{float(score):.2f}"
