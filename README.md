# 日文發音評分系統

---
## DEMO
![demo影片](demo.gif)

---
### 專題概述
本專題旨在開發一套自動化的日文語音評分系統，幫助學習者提升發音準確性，同時為教師提供一個高效的教學輔助工具。這個系統使用最新的自監督學習模型，從語音中直接提取特徵並供模型評估發音，以支持日語學習者的自我提升。

---
### 系統架構

- **前端**：
  - **技術**: HTML, CSS, JavaScript
  - **作用**: 實現使用者交互界面。

- **後端**：
  - **技術**: Python, Flask
  - **作用**: 處理 HTTP 請求，並透過模型進行音檔降噪、語音分析和評分。

- **深度學習**：
  - **框架**: PyTorch
  - **模型**：
    - [HuBERT](https://huggingface.co/TKU410410103/uniTKU-hubert-japanese-asr): 用於語音特徵提取。
    - [ReazonSpeech_v2](https://github.com/reazon-research/ReazonSpeech): 用於語音辨識。
    - [FRCRN](https://github.com/alibabasglab/FRCRN): 提供音檔降噪。

---
### 資料集
- [**Common Voice 11.0 Dataset**](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/ja)：用於微調 SSL 模型，以提高對日語語音的特徵提取能力。
- **從[淡江大學日文系](https://www.tfjx.tku.edu.tw/japanese/)收集的資料集**：專門用於訓練和驗證發音評分模型，這些資料由日文系教授評估，以確保評分的準確性和可靠性。

---
### 模型訓練與評估
- **SSL 模型**：用於語音特徵提取，訓練和評估細節見 [Hugging Face](https://huggingface.co/TKU410410103)。
- **評分模型**：至[對應的訓練或評估程式碼](/AI組/ScoringModule)查看詳細過程。

---
### 團隊
- **林聿朔**：
  - 負責語音增量、特徵提取、模型訓練、程式整合。
  - 報告撰寫。
  
- **施吉益**：
  - 負責語音辨識、前後端串接、音檔切割。
  - PPT 製作。

- **陳緯榛**：
  - 負責聲音降噪前處理。
  - 海報和 PPT 製作。

- **蘇柏修**：
  - 負責網頁相關架構和設計。
  - 報告撰寫和 PPT 製作。
  
- **吳天宇**：
  - 負責頁面優化和前端系統維護。

---
