# 日文發音評分系統

---
# Demo
![demo影片](demo.gif)

---
### 專題概述
本專題旨在開發一套自動化的日文語音評分系統，幫助學習者提升發音準確性，同時為教師提供一個高效的教學輔助工具。這個系統使用最新的自監督學習模型，從語音中直接提取特徵並供模型評估發音，以支持日語學習者的自我提升。

---
### 系統架構

- **前端**: 使用 HTML, CSS, JavaScript 實現使用者交互界面。
- **後端**: 使用 Flask 框架處理 HTTP 請求，並透過模型進行音檔降噪、語音分析和評分。

---
### 技術棧
- **前端**：HTML, CSS, JavaScript
- **後端**：Python, Flask
- **深度學習**：Pytorch, wav2vec 2.0, HuBERT, BLSTM

---
### 資料集
- [Common Voice 11.0 Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/ja)
- 從[淡江大學日文系](https://www.tfjx.tku.edu.tw/japanese/)收集的資料集

---
### 模型訓練與評估
- SSL模型可到 [Hugging Face](https://huggingface.co/TKU410410103)上看到詳細過程與評估
- 評分模型可到[對應的程式碼、evaluate](/AI組/ScoringModule)查看