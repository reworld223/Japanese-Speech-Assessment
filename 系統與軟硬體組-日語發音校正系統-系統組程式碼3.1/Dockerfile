#以python3.8為基底
FROM python:3.8
#工作的目錄
WORKDIR /app
#將本地端的檔案複製進image
ADD . /app
#安裝所需要的套件
RUN pip install -r requirements.txt
RUN pip install --upgrade tensorflow && pip install librosa && pip install pandas && pip install joblib && pip install seaborn && pip install SpeechRecognition && pip install fastdtw && pip install vosk && pip install IPython
RUN apt-get update && apt-get install apt-file -y && apt-file update && apt-get install -y ffmpeg
#下載支援linux的ngrok壓縮檔
RUN wget --no-check-certificate https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
#解壓縮
RUN tar zxvf ngrok-v3-stable-linux-amd64.tgz
#加入個人的ngrok authtoken
RUN ./ngrok config add-authtoken 2DhccdBAYyltWePOATEEfIbpPk3_5DLX6tqwwnXqPNVNLx2q1
#改變run.sh的權限
RUN ["chmod","+x","./run.sh"]
#執行run.sh
CMD ./run.sh