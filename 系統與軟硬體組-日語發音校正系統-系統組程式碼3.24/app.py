from flask import Flask, request, jsonify, render_template ,redirect,url_for
import os
from pyngrok import ngrok
import datetime
UPLOADFOLDER = '/content'#儲存音檔的資料夾

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
  now = datetime.datetime.now()  #讀取當前時間
  fileName = now.strftime("%Y-%m-%d%H-%M") + ".mp3"  #以現在時間作為檔名
  global nowtime
  nowtime = fileName  #傳入全域變數中
  print(nowtime)
  return render_template('index1.html') #index1的路徑

@app.route('/', methods=['POST'])
def upload_file():
  file = request.files['AUDIO']  #從網頁抓取使用者輸入的語音
  print(type(file))
  global nowtime
  fileName = nowtime  #放入剛進入介面時產生的檔名
  file.filename = fileName  #將檔名設為現在時間
  print('file.filename:',file.filename)
  fileurl = os.path.join(UPLOADFOLDER, file.filename)  #找到檔案要存的絕對路徑
  print('fileurl:',fileurl)
  file.save(fileurl)  #儲存檔案
  return redirect(url_for('end'))  #將頁面重新導向到'/result'

@app.route('/index3.html',methods=['GET','POST'])
def index3():
    return render_template('index3.html')

@app.route('/result',methods=['GET','POST'])
def end():

  return render_template('index2.html',word_str = '87',word_dict = '')#index2的路徑

if __name__ =='__main__':
  public_url = ngrok.connect(5000)
  print("Public URL:", public_url)
  app.run(port=5000)