from flask import Flask, request, jsonify, render_template, url_for, redirect, send_from_directory
from werkzeug.utils import secure_filename 
from datetime import datetime
import os
from pyngrok import ngrok
import collect

UPLOAD_FOLDER = './audio'#儲存音檔的資料夾
AUDIO_DIRECTORY = 'templates/audio'  #儲存音檔的資料夾

app = Flask(__name__, template_folder='./templates')

@app.route("/", methods=['GET'])
def index():
    now = datetime.now()  #讀取當前時間
    fileName = now.strftime("%Y-%m-%d%H-%M%S") + ".mp3"  #以現在時間作為檔名
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
    fileurl = os.path.join(UPLOAD_FOLDER, file.filename)  #找到檔案要存的絕對路徑
    print('fileurl:',fileurl)
    file.save(fileurl)  #儲存檔案
    return redirect(url_for('end'))  #將頁面重新導向到'/result'

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    # 返回音檔
    return send_from_directory(AUDIO_DIRECTORY, filename)

@app.route('/index3.html',methods=['GET','POST'])
def index3():
    return render_template('index3.html')

@app.route('/index4.html',methods=['GET','POST'])
def index4():
    return render_template('index4.html')

@app.route('/result',methods=['GET','POST'])
def end():
    global nowtime
    fileName = nowtime  #放入剛進入介面時產生的檔名
    fileurl = os.path.join(UPLOAD_FOLDER, fileName)  #獲取檔案儲存的位置路徑
    result, asr_text = collect.collect(fileurl)
    if result == -1:
        return render_template('index4.html',word_str=result)#index4的路徑
    else:
        return render_template('index2.html',word_str=result)#index2的路徑

if __name__=='__main__':
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)
    app.run(debug=True, use_reloader=False)
