import os
import pathlib
from flask import Flask, url_for, redirect,  render_template, request,g
import filetype
import datetime
import pytest
from pydub import AudioSegment
 
SRC_PATH =  pathlib.Path(__file__).parent.absolute()  #此檔案的絕對路徑
UPLOAD_FOLDER = os.path.join(SRC_PATH,  'static', 'uploads')  #儲存上傳資料的資料夾路徑
nowtime = None  #記錄使用者輸入檔案的檔名

app = Flask(__name__)

#初始介面
@app.route('/', methods=['GET'])  
def index():
    now = datetime.datetime.now()  #讀取當前時間
    fileName = now.strftime("%Y-%m-%d_%H-%M") + ".mp3"  #以現在時間作為檔名
    global nowtime
    nowtime = fileName  #傳入全域變數中
    return render_template('index1.html')  #顯示錄音介面

#進行存檔
@app.route('/', methods=['GET','POST'])
def upload_file():
    file = request.files['AUDIO']  #從網頁抓取使用者輸入的語音
    global nowtime
    fileName = nowtime  #放入剛進入介面時產生的檔名
    file.filename = fileName  #將檔名設為現在時間
    fileurl = os.path.join(UPLOAD_FOLDER, file.filename)  #找到檔案要存的絕對路徑
    file.save(fileurl)  #儲存檔案
    return redirect(url_for('end'))  #將頁面重新導向到'/result'

@app.route('/index3.html',methods=['GET','POST'])
def index3():
    return render_template('index3.html')

#顯示結果
@app.route('/result',methods=['GET','POST'])
def end():
    global nowtime
    fileName = nowtime  #放入剛進入介面時產生的檔名
    fileurl = os.path.join(UPLOAD_FOLDER, fileName)  #獲取檔案儲存的位置路徑
    if  request.values.get("result") == '獲取結果' :  #如果使用者按下'獲取結果'按鈕
        result = pytest.load_file(os.path.realpath(fileurl))  #將使用者的語音載入
        if result == "nothing" :   #若得到的結果nothing
            word_str = "請重新測試"  #要顯示的輸入字詞為'請重新測試'
            return render_template('index2.html',word_str = word_str,word_dict='')  
            #回傳輸入字詞和結果到結果頁面
        else :
            result_list = pytest.s2t_predict(result)  #讀取判斷結果
            word_str = pytest.word_str()  #讀取判斷字串
            word_list = pytest.word_list()  #讀取分隔的字串
            word_dict = zip(word_list,result_list)  #合併分隔字串和評分字串
            return render_template('index2.html',word_str = word_str,word_dict = dict(word_dict))  
            #回傳輸入字詞和結果到結果頁面
    else:
        return render_template('index2.html',word_str = '',word_dict = '')
        #回傳輸入字詞和結果到結果頁面
    
if __name__ == "__main__":
    app.run(debug=True)