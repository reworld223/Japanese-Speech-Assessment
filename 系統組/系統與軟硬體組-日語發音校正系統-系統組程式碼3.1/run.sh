#!/bin/sh
./ngrok http 5000 &
python app.py

#在本地的port 5000連接ngrok並執行app.py