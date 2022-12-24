1.先使用preprocess.py來預處理資料,給予評分,注意recNEW要用userID itemID(原course_id) rating,所以這裡寫入csv就要這樣 <br>
2.使用recNEW.ipynb,從上數來第6格 bivae.fit 會開始train,gpu大約20min就能跑完。如果不想train,就直接跳到load那邊往下點就好,最後一格是生成test_seen的預測 <br>
3.最後用evaluate.py來評估,裡面引入老師給的average_precision.py,open csv的檔名得自行確認(可能有和你的不同或是你想預測test_seen那裏的csv)
