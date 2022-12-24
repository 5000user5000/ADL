from average_precision import *
import csv


#放置每一行資料
rows = []
train = {}
name_val = []
# 開啟 CSV 檔案
with open('./data/val_seen.csv', newline='') as csvfile:
    
    # 讀取 CSV 檔案內容
    reader = csv.reader(csvfile)

    #避免輸入第一行的標題
    first = True

    # 以迴圈輸出每一列
    for row in reader:
        if(first==True):
            first = False
        else:
            train[row[0]] = str(row[1]).split()
            name_val.append(row[0])

predict = {}
with open('./predict10Total.csv', newline='') as csvfile:
    
    # 讀取 CSV 檔案內容
    reader = csv.reader(csvfile)

    #避免輸入第一行的標題
    first = True
    
    # 以迴圈輸出每一列
    for row in reader:
        if(first==True):
            first = False
        else:
            predict[row[0]] = str(row[1]).split()


score = 0
#每一個客戶計算一次成績,之後再平均起來
for i in name_val:
    train_list = train.get(i) #取得該用戶的購買課程
    if(train_list==None):
        continue
    predict_list = predict.get(i)
    if(predict_list==None):
        continue
    score += apk(train.get(i),predict.get(i)) #dict不能直接 train[i]

score = score/len(name_val)
print(score)

#print( mapk(train,predict) ) #會錯誤