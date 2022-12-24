import csv


#放置每一行資料
rows = []


# 開啟 CSV 檔案
with open('./data/train.csv', newline='') as csvfile:
    
    # 讀取 CSV 檔案內容
    reader = csv.reader(csvfile)

    #避免輸入第一行的標題
    first = True

    # 以迴圈輸出每一列
    for row in reader:
        if(first==True):
            first = False
        else:
            rows.append(row)

with open('./preprocessed_train.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    # 寫入一列資料
    writer.writerow(['userID', 'itemID', 'rating']) #courseID->itemID

    for row in rows:
        userID = row[0]
        courseID_list = str(row[1]).split()
        rating = 100
        for courseID in courseID_list:
            writer.writerow([userID, courseID, rating])
            rating -= 2 #每一堂後選的課程少一點分數

    print("done!")