from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score

import csv
import json

y_pred = []
y_true = []

temp_pred = []
with open('eval.csv', newline='') as csvfile:
    # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
    rows = csv.DictReader(csvfile)
    # 以迴圈輸出指定欄位
    for row in rows:
        #print(row["id"], row["tags"])
        temp_true = row["tags"]
        y_pred.append(temp_true.split())

test_file = './data/slot/eval.json'
temp_true = []
with open(test_file) as f:
    temp_true = json.load(f)
y_true = [sample['tags'] for sample in temp_true]
    


#print(y_pred)
#print(len(y_true))
#classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
