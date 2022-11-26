# Code for Homework 3 ADL NTU 111 Fall

# environment

```shell

pip install -r requirements.txt

git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
cd ..


```

# train

```shell

bash train.sh

```
注意資料要放在 ./data 資料夾中
輸出預設是放在 ./ckpt

# predict

```shell

bash run.sh
python postprocess.py  --output ans.jsonl

```
注意資料要放在 ./data 資料夾中，注意下載的模型要放在 ./ckpt 資料夾中，預設的batch數為8,如果過大可自行調小
run.sh 生成的資料預設是放在 ./generated_predictions.txt ,  postprocess會做後處理把這txt轉成jsonl，預計放在 ./ans.jsonl (這裡名稱可自行設定)

# evaluate

```shell

python ./ADL22-HW3/eval.py -r ./data/public.jsonl -s ans.jsonl

```
會用到網路來下載data.zip
-r 解答 -s 提交的答案 , 去比對
-s ans.jsonl這部分如果predict改用其他名稱就要換那名稱

# download

```shell

bash download.sh

```
注意,如果有檔案無法下載,例如出現access error的話,記得要pip install --upgrade --no-cache-dir gdown , 並且再bash download.sh一次

