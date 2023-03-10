# Code for Homework 3 ADL NTU 111 Fall

# environment

```shell
pip install -r requirements.txt

git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
cd ..
```
這個是我們評估模型所需要的檔案

# train

```shell
bash train.sh
```
注意資料要放在 ./data 資料夾中
輸出預設是放在 ./ckpt

# predict

```shell
bash run.sh path_of_input_file path_of_output_dir
// 特別注意:如果跑的時候出現 --xxx : command not found 就輸入 sed -i 's/\r$//' run.sh 
// 然後最好是 bash run.sh ./data/public.jsonl ./ 這樣的方式
python postprocess.py  --output ans.jsonl

```
資料(jsonl)位置請放在 ./data 資料夾中會比較好，注意下載的模型要放在 ./ckpt 資料夾中(download.sh時預設會放在那)，預設的batch數為8,如果過大可自行調小
run.sh 生成的資料預設是放在 path_of_output_dir/generated_predictions.txt ,  
注意如果生成的資料不是放在./generated_predictions.txt (預設路徑),postprocess那裏就要加上--raw_preds path_of_output_dir/generated_predictions.txt
postprocess會做後處理把這txt轉成jsonl，預計放在 ./ans.jsonl (這裡位置和名稱可自行設定)
strategy可以使用最優的num_beams=8或是不用,都會過baseline。
順帶一題,run.sh是不用do_sample的參數的,當有使用top_p或top_k或temperature時就會設定成true。
p.s. 最後發現要直接生成jsonl，也就是把轉jsonl的code納入總體中。


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
注意,如果檔案下載後, 跑run.sh出現沒有./ckpt/config.json之類的報錯,可以用ls ./ckpt確認,如果有混進'/r'之類的符號就sed -i 's/\r$//' download.sh後把ckpt裡的檔案刪掉(注意ckpt檔案夾要留著),再跑一次bash download.sh
還有得先有空的ckpt檔案夾,原則上我應該有附在作業中,就不用再創一個。'/r'的錯誤應該也清過了。

附註:本作業沒有plot figure的code,因為我是每500steps的ckpt讀取後tw_rouge計算一次,並把數據點用到excel上作圖。

## 作業公告
[ADL2022-HW3](https://docs.google.com/presentation/d/11pV5rM4-pxy7Aam5wZwaXHDNIuFEthdhlEXNXLDuWxc/edit#slide=id.p)
