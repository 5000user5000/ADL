# Sample Code for Homework 2 ADL NTU 111 Fall

  

## Environment

```shell

# If you have conda, we recommend you to build a conda environment called "adl"

make

# otherwise

pip install -r requirements.txt

```
train的參數都是預設好的,train和valid和test資料都要放在./data的資料夾下


## Context Select

```shell

python cs_train.py

```

## Question Answer

```shell

python qa_trainer.py

```


## Inference

```shell

python inference.py

```
Inference注意,即使使用先前下載好的pretrain model(我自己的),前面似乎還是會下載東西到cache,所以無法做到完全斷網
開始run prediction時就可以斷網了

## Download model

```shell

bash download.sh

```
注意,download模型之前,得先要有./ckpt/CS_pretrain 和./ckpt/QA_pretrain 等兩個資料夾,因為模型會下載到那裏,否則會出錯(原則上我的檔案夾應該已經放上這兩個空的資料夾)
