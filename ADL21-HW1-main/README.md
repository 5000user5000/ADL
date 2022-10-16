# Sample Code for Homework 1 ADL NTU 111 Fall

  

## Environment

```shell

# If you have conda, we recommend you to build a conda environment called "adl"

make

# otherwise

pip install -r requirements.txt

```


  

## Preprocessing

```shell

# To preprocess intent detectiona and slot tagging datasets

bash preprocess.sh

```

  

## Intent detection

```shell

python train_intent.py

```

因為args.ckpt_dir用在torch.save()會錯誤,只好先存在當前路徑,並使用以下linux語法移動到該有的位置(/ckpt/intent)

```
mv ./model_state_dict.pt ./ckpt/intent
```

## Slot filling

```shell

python train_slot.py

```

同Intent detection

```
mv ./model_state_dict_slot.pt ./ckpt/slot
```


## Download model

```shell

bash download.sh

```
