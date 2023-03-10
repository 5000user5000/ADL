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

## Slot filling

```shell

python train_slot.py

```


## Download model

```shell

bash download.sh

```

## 作業公布
[Applied Deep Learning Homework 1 - Fall 111](https://docs.google.com/presentation/d/1X53LNiayg7md9nQ9qNP7H_cZRl6tTxqVg7UpZmys8kc/edit#slide=id.g8130877143_0_0)
