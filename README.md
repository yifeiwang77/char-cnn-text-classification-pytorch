## Introduction

**This project is currently under construction!**

This is an implementation of several recent adavances in character-level text classification (particular with CNN), including 

* (CharCNN)        Xiang Zhang, Junbo Zhao, Yann LeCun.(2015) [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). 
* (CharCRNN)       Yijun Xiao, Kyunghyun Cho. (2016) [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://arxiv.org/pdf/1602.00367).
* (VDCNN)          Conneau, A., Schwenk, H., Barrault, L., & Lecun, Y. (2017). [Very deep convolutional networks for text classification.](http://www.aclweb.org/anthology/E17-1104).
* (ShadowWideCNN)  Le, H. T., Cerisara, C., & Denis, A. (2017). [Do Convolutional Networks need to be Deep for Text Classification?](https://arxiv.org/pdf/1707.04108.pdf).

And one (**CharCHRNN**) that is created by me that modifies the CharCRNN a little by adding a highway layer between conv layers and rnn layer.

My inital motivation is to compare my simple idea CHRNN with several baselines here. But the project can be extended to more models to provide a benchmark for text classification tasks. So any improvement or extention to the codes is warmly welcomed.

The main framework of this code follows https://github.com/srviest/char-cnn-pytorch, which imeplements Zhang et al.'s CharCNN. But several important modifications are made.

### **Caution**: 

* This is only a homework-level project to carry out the experiments, and no more comments and beautifying will be added. Thus, there could cause some puzzling when you read the code.
* I don't run all the expriments possible due to the limit of time.

## Requirement
* python 2
* pytorch > 0.2
* numpy
* termcolor


## Datasets:
| Dataset                | Classes | Train samples | Test samples | source |
|------------------------|:---------:|:---------------:|:--------------:|:--------:|
| Imdb                   |    2    |    25 000     |     25 000   |[link](https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/imdb_csv.tar.gz)|
| AG’s News              |    4    |    120 000    |     7 600    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Sogou News             |    5    |    450 000    |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| DBPedia                |    14   |    560 000    |    70 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Full       |    5    |    650 000    |    50 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yahoo! Answers         |    10   |   1 400 000   |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Full     |    5    |   3 000 000   |    650 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Polarity |    2    |   3 600 000   |    400 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|



## Train
```
python train_CNN.py -h
```

You will get:

```
Character-level CNN text classifier

optional arguments:
  -h, --help            show this help message and exit
  --train-path DIR      path to training data csv
  --val-path DIR        path to validating data csv

Learning options:
  --lr LR               initial learning rate [default: 0.0005]
  --epochs EPOCHS       number of epochs for train [default: 200]
  --batch-size BATCH_SIZE
                        batch size for training [default: 128]
  --max-norm MAX_NORM   Norm cutoff to prevent explosion of gradients
  --optimizer OPTIMIZER
                        Type of optimizer. SGD|Adam|ASGD are supported
                        [default: Adam]
  --class-weight        Weights should be a 1D Tensor assigning weight to each
                        of the classes.
  --dynamic-lr          Use dynamic learning schedule.
  --milestones MILESTONES [MILESTONES ...]
                        List of epoch indices. Must be increasing.
                        Default:[5,10,15]
  --decay-factor DECAY_FACTOR
                        Decay factor for reducing learning rate [default: 0.5]

Model options:
  --alphabet-path ALPHABET_PATH
                        Contains all characters for prediction
  --l0 L0               maximum length of input sequence to CNNs [default:
                        1014]
  --shuffle             shuffle the data every epoch
  --dropout DROPOUT     the probability for dropout [default: 0.5]
  -kernel-num KERNEL_NUM
                        number of each kind of kernel
  -kernel-sizes KERNEL_SIZES
                        comma-separated kernel size to use for convolution

Device options:
  --num-workers NUM_WORKERS
                        Number of workers used in data-loading
  --cuda                enable the gpu

Experiment options:
  --verbose             Turn on progress tracking per iteration for debugging
  --continue-from CONTINUE_FROM
                        Continue from checkpoint model
  --checkpoint          Enables checkpoint saving of model
  --checkpoint-per-batch CHECKPOINT_PER_BATCH
                        Save checkpoint per batch. 0 means never save
                        [default: 10000]
  --save-folder SAVE_FOLDER
                        Location to save epoch models, training configurations
                        and results.
  --log-config          Store experiment configuration
  --log-result          Store experiment result
  --log-interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 1]
  --val-interval VAL_INTERVAL
                        how many steps to wait before vaidation [default: 200]
  --save-interval SAVE_INTERVAL
                        how many epochs to wait before saving [default:1]
```


```
python train.py
```
You will get:

```
Epoch[8] Batch[200] - loss: 0.237892  lr: 0.00050  acc: 93.7500%(120/128))
Evaluation - loss: 0.363364  acc: 89.1155%(6730/7552)
Label:   0      Prec:  93.2% (1636/1755)  Recall:  86.6% (1636/1890)  F-Score:  89.8%
Label:   1      Prec:  94.6% (1802/1905)  Recall:  95.6% (1802/1884)  F-Score:  95.1%
Label:   2      Prec:  85.6% (1587/1854)  Recall:  84.1% (1587/1888)  F-Score:  84.8%
Label:   3      Prec:  83.7% (1705/2038)  Recall:  90.2% (1705/1890)  F-Score:  86.8%
```

## Test
If you has construct you test set, you make testing like:

```
python test.py --test-path='data/ag_news_csv/test.csv' --model-path='models_CharCNN/CharCNN_best.pth.tar'
```
The model-path option means where your model load from.

## Model Size
| model                | model_size|
|:--------------------:|:---------:|
|CNN                   |11.34 M    |
|CRNN                  |0.32 M     |
|Shadow and Wide CNN   |6.15 M     |
|CHRNN                 |0.35 M     |
|VDCNN                 |68.88 M    |


## Experiments:
Results are reported as follows:  (i) / (ii)
 - (i): Test set accuracy reported by the paper  
 - (ii): Test set accuracy reproduced here  

|                                 | imdb |       ag_news  |     sogu_news     |      db_pedia      | yelp_polarity | yelp_review   | yahoo_answer | amazon_review | amazon_polarity |
|:-------------------------------:|:----:|:--------------:|:-----------------:|:------------------:|:-------------:|:-------------:|:------------:|:-------------:|:---------------:|
|CNN small                        |      | 84.35 / 87.10  | 91.35 / 93.53     | 98.02 / 98.15      |               |               |              |               |                 |
|VDCNN (9 layers, k-max-pooling)  |      | 90.17 / 89.22  | 96.30 / 93.50     | 98.75 / 98.35      | 94.73 / 93.97 | 61.96 / 61.18 |              |               |                 |
|VDCNN (17 layers, k-max-pooling) |      | 90.61 / 90.00  |      -/           | - /                | 94.95 / 94.73 | 62.59 /       |              |               |                 |
|VDCNN (29 layers, k-max-pooling) |      | 91.33 / 91.22  |      -/           | - /                | 95.37 / 94.82 | 63.00 /       |              |               |                 |
|    HAN                          |      |                |                   |                    |               |               |              |               |                 |


## Reference
* Xiang Zhang, Junbo Zhao, Yann LeCun.(2017) [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). 
* Yijun Xiao, Kyunghyun Cho, [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://arxiv.org/pdf/1602.00367).
* Conneau, A., Schwenk, H., Barrault, L., & Lecun, Y. (2017). [Very deep convolutional networks for text classification.](http://www.aclweb.org/anthology/E17-1104).
* Le, H. T., Cerisara, C., & Denis, A. (2017). [Do Convolutional Networks need to be Deep for Text Classification?](https://arxiv.org/pdf/1707.04108.pdf).

## Acknowledgement
Many codes inspired and borrowed from repos below.

https://github.com/srviest/char-cnn-pytorch

https://github.com/ArdalanM/nlp-benchmarks
