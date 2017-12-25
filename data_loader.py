import csv
import os.path as op
import re
import torch
import codecs
import json
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd
from sklearn.model_selection import train_test_split
import numpy as np

import sys
import random

maxInt = sys.maxsize  
decrement = True 

while decrement:  
    # decrease the maxInt value by factor 10   
    # as long as the OverflowError occurs.  
  
    decrement = False  
    try:  
        csv.field_size_limit(maxInt)  
    except OverflowError:  
        maxInt = int(maxInt/10)  
        decrement = True  

class AGNEWs(Dataset):
    def __init__(self, data, label, alphabet_path, l0=1014,onehot=True):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        # read alphabet
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        self.alphabet = alphabet
        self.l0 = l0
        self.onehot = onehot
        self.data = data
        self.y = label
            
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.onehot:
            X = self.oneHotEncode(idx)
        else:
            X = self.get_id_list(idx)
        y = self.y[idx]
        return X, y

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        # print len(self.alphabet)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if index_char >= self.l0: 
                break # limit the sequence length to l0
            if self.char2Index(char)!=-1:
                X[self.char2Index(char)][index_char] = 1.0
            else:
                X[self.char2Index("#")][index_char] = 1.0               
        return X

    def get_id_list(self, idx):
        X = torch.zeros(self.l0).long()
        for char_idx, char in enumerate(self.data[idx]):
            if char_idx >= self.l0:
                break
            if self.char2Index(char) != -1:
                X[char_idx] = self.char2Index(char)
            else:
                X[char_idx] = self.char2Index("#")
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def get_class_weight(self):
        num_samples = self.__len__()
        label_set = set(self.y)
        num_class = [self.y.count(c) for c in label_set]
        class_weight = [num_samples/float(self.y.count(c)) for c in label_set]    
        return class_weight, num_class 

def class_num(label_data_path):
    if 'ag_news' in label_data_path:
        nclass = 4
    if 'dbpedia' in label_data_path:
        nclass = 14
    if 'yelp_review_polarity' in label_data_path:
        nclass = 14
    if 'yelp_review_ful' in label_data_path:
        nclass = 5
    if 'yahoo_answers' in label_data_path:
        nclass = 10
    if 'sogou_news' in label_data_path:
        nclass = 5
    if 'amazon_review_full' in label_data_path:
        nclass = 5
    if 'amazon_review_polarity' in label_data_path:
        nclass = 5
    return nclass

def load(label_data_path, lowercase=True):
    label = []
    data = []
    with open(label_data_path, 'rb') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        # num_samples = sum(1 for row in rdr)
        for index, row in enumerate(rdr):
            label.append(int(row[0]))
            txt = ' '.join(row[1:])
            if lowercase:
                txt = txt.lower()                
            data.append(txt)
    return data, label 

def get_train_val_data(label_data_path, alphabet_path, l0=1014, onehot=True):
    nclass = class_num(label_data_path)
    data, label = load(label_data_path)
    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.1, shuffle=True) 
    train_dataset = AGNEWs(X_train, y_train, alphabet_path, l0, onehot)
    val_dataset = AGNEWs(X_train, y_train, alphabet_path, l0, onehot)
    return nclass, train_dataset, val_dataset

def get_test_data(label_data_path, alphabet_path, l0=1014, onehot=True):
    print label_data_path
    nclass = class_num(label_data_path)
    data, label = load(label_data_path)
    # data, label = zip(*(random.sample(zip(data,label),20000)))
    return nclass, AGNEWs(data,label, alphabet_path,l0,onehot)


def get_model_param_num(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

if __name__ == '__main__':
    
    label_data_path = '/Users/ychen/Documents/TextClfy/data/ag_news_csv/test.csv'
    alphabet_path = '/Users/ychen/Documents/TextClfy/alphabet.json'

    train_dataset = AGNEWs(label_data_path, alphabet_path)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, drop_last=False)
    # print(len(train_loader))
    # print(train_loader.__len__())

    # size = 0
    for i_batch, sample_batched in enumerate(train_loader):

        # len(i_batch)
        # print(sample_batched['label'].size())
        inputs = sample_batched['data']
        print(inputs.size())
        # print('type(target): ', target)
        
