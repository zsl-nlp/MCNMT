#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 

# Initialize parameter Settings
UNK = 0  # The dictionary ID corresponding to the identifier of the unlogged word
PAD = 1  # Padding placeholder dictionary ID
MAX_LENGTH = 49
def load_data_dict(path):
    en = []
    cn = []
    with open(path, 'r', encoding='utf-8') as fin:
        fin = fin.read().split('\n')
        for line in fin[:-1]:
            list_content = line.split('\t')
            en.append(['BOS'] + word_tokenize(list_content[0]) + ['EOS'])
            cn.append(['BOS'] + word_tokenize(list_content[1]) + ['EOS'])
    return en, cn


                
def build_dict(sentences, max_words = 10000):
        """
        List data after passing in the participle of load_data construct
        Build a dictionary (key for words, value for id values)
        """
        # Count all the words in the data
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        # Only the first max_words of the highest frequency are retained to build the dictionary
        # Add the words "UNK" and "PAD", and the corresponding ID has been initialized
        ls = word_count.most_common(max_words)
        # Count the total number of words in the dictionary
        total_words = len(ls) + 2

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        # Then build a reverse dictionary, for ID to use the word
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

class PrepareData(Dataset):
    def __init__(self, file_path,en_word_dict,cn_word_dict):
        # Read the data and segment the words
        self.train_en, self.train_cn , self.train_vid = self.load_data(file_path)
        
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, en_word_dict, cn_word_dict)

    def load_data(self, path):
        """
        Read pre - translation (English) and post - translation (Chinese) data files
        Each piece of data is particified and then constructed into a list of words (characters in Chinese) containing the start (BOS) and stop (EOS) characters
        Form such as：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        vid = []
        # TODO ...
        with open(path, 'r', encoding='utf-8') as fin:
            fin = fin.read().split('\n')
            print(len(fin))
            for line in fin[:-1]:
                list_content = line.split('\t')
                en.append(['BOS'] + word_tokenize(list_content[0]) + ['EOS'])
                cn.append(['BOS'] + word_tokenize(list_content[1]) + ['EOS'])
                vid.append(self.load_video_features(list_content[2]))
        print(f'The number of data {len(en)}')
        return en, cn , vid
    
    def load_video_features(self,fpath, max_length=49):
        #Load image vector

        feats = np.load(fpath, encoding='latin1')  # encoding='latin1' to handle the inconsistency between python 2 and 3

        assert feats.shape[0] == max_length
        return np.float32(feats)
    
 

    def wordToID(self, en, cn, en_dict, cn_dict, sort=False):
        """
        This method can translate the data represented by the word list of the pre-translation (English) data and the post-translation (Chinese) data
        Are converted to data represented by an ID list
        If the sort parameter is set to True, it will be sorted by the length of the sentence (number of words) before translation
        So that each sentence in the same batch should have the same length of padding to reduce the amount of padding
        """
        # Count the number of English data bars
        length = len(en)
        
        # TODO: Convert both pre-translation (English) and post-translation (Chinese) data to the form represented by ID
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # Build a function that sorts by sentence length
        def len_argsort(seq):
            """
            Passing in a series of sentence data (in the form of a list of broken words),
            After sorting by sentence length, returns the index index of each sentence in the data after sorting
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # Sort Chinese and English in the same order
        if sort:
            # Based on the English sentence length order (sentence subscript)
            sorted_index = len_argsort(out_en_ids)
            
            # TODO: Both pre-translation (English) and post-translation (Chinese) data are sorted according to this benchmark
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
            
        return out_en_ids, out_cn_ids

    def __len__(self):
        return len(self.train_en)

    def __getitem__(self,idx):
        pad=0
        data_en = self.train_en[idx]
        data_en = seq_padding(data_en,ML=MAX_LENGTH)
        data_cn = self.train_cn[idx]
        data_cn = seq_padding(data_cn,ML=MAX_LENGTH+1)
        data_vid = self.train_vid[idx]
        
        return data_en, data_cn, data_vid

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
        
def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    
    # TODO: 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    # TODO: 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0
         
def seq_padding(X, padding=0,ML=None):
    """
    Padding the alignment length for a batch of data (represented by the word ID)
    """
    if ML == None:
        # Calculate the sentence length of each piece of data in this batch
        L = [len(x) for x in X]
        # Gets the maximum sentence length of the batch data
        ML = max(L)
    # Perform traversal for each piece of data X in X. If the length is shorter than the maximum length of the batch of data ML, then add the missing length mL-len (X) with Padding ID.
    
    return np.array(
        np.concatenate([X, [padding] * (ML - len(X))]) if len(X) < ML else X[:ML])
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None,vid = None, pad=0):
        # Canonize the data represented by the input and output word IDs into integer types

        src = torch.LongTensor(src).cuda()
        trg = torch.LongTensor(trg).cuda()
        self.vid = torch.FloatTensor(vid).cuda()
        self.src = src
        # The non-empty part of the current input sentence is judged as a BOOL sequence
        # And add one dimension in front of Seq Length to form a matrix with dimensions of 1× Seq Length
        self.src_mask = (src != pad).unsqueeze(-2)
        # If the output target is not empty, the target sentence to be used by the Decoder needs to be masked
        if trg is not None:
            # Target input that Decoder uses
            self.trg = trg[:, :-1]
            # Decoder training should predict the output target results
            self.trg_y = trg[:, 1:]
            # Put the target input part into the Attention Mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # The actual number of words in the target result that should be output is counted
            self.ntokens = (self.trg_y != pad).data.sum()
    
    # Mask operation
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask