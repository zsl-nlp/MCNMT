#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.meteor_score import single_meteor_score as meteor
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import nltk
from model import *
from data_process import *

# Initialize parameter Settings
UNK = 0  # The dictionary ID corresponding to the identifier of the login word
PAD = 1  # Padding placeholder dictionary ID
BATCH_SIZE = 64  # Number of training data per batch
EPOCHS = 50  # Number of training epochs
LAYERS = 6  # The number of encoder and decoder block layers stacked in Transformer
H_NUM = 8  # Multihead attention hidden number
D_MODEL = 256  # embedding dimension
D_FF = 1024  # The first full connection layer dimension of feed forward
DROPOUT = 0.5  # Dropout rate
MAX_LENGTH = 49  # Maximum sentence length
pad = 0

NOTE = 'MCNMT'
if not os.path.exists('save_model'):os.mkdir('save_model')
SAVE_PATH = os.path.join('save_model',NOTE)
SAVE_FILE   = os.path.join('save_model','MCNMT.pt')  
SAVE_FILE2   = os.path.join('save_model','MCNMT_before.pt')  
RESULT_PATH = os.path.join('save_result',NOTE)
if not os.path.exists(RESULT_PATH):os.mkdir(RESULT_PATH)
PATH_17 = os.path.join(RESULT_PATH,'test_2017_1')  
PATH_16 = os.path.join(RESULT_PATH,'test_2016_1')  
TEST_FILE_2017 = "data_image_conv/en2de/test_2017.txt"
TEST_FILE_2016 = "data_image_conv/en2de/test_2016.txt"
RESULT_SAVE = 'result_'+NOTE+'.csv'
print('The model will be saved in ',SAVE_FILE)
print('The model will be backed up in ',SAVE_FILE2)
print('The test file 1 is ',PATH_17)
print('The test file 2 is ',PATH_16)
# print('Test results are output to ',RESULT_SAVE)

TRAIN_FILE = 'data_image_conv/en2de/train.txt'  # Training set data file
DEV_FILE = "data_image_conv/en2de/val.txt"  # Verify (develop) set data files
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def val_epoch(data, model, loss_compute, epoch):
    # validation
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    loss3s = 0.

    i = 0
    for src, trg, vid in data:
        i += 1
        batch = Batch(src,trg,vid)
        
        en_out,(out,q) = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask,batch.vid)
        loss,loss0,loss3 = loss_compute(out, batch.trg_y, batch.ntokens,en_out,batch.vid,q)
        
        loss3s += loss3
        total_loss += loss0
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 100 == 1:
            elapsed = time.time() - start
#             print("Epoch %d Batch: %d Loss: %f Loss3: %f Tokens per Sec: %fs" % (epoch, i - 1, loss0 / batch.ntokens, loss3s / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (epoch, i - 1, total_loss /i, (tokens.float() / elapsed / 1000.)))
            
            start = time.time()
            tokens = 0

    sentbleu,corpbleu,scores = 0,0,0

    return total_loss / total_tokens,sentbleu,corpbleu,scores,loss3s/ total_tokens

def test_epoch(data, model, loss_compute,epoch,path):
    # test
    lit_tok = []
    lit_translate = []
    with torch.no_grad():
        # Traverse the subscript over the length of the English data in data
        for src, trg, vid in data:
            batch = Batch(src,trg,vid)
            for i in range(len(batch.src)):
                src_c = batch.src[i].clone()
                trg_c = batch.trg[i].clone()
                src_lit = src_c.cpu().numpy()
                trg_lit = trg_c.cpu().numpy()
                en_sent = [en_index_dict[w] for w in  src_lit]
                cn_sent = [cn_index_dict[w] for w in  trg_lit]
                try:
                    if cn_sent.index('EOS') is not None:
                        index = cn_sent.index('EOS')
                        cn_sent = cn_sent[1:index]
                except:
                    print('EOS not find')

                cn_sent = ' '.join(cn_sent)
                lit_tok.append(cn_sent)
                # The current English sentence data represented by word id is converted into tensor and put in the DEVICE
                src_i = torch.from_numpy(np.array(batch.src[i].cpu())).long().cuda()
                # Add a dimension
                src_i = src_i.unsqueeze(0)
                #image            
                div = torch.FloatTensor(batch.vid[i].cpu()).cuda()
                src_mask = (src_i != 0).unsqueeze(-2)
                # The decode prediction was performed using the trained model
                out = greedy_decode(model, src_i, src_mask,div, max_len=MAX_LENGTH, start_symbol=cn_word_dict["BOS"])
                # Initializes a list of sentence words used to hold model translation results
                translation = []
                # Traverse the subscript of the translated output character (note: index 0 of the beginning character "BOS" does not traverse)
                for j in range(1, out.size(1)):
                    # Gets the output character of the current subscript
                    sym = cn_index_dict[out[0, j].item()]
                    # If the output character is not an 'EOS' terminator, it is added to the list of translations for the current sentence
                    if sym != 'EOS':
                        translation.append(sym)
                    # Otherwise the traversal terminates
                    else:
                        break

                sent = " ".join(translation)
                lit_translate.append(sent)
                
        with open(path+'tok','w') as f2:
            for ff2 in lit_tok:
                f2.write(ff2)
                f2.write('\n')
        with open(path+'translate','w') as f:
            for ff in lit_translate:
                f.write(ff)
                f.write('\n')
def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    i = 0
    for src, trg, vid in data:
        i += 1

        batch = Batch(src,trg,vid)
        
        en_out,(out,q) = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask,batch.vid)
        loss,loss0,loss3 = loss_compute(out, batch.trg_y, batch.ntokens,en_out,batch.vid,q)
        total_loss += loss0
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 100 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (epoch, i - 1, loss0 / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0
    
    
    return total_loss / total_tokens
        
def train(train_loader,val_loader, model, criterion, optimizer):
    """
    Train and save the model
    """
    # Initialize the optimal Loss of the model on the DEV set to a large value
    best_test_loss = 1e5
    best_test_bleu = 0
    best_epoch = 0
    best_scores = 0
    for epoch in range(EPOCHS):
        # Model training
        model.train()
        run_epoch(train_loader, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
#         val_epoch(val_loader, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)

    torch.save(model.state_dict(), SAVE_FILE)
if __name__ == "__main__":
    # Data preprocessing
    data_dic_en,data_dic_cn = load_data_dict(TRAIN_FILE)
    en_word_dict,en_total_words,en_index_dict = build_dict(data_dic_en)
    cn_word_dict,cn_total_words,cn_index_dict = build_dict(data_dic_cn)

    train_dataset = PrepareData(TRAIN_FILE, en_word_dict, cn_word_dict)
    val_dataset = PrepareData(DEV_FILE, en_word_dict, cn_word_dict)
    train_loader = DataLoader(dataset=train_dataset,batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size = BATCH_SIZE, shuffle=True)


    src_vocab = len(en_word_dict)
    tgt_vocab = len(cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)

    # Initialization model
    model = make_model(
                        src_vocab, 
                        tgt_vocab, 
                        LAYERS, 
                        D_MODEL, 
                        D_FF,
                        H_NUM,
                        DROPOUT
                    )
    # train

    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0)
    optimizer = NoamOpt(D_MODEL, 2, 8000, torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.98), eps=1e-9))
    
#     print(">>>>>>> start train")
#     train(train_loader,val_loader, model, criterion, optimizer)
#     print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")
#     torch.save(model.state_dict(), SAVE_FILE2)
    
    #test
    test_dataset = PrepareData(TEST_FILE_2016, en_word_dict, cn_word_dict)
    test_loader = DataLoader(dataset=test_dataset,batch_size = BATCH_SIZE, shuffle=False)
    model.load_state_dict(torch.load(SAVE_FILE2))
    model.eval()
    epoch = 0
    PATH = PATH_16+str(epoch)
    print(PATH)
    test_loss, test_sentbleu, test_corpbleu, test_scores = test_epoch(test_loader, model, SimpleLossCompute(model.generator, criterion, None), epoch,PATH)