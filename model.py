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
from torch.autograd import Variable
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# 导入依赖库
import matplotlib.pyplot as plt
import seaborn as sns

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model).cuda()
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的porisional embedding
        pe = torch.zeros(max_len, d_model).cuda()
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        形式如：
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
        """
        position = torch.arange(0., max_len).cuda().unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2).cuda() * -(math.log(10000.0) / d_model))
        
        # TODO: 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] =  torch.sin(position * div_term)
        pe[:, 1::2] =  torch.cos(position * div_term)
        
        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0) 
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)
    
    # TODO: 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # TODO: 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim = -1)
    
    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return  self.a_2 * ( x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2 
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # TODO: 请利用init中的成员变量实现Feed Forward层的功能
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EmbedVid(nn.Module):
    '''
    图像的embeded
    inputs:batch_size * 32 * 1024
    outputs:batch_size * 32 * d_model
    '''
    def __init__(self,d_model, dropout=0.1):
        super(EmbedVid,self).__init__()
        self.linear = nn.Linear(1024,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return x
    
def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)
    
    # TODO: 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # TODO: 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim = -1)
    
    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask,vid):
        "Pass the input (and mask) through each layer in turn."
        if len(vid.size()) == 3:
            vid = vid.permute(0,2,1)
            vid = nn.Linear(vid.size()[-1],x.size()[1]).cuda()(vid)
            vid = vid.permute(0,2,1)
        elif len(vid.size()) == 2:
            vid = vid.permute(1,0)
            vid = nn.Linear(vid.size()[-1],x.size()[1]).cuda()(vid)
            vid = vid.permute(1,0)
        vid = torch.LongTensor(vid.data.cpu().numpy()).cuda()
        q = torch.cat([x,vid],axis=-1)
        q = nn.Linear(q.size(-1),x.size(-1)).cuda()(q)
        for layer in self.layers:
            x = layer(q,x,mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # The sublayerConnection is used to connect MULTI and FFN together
        # However, after each output Layer, the Layer Norm must be followed by the residual connection
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, q,x, mask):
        # Apply Multi Head Attention to the embedding layer
        x = self.sublayer[0](x, lambda x: self.self_attn(q, x, x, mask))
        # Notice that attn gets the result x as the input to the next level
        return self.sublayer[1](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    # Sets the SHAPE of the val ent_mask matrix
    attn_shape = (1, size, size)
    
    # TODO: Returns a val ent_mask matrix whose upper right corner (excluding the main diagonal) is all 1 and lower left corner (including the main diagonal) is all 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    # TODO: Returns a val ent_mask matrix whose upper right corner (excluding the main diagonal) is all False and the lower left corner (including the main diagonal) is all True
    return torch.from_numpy(subsequent_mask) == 0
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # TODO: Refer to EncoderLayer to complete the member variable definition
        # Copy N encoder layers
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask,vid):
        """
        Use a loop to decode N times (6 times in this case)
The DecoderLayer here receives an Attention Mask processing for the input
And a subsequent attention mask on output + subsequent mask processing
        """
        if len(vid.size()) == 3:#The image is fully connected
            vid = vid.permute(0,2,1)
            vid = nn.Linear(vid.size()[-1],x.size()[1]).cuda()(vid)
            vid = vid.permute(0,2,1)
        elif len(vid.size()) == 2:
            vid = vid.permute(1,0)
            vid = nn.Linear(vid.size()[-1],x.size()[1]).cuda()(vid)
            vid = vid.permute(1,0)
        vid = torch.LongTensor(vid.data.cpu().numpy()).cuda()
        q = torch.cat([x,vid],axis=-1)
        q = nn.Linear(q.size(-1),x.size(-1)).cuda()(q)
        for layer in self.layers:
            x = layer(q,x, memory, src_mask, tgt_mask)
        return self.norm(x),q


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # Attend the Context that the Encoder passed in
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self,q, x, memory, src_mask, tgt_mask):
        # Use m to store the final hidden of the encoder to represent the result
        m = memory
        
        # TODO: Refer to EncoderLayer to complete the Forwark function of DecoderLayer
        # Q, K and V of self-attention are decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(q, x, x, tgt_mask))
        # Context-Attention：Notice that the context-attention q is decoder hidden, and k and v are encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator,embed_vid):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator 
        self.embed_vid = embed_vid

    def encode(self, src, src_mask,vid):
        return self.encoder(self.src_embed(src), src_mask,self.embed_vid(vid))

    def decode(self, memory, src_mask, tgt, tgt_mask,vid):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask,self.embed_vid(vid))

    def forward(self, src, tgt, src_mask, tgt_mask,vid):
        "Take in and process masked src and target sequences."
        return self.encode(src, src_mask,vid),self.decode(self.encode(src, src_mask,vid), src_mask, tgt, tgt_mask,vid)

class Generator(nn.Module):
    # vocab: tgt_vocab
    # The result of the encoder is passed in as the memory parameter of the decoder to decode
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h = 10, dropout=0.1):
    c = copy.deepcopy
    # Instantiate the Attention object
    attn = MultiHeadedAttention(h, d_model).cuda()
    # Instantiate the Feedforward object
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).cuda()
    # Instantiate the PositionalEncoding object
    position = PositionalEncoding(d_model, dropout).cuda()
    # Instantiate the Transformer model objects
    embed_vid = EmbedVid(d_model)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).cuda(), N).cuda(),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).cuda(), N).cuda(),
        nn.Sequential(Embeddings(d_model, src_vocab).cuda(), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).cuda(), c(position)),
        Generator(d_model, tgt_vocab),
        EmbedVid(d_model,dropout)).cuda()
        
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            #This is initialized using nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.cuda()
class LabelSmoothing(nn.Module):
    """标签平滑处理"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
class SimpleLossCompute:
    """
    简单的计算损失和进行参数反向传播更新训练的函数
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def conv1Linear(self,x,vid):
        #Processing of model output
        conv1 = nn.Conv1d(in_channels=x.size()[-1],out_channels = vid.size()[-1], kernel_size = 5).cuda()
        linear = nn.Linear(x.size(1)-4,vid.size(1)).cuda()
        x = x.permute(0, 2, 1)
        out = conv1(x)
        out = linear(out)
        out = out.permute(0, 2, 1)
        return out
    def cal_encoder_loss(self,x,vid):
        #Calculate the output of the Encoder with the loss of the picture
        out = self.conv1Linear(x,vid)
        l1_loss_fn = torch.nn.L1Loss(reduce='sum', size_average=False)
        loss = l1_loss_fn(out, vid)
        return loss

    def cal_decoder_loss(self,x,vid):
        #Calculate the output of Decoder and loss of pictures
        out = self.conv1Linear(x,vid)
        l1_loss_fn = torch.nn.L1Loss(reduce='sum', size_average=False)
        loss = l1_loss_fn(out, vid)
        return loss
    
    def __call__(self, out, y, norm,en_out,vid,q):
        x = self.generator(out)
        loss0 = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss1 = self.cal_encoder_loss(en_out,vid) /(64*256*49)
        loss2 = self.cal_decoder_loss(out,vid) /(64*256*49)
        q = self.generator(q)
        loss3 = self.criterion(q.contiguous().view(-1, q.size(-1)), 
                              y.contiguous().view(-1)) / norm
        
        loss = loss0+loss1+loss2+loss3
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float(), loss0.data.item() * norm.float(),loss3.data.item() * norm.float()
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup # 2000
        self.factor = factor # 1
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
def greedy_decode(model, src, src_mask,vid, max_len, start_symbol):
    """
   A trained model is passed in to make predictions for the specified data
    """
    #Start with encoder for encode
    vid = vid.unsqueeze(0)
    memory = model.encode(src, src_mask,vid)
    # Initialize the tensor with a prediction of 1×1, fill in the id of the start character ('BOS'), and set the type to the input data type (LongTensor).
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # Traverse the length subscript of the output
    for i in range(max_len-1):
        # The decode gets the hidden layer representation
        (out,q) = model.decode(memory, 
                           src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
                           vid)
        # Converts the hidden representation to a log_softmax probability distribution representation of each word in the dictionary
        prob = model.generator(out[:, -1])
        # Gets the prediction word id of the maximum probability of the current position
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        # Stitches the character ID of the current position prediction with the previous prediction
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
