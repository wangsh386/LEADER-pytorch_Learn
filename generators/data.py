# here put the import lib
import numpy as np
import pandas as pd
import pickle
import copy
import os
import random
import dill

import torch
from torch.utils.data import Dataset


class Voc(object):
    '''Define the vocabulary (token) dict'''

    def __init__(self):

        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        '''add vocabulary to dict via a list of words'''
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)



class EHRTokenizer(object):
    """医疗记录(EHR)数据的分词器，提供ID与token相互转换的功能
    处理诊断(diagnosis)、药物(medication)、操作(procedure)三类医疗编码"""

    def __init__(self, voc_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):
        """初始化方法
        Args:
            voc_dir:  词汇表文件路径，包含诊断/药物/操作编码的预定义词典
            special_tokens: 特殊标记，默认为填充符/句子开头/掩码符"""
        
        # 主词汇表，包含所有token（特殊标记 + 三类医疗编码）
        self.vocab = Voc()  # 假设Voc类实现字典功能，包含word2idx和idx2word

        # 步骤1：添加特殊符号
        self.vocab.add_sentence(special_tokens)  # 将[PAD], [CLS]等加入主词汇表

        # 步骤2：加载三类医疗专用词典
        self.diag_voc, self.med_voc, self.pro_voc = self.read_voc(voc_dir)
        
        # 步骤3：合并编码到主词汇表
        self.vocab.add_sentence(self.med_voc.word2idx.keys())  # 添加药物编码
        self.vocab.add_sentence(self.diag_voc.word2idx.keys()) # 添加诊断编码
        self.vocab.add_sentence(self.pro_voc.word2idx.keys())  # 添加操作编码

        # 可选属性初始化（当前未使用）
        self.attri_num = None  # 可扩展用于属性数量记录
        self.hos_num = None    # 可扩展用于医院编号记录

    def read_voc(self, voc_dir):
        """从二进制文件加载预定义的医疗词典
        Returns:
            tuple: (诊断词典, 药物词典, 操作词典)"""
        with open(voc_dir, 'rb') as f:
            voc_dict = dill.load(f)  # 使用dill反序列化词典对象
            
        return voc_dict['diag_voc'], voc_dict['med_voc'], voc_dict['pro_voc']

    def add_vocab(self, vocab_file):
        """扩展词汇表（例如添加新发现的医疗编码）
        Args:
            vocab_file: 新词汇文本文件，每行一个编码
        Returns:
            Voc: 新增的专用词汇表"""
        voc = self.vocab
        specific_voc = Voc()  # 新建子词汇表

        # 逐行读取并添加新编码
        with open(vocab_file, 'r') as fin:
            for code in fin:
                stripped_code = code.rstrip('\n')  # 去除换行符
                voc.add_sentence([stripped_code])   # 添加到主词汇表
                specific_voc.add_sentence([stripped_code]) # 添加到子词汇表

        return specific_voc

    def convert_med_tokens_to_ids(self, tokens):
        """【药物专用】将token列表转换为ID列表
        Args:
            tokens: 药物编码列表，如["J01EA04", "N02BB02"]
        Returns:
            list: 对应的ID列表，自动跳过未知token"""
        ids = []
        unknown_tokens = set()  # 记录未知编码
        
        for token in tokens:
            if token in self.med_voc.word2idx:    # 检查药物词典
                ids.append(self.med_voc.word2idx[token])
            else:
                unknown_tokens.add(token)  # 收集未知药物编码

        # 打印警告信息（示例显示前3个未知编码）
        if unknown_tokens:
            print(f"发现{len(unknown_tokens)}个未知药物编码，例如：{list(unknown_tokens)[:3]}...")
            
        return ids

    def convert_ids_to_tokens(self, ids):
        """通用方法：将ID序列转换回token（使用主词汇表）
        Args:
            ids: 如[101, 234, 56]
        Returns:
            list: 对应的token列表，如["[CLS]", "I25", "[PAD]"]"""
        return [self.vocab.idx2word[i] for i in ids]

    def convert_tokens_to_ids(self, tokens):
        """通用方法：将token序列转换为ID（使用主词汇表）
        Args:
            tokens: 如["[CLS]", "E11", "[MASK]"]
        Returns:
            list: 对应的ID列表"""
        return [self.vocab.word2idx[token] for token in tokens]

    def convert_med_ids_to_tokens(self, ids):
        """【药物专用】将ID转换回药物编码（使用药物子词典）
        Args:
            ids: 如[201, 305]
        Returns:
            list: 药物编码列表，如["J01EA04", "N02BB02"]"""
        return [self.med_voc.idx2word[i] for i in ids]



class EHRDataset(Dataset):
    '''The dataset for medication recommendation'''

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len  # the maximum length of a diagnosis/procedure record

        self.sample_counter = 0
        self.records = data_pd

        self.var_name = []


    def __len__(self):

        return NotImplementedError

    def __getitem__(self, item):

        return NotImplementedError



####################################
'''Finetune Dataset'''
####################################

class FinetuneEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        
        super().__init__(data_pd, tokenizer, max_seq_len)
        self.max_seq = 10
        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", "labels"]


    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):

        # one admission: [diagnosis, procedure, medication]
        adm = copy.deepcopy(self.records[item])

        med_seq = [meta_adm[2] for meta_adm in adm]
        diag_seq = [meta_adm[0] for meta_adm in adm]
        proc_seq = [meta_adm[1] for meta_adm in adm]

        # get the medcation recommendation label -- multi-hot vector
        label_index = self.tokenizer.convert_med_tokens_to_ids(med_seq[-1])
        label = np.zeros(len(self.tokenizer.med_voc.word2idx))
        for index in label_index:
            label[index] = 1

        # get the seq len
        # pad the sequence to longest med / diag / proc sequences
        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        # convert raw tokens to unified ids
        med_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in med_seq]
        diag_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in diag_seq]
        proc_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in proc_seq]

        # pad the sequence to max possible records
        pad_seq = ["[PAD]" for _ in range(self.seq_len)]
        pad_seq = self.tokenizer.convert_tokens_to_ids(pad_seq)
        def fill_to_max_seq(l, seq):
            pad_num = 0
            while len(l) < seq:
                l.append(pad_seq)
                pad_num += 1
            if len(l) > seq:
                l = l[:seq]
            return l, pad_num
        med_seq = med_seq[:-1]  # remove the current medication set, which is label
        med_seq, _ = fill_to_max_seq(med_seq, self.max_seq)
        diag_seq, pad_num = fill_to_max_seq(diag_seq, self.max_seq)
        proc_seq, _ = fill_to_max_seq(proc_seq, self.max_seq)

        # get mask
        mask = np.ones(self.max_seq)
        if pad_num != 0:
            mask[-pad_num:] = 0

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
               np.array(med_seq, dtype=int), mask.astype(int), label.astype(float)



####################################
'''MedRec Dataset'''
####################################

class MedRecEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, profile_tokenizer, args):
        
        super().__init__(data_pd, tokenizer, args.max_seq_length)

        if args.filter:
            self._filter_data()

        self.max_seq = args.max_record_num
        self.profile_tokenizer = profile_tokenizer
        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", "labels", "multi_label", "profile"]


    def __len__(self):

        return len(self.records)

    
    def __getitem__(self, item):

        # one admission: [diagnosis, procedure, medication]
        adm = copy.deepcopy(self.records[item])

        med_seq = adm["records"]["medication"]
        diag_seq = adm["records"]["diagnosis"]
        proc_seq = adm["records"]["procedure"]

        # encode profile, get a vector to organize all feature orderly
        profile = []
        for k, v in adm["profile"].items():
            profile.append(self.profile_tokenizer["word2idx"][k][v])

        # get the medcation recommendation label -- multi-hot vector
        label_index = self.tokenizer.convert_med_tokens_to_ids(med_seq[-1])
        label = np.zeros(len(self.tokenizer.med_voc.word2idx))
        multi_label = np.full(len(self.tokenizer.med_voc.word2idx), -1)
        for i, index in enumerate(label_index):
            label[index] = 1
            multi_label[i] = index

        # get the seq len
        # pad the sequence to longest med / diag / proc sequences
        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        # convert raw tokens to unified ids
        med_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in med_seq]
        diag_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in diag_seq]
        proc_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in proc_seq]

        # pad the sequence to max possible records
        pad_seq = ["[PAD]" for _ in range(self.seq_len)]
        pad_seq = self.tokenizer.convert_tokens_to_ids(pad_seq)
        def fill_to_max_seq(l, seq):
            pad_num = 0
            while len(l) < seq:
                l.append(pad_seq)
                pad_num += 1
            if len(l) > seq:
                l = l[:seq]
            return l, pad_num
        med_seq = med_seq[:-1]  # remove the current medication set, which is label
        med_seq, _ = fill_to_max_seq(med_seq, self.max_seq)
        diag_seq, pad_num = fill_to_max_seq(diag_seq, self.max_seq)
        proc_seq, _ = fill_to_max_seq(proc_seq, self.max_seq)

        # get mask
        mask = np.ones(self.max_seq)
        if pad_num != 0:
            mask[-pad_num:] = 0

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
               np.array(med_seq, dtype=int), mask.astype(int), label.astype(float), \
               multi_label.astype(int), np.array(profile, dtype=int)
    

    def _filter_data(self):

        new_records = []

        for record in self.records:
            if len(record["records"]["medication"]) > 1:
                new_records.append(record)

        self.records = new_records
