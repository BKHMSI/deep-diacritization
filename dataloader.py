import os
import yaml
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from collections import deque
from prettytable import PrettyTable
from pyarabic.araby import tokenize, strip_tashkeel

import torch as T
from torch.utils.data import DataLoader, Dataset

from data_utils import DatasetUtils

class DataRetriever(Dataset):
    def __init__(self, dtype, data_utils : DatasetUtils, is_test : bool = False):
        super(DataRetriever).__init__()

        self.data_utils = data_utils
        self.is_test = is_test

        subpath = f"-{data_utils.stride}-{data_utils.window}.txt"
        if is_test:
            subpath = f"-{data_utils.test_stride}-{data_utils.test_window}.txt"
   
        path = os.path.join(data_utils.base_path, dtype, dtype + subpath)
        with open(path, 'r', encoding="utf-8") as fin:
            self.lines = fin.readlines()

        if data_utils.debug:
            self.lines = self.lines[:256]

    def preprocess(self, data, dtype=T.long):
        return [T.tensor(x, dtype=dtype) for x in data]

    def __len__(self):
        return len(self.lines) 

    def __getitem__(self, idx):
        word_x, char_x, diac_x, diac_y = self.create_sentence(idx)
        return self.preprocess((word_x, char_x, diac_x)), T.tensor(diac_y, dtype=T.long)

    def create_sentence(self, idx):
        line = self.lines[idx]
        tokens = tokenize(line.strip())

        word_x = []
        char_x = []
        diac_x = []
        diac_y = []
        diac_y_tmp = []
        
        for word in tokens:
            split_word = self.data_utils.split_word_on_characters_with_diacritics(word)
            cx, cy, cy_3head = self.data_utils.create_label_for_word(split_word)

            word_strip = strip_tashkeel(word)
            word_x += [self.data_utils.w2idx[word_strip] if word_strip in self.data_utils.w2idx else self.data_utils.w2idx["<pad>"]]

            char_x += [self.data_utils.pad_sequence(cx, self.data_utils.max_word_len)]
            
            diac_y += [self.data_utils.pad_sequence(cy, self.data_utils.max_word_len, pad=self.data_utils.pad_target_val)]
            diac_y_tmp += [self.data_utils.pad_sequence(cy_3head, self.data_utils.max_word_len, pad=[self.data_utils.pad_target_val]*3)]

        diac_x = self.data_utils.create_decoder_input(diac_y_tmp)

        max_slen = self.data_utils.max_sent_len
        max_wlen = self.data_utils.max_word_len
        p_val = self.data_utils.pad_val
        pt_val = self.data_utils.pad_target_val

        word_x = self.data_utils.pad_sequence(word_x, max_slen)
        char_x = self.data_utils.pad_sequence(char_x, max_slen, pad=[p_val]*max_wlen)
        diac_x = self.data_utils.pad_sequence(diac_x, max_slen, pad=[[p_val]*8]*max_wlen)
        diac_y = self.data_utils.pad_sequence(diac_y, max_slen, pad=[pt_val]*max_wlen)

        return word_x, char_x, diac_x, diac_y