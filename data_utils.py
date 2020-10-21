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

class DatasetUtils:
    def __init__(self, config):
        self.base_path = config["paths"]["base"]
        self.special_tokens = ['<pad>', '<unk>', '<num>', '<punc>'] 
        self.delimeters = config["sentence-break"]["delimeters"]
        self.load_constants(config["paths"]["constants"])
        self.debug = config["debug"]

        self.stride = config["sentence-break"]["stride"]  
        self.window = config["sentence-break"]["window"]  

        self.test_stride = config["predictor"]["stride"]  
        self.test_window = config["predictor"]["window"] 
        
        self.max_word_len = config["train"]["max-word-len"]
        self.max_sent_len = config["train"]["max-sent-len"]
        self.pad_val = self.special_tokens.index("<pad>")
        self.pad_target_val = -100

        self.markov_signal = config['train'].get('markov-signal', False)
        self.batch_first = config['train'].get('batch-first', True)

        self.gt_prob = config["predictor"]["gt-signal-prob"]
        if self.gt_prob > 0:
            self.s_idx = config["predictor"]["seed-idx"]
            subpath = f"test_gt_mask_{self.gt_prob}_{self.s_idx}.txt"
            mask_path = os.path.join(self.base_path, "test", subpath)
            with open(mask_path, 'r') as fin:
                self.gt_mask = fin.readlines()

        self.embeddings, self.vocab = self.load_embeddings(config["paths"]["word-embs"], config["loader"]["wembs-limit"])
        self.embeddings = self.normalize(self.embeddings, ["unit", "centeremb", "unit"])
        self.w2idx = {word: i for i, word in enumerate(self.vocab)}

    def load_file(self, path):
        with open(path, 'rb') as f:
            return list(pickle.load(f))

    def normalize(self, matrix, actions, mean=None):
        def length_normalize(matrix):
            norms = np.sqrt(np.sum(matrix**2, axis=1))
            norms[norms == 0] = 1
            matrix = matrix / norms[:, np.newaxis]
            return matrix

        def mean_center(matrix):
            return matrix - mean

        def length_normalize_dimensionwise(matrix):
            norms = np.sqrt(np.sum(matrix**2, axis=0))
            norms[norms == 0] = 1
            matrix = matrix / norms
            return matrix

        def mean_center_embeddingwise(matrix):
            avg = np.mean(matrix, axis=1)
            matrix = matrix - avg[:, np.newaxis]
            return matrix

        for action in actions:
            if action == 'unit':
                matrix = length_normalize(matrix)
            elif action == 'center':
                matrix = mean_center(matrix)
            elif action == 'unitdim':
                matrix = length_normalize_dimensionwise(matrix)
            elif action == 'centeremb':
                matrix = mean_center_embeddingwise(matrix)

        return matrix

    def load_constants(self, path):
        self.numbers = [c for c in "0123456789"]
        self.letter_list = self.special_tokens + self.load_file(os.path.join(path, 'ARABIC_LETTERS_LIST.pickle'))
        self.diacritic_list = [' '] + self.load_file(os.path.join(path, 'DIACRITICS_LIST.pickle'))

    def char_type(self, char):
        if char in self.letter_list:
            return self.letter_list.index(char)
        elif char in self.numbers:
            return self.letter_list.index('<num>')
        elif char in self.delimeters:
            return self.letter_list.index('<punc>')
        else:
            return self.letter_list.index('<unk>')        

    def split_word_on_characters_with_diacritics(self, word):
        word_queue = deque(word)
        split_word_on_characters = []
        temp_string = [word_queue.popleft()]
        while len(word_queue) > 0:
            poping_left = word_queue.popleft()
            if poping_left not in self.diacritic_list:
                split_word_on_characters.append(temp_string)
                temp_string = [poping_left]
            else:
                temp_string += [poping_left]
        split_word_on_characters.append(temp_string)
        return split_word_on_characters

    def load_mapping_v3(self, dtype, file_ext=None):
        mapping = {}
        if file_ext is None:
            file_ext = f"-{self.test_stride}-{self.test_window}.map"
        f_name = os.path.join(self.base_path, dtype, dtype + file_ext)
        with open(f_name, 'r') as fin:
            for line in fin:
                sent_idx, seg_idx, t_idx, c_idx = map(int, line.split(','))
                if sent_idx not in mapping:
                    mapping[sent_idx] = {}
                if seg_idx not in mapping[sent_idx]:
                    mapping[sent_idx][seg_idx] = {}
                if t_idx not in mapping[sent_idx][seg_idx]:
                    mapping[sent_idx][seg_idx][t_idx] = []
                mapping[sent_idx][seg_idx][t_idx] += [c_idx]
        return mapping

    def load_embeddings(self, embs_path, limit=-1):
        if self.debug:
            return np.zeros((200+len(self.special_tokens),300)), self.special_tokens + ["c"] * 200

        words = [self.special_tokens[0]]
        print(f"[INFO] Reading Embeddings from {embs_path}")
        with open(embs_path, encoding='utf-8', mode='r') as fin:
            n, d = map(int, fin.readline().split())
            limit = n if limit <= 0 else limit
            embeddings = np.zeros((limit+1, d))
            for i, line in tqdm(enumerate(fin), total=limit):
                if i >= limit: break
                tokens = line.rstrip().split()
                words += [tokens[0]]
                embeddings[i+1] = list(map(float, tokens[1:]))
        return embeddings, words

    def load_file_clean(self, dtype, strip=False):
        f_name = os.path.join(self.base_path, dtype, dtype + ".txt")
        with open(f_name, 'r', encoding="utf-8") as fin:
            original_lines = [strip_tashkeel(self.preprocess(line)) if strip else self.preprocess(line) for line in fin.readlines()]
        return original_lines

    def preprocess(self, line):
        return ' '.join(tokenize(line))

    def pad_sequence(self, tokens, max_len, pad=None):
        if pad is None: 
            pad = self.special_tokens.index("<pad>")
        if len(tokens) < max_len:
            offset = max_len - len(tokens)
            return tokens + [pad] * offset
        else:
            return tokens[:max_len]

    def stats(self, freq, percentile=90, name="stats"):
        table = PrettyTable(["Dataset", "Mean", "Std", "Min", "Max", f"{percentile}th Percentile"])
        freq = np.array(sorted(freq))
        table.add_row([name, freq.mean(), freq.std(), freq.min(), freq.max(), np.percentile(freq, percentile)])
        print(table)

    def create_labels(self, char):
        remap_dict = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4}
        char = [char[0]] + list(set(char[1:]))
        if len(char) > 3:
            char = char[:2] if self.diacritic_list[8] not in char else char[:3]

        char_idx = self.char_type(char[0])
        if len(char) == 1:
            return char_idx, 0, [remap_dict[0], 0, 0]
        elif len(char) == 2:  # If not shadda
            diacritic_index = self.diacritic_list.index(char[1])
            if diacritic_index in [2, 4, 6]:  # list of doubles
                return char_idx, diacritic_index, [remap_dict[diacritic_index - 1], 1, 0]
            elif diacritic_index == 8:
                return char_idx, diacritic_index, [0, 0, 1]
            else:
                return char_idx, diacritic_index, [remap_dict[diacritic_index], 0, 0]
        elif len(char) == 3:  # If shadda
            if self.diacritic_list[8] == char[1]:
                diacritic_index = self.diacritic_list.index(char[2])
            else:
                diacritic_index = self.diacritic_list.index(char[1])

            if diacritic_index in [2, 4, 6]:  # list of doubles
                return char_idx, diacritic_index+8, [remap_dict[diacritic_index - 1], 1, 1]
            else:
                return char_idx, diacritic_index+8, [remap_dict[diacritic_index], 0, 1]
        return None, None, None

    def create_label_for_word(self, split_word_):
        word_label_x = []
        diac_label_x = []
        diac_label_y = []
        for character_ in split_word_:
            char_x, diac_x, diac_y = self.create_labels(character_)
            if char_x == None:
                print(split_word_)
                raise ValueError(char_x)
            word_label_x.append(char_x)
            diac_label_x.append(diac_x)
            diac_label_y.append(diac_y)
        return word_label_x, diac_label_x, diac_label_y

    def create_gt_mask(self, lines, prob, idx, seed=1111):
        np.random.seed(seed)

        gt_masks = []
        for line in lines:
            tokens = tokenize(line.strip())
            gt_mask_token = ""
            for t_idx, token in enumerate(tokens):
                gt_mask_token += ''.join(map(str, np.random.binomial(1, prob, len(token))))
                if t_idx+1 < len(tokens):
                    gt_mask_token += " "
            gt_masks += [gt_mask_token]

        subpath = f"test_gt_mask_{prob}_{idx}.txt"
        mask_path = os.path.join(self.base_path, "test", subpath)

        with open(mask_path, 'w') as fout:
            fout.write('\n'.join(gt_masks))        
      
    def create_gt_labels(self, lines):
        gt_labels = []
        for line in lines:
            gt_labels_line = []
            tokens = tokenize(line.strip())
            for w_idx, word in enumerate(tokens):
                split_word = self.split_word_on_characters_with_diacritics(word)
                _, cy_flat, _ = self.create_label_for_word(split_word)

                gt_labels_line.extend(cy_flat)
                if w_idx+1 < len(tokens):
                    gt_labels_line += [0]

            gt_labels += [gt_labels_line]
        return gt_labels
            
    def get_ce(self, diac_word_y, e_idx=None, return_idx=False):
        #^ diac_word_y: [Tw 3]
        if e_idx is None: e_idx = len(diac_word_y)
        for c_idx in reversed(range(e_idx)):
            if diac_word_y[c_idx] != [0,0,0]:
                return diac_word_y[c_idx] if not return_idx else c_idx
        return diac_word_y[e_idx-1] if not return_idx else e_idx-1

    def create_decoder_input(self, diac_code_y, prob=0):
        #^ diac_code_y: [Ts Tw 3]
        diac_code_x = np.zeros((*np.array(diac_code_y).shape[:-1], 8))
        if not self.markov_signal:
            return list(diac_code_x)
        prev_ce = list(np.eye(6)[-1]) + [0,0] # bos tag
        for w_idx, word in enumerate(diac_code_y):
            diac_code_x[w_idx, 0, :] = prev_ce
            for c_idx, char in enumerate(word[:-1]):
                # if np.random.rand() < prob: 
                #     continue 
                if char[0] == self.pad_target_val: 
                    break
                haraka = list(np.eye(6)[char[0]])
                diac_code_x[w_idx, c_idx+1, :] = haraka + char[1:]
            ce = self.get_ce(diac_code_y[w_idx], c_idx)
            prev_ce = list(np.eye(6)[ce[0]]) + ce[1:]
        return list(diac_code_x)
