import os 
import yaml 
import pickle
import argparse
import numpy as np 

from tqdm import tqdm
from pyarabic.araby import tokenize, strip_tashkeel

def extract_vocab(f_name):
    vocab = set()
    with open(f_name, 'r', encoding="utf-8") as fin:
        lines = fin.readlines()
    for line in tqdm(lines):
        vocab.update(
            strip_tashkeel(t)
            for t in tokenize(line)
        )
    return vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config', type=str,
                        default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    f_name = config["paths"]["base"]

    v_train = extract_vocab(os.path.join(f_name, "train", "train.txt"))
    v_val = extract_vocab(os.path.join(f_name, "val", "val.txt"))
    v_test = extract_vocab(os.path.join(f_name, "test", "test.txt"))

    vocab = v_train.union(v_val).union(v_test)

    with open(os.path.join(f_name, "vocab.txt"), 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(vocab))