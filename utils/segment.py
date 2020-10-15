import argparse
import yaml
import os
import pickle as pkl

from tqdm import tqdm
from pyarabic.araby import tokenize, strip_tashkeel

def export(path, text):
    with open(path, 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(text))
    
def segment(lines, stride, window_sz, min_window_sz):
    segments, mapping = [], []
    real_seg_idx = 0

    for sent_idx, line in tqdm(enumerate(lines), total=len(lines)):
        line = line.strip()
        tokens = tokenize(line)
        if len(tokens) == 0: continue
        if tokens[-1] == '\n': tokens = tokens[:-1]
        seg_idx, idx = 0, 0
        while idx < len(tokens):
            window = tokens[idx:idx+window_sz]
            if window_sz == -1: window = tokens  
            if len(window) < min_window_sz and seg_idx != 0: break

            segment = ' '.join(window)
            segments += [segment]
            char_offset = len(strip_tashkeel(' '.join(tokens[:idx])))
    
            if seg_idx > 0:
                char_offset += 1

            seg_tokens = tokenize(strip_tashkeel(segment))

            j = 0
            for st_idx, st in enumerate(seg_tokens):
                for _ in range(len(st)):
                    mapping += [(sent_idx, real_seg_idx, st_idx, j+char_offset)]
                    j += 1
                j += 1

            real_seg_idx += 1
            seg_idx += 1

            if stride == -1: break

            idx += (window_sz if stride >= window_sz else stride)
          
    return segments, mapping

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sentence Breaker')
    parser.add_argument('-c', '--config',  type=str,
                        default="config.yaml", help='Run Configs')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    BASE_PATH = config["paths"]["base"]

    stride = config["segment"]["stride"]
    window = config["segment"]["window"]
    min_window = config["segment"]["min-window"]
    export_map = config["segment"]["export-map"]

    for fpath in tqdm(config["segment"]["files"]):
        FILE_PATH = os.path.join(BASE_PATH, fpath)
        SAVE_PATH = os.path.join(BASE_PATH, fpath[:-4] + f"-{stride}-{window}.txt")
        MAP_PATH  = os.path.join(BASE_PATH, fpath[:-4] + f"-{stride}-{window}.map")

        with open(FILE_PATH, 'r', encoding="utf-8") as fin:
            lines = fin.readlines()

        segments, mapping = segment(lines, stride, window, min_window)

        with open(SAVE_PATH, 'w', encoding="utf-8") as fout:
            fout.write('\n'.join(segments))
        
        if not export_map: continue

        with open(MAP_PATH, 'w', encoding="utf-8") as fout:
            for sent_idx, seg_idx, word_idx, char_idx in mapping:
                fout.write(f"{sent_idx}, {seg_idx}, {word_idx}, {char_idx}\n")
