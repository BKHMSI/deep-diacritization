import os 
import yaml 
import argparse

from tqdm import tqdm

import fasttext as ft

def read_vocab(fn):
    with open(fn, encoding='utf-8') as fin:
        vv = [w.strip() for w in tqdm(fin) if w.strip()]
    return vv

def embed_vocab(ff, vv):
    return [
        ff.get_word_vector(w)
        for w in tqdm(vv)
    ]

def vround(val):
    return str(round(val, 4))

def render(word, feats):
    ff = ' '.join(map(vround, feats))
    return f'{word} {ff}\n'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config', type=str,
                        default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    base = config["paths"]["base"]
    save_path = config["paths"]["word-embs"]

    words = sorted(read_vocab(os.path.join(base, "vocab.txt")))

    path = config["paths"]["ft-bin"]

    ember = ft.load_model(os.path.join(base, path))
    vocab = embed_vocab(ember, words)
    del ember

    with open(save_path, 'w', encoding="utf-8") as fout:
        fout.write(f'{len(vocab)} 300\n')
        fout.writelines(
            render(word, feats)
            for word, feats in zip(tqdm(words), vocab)
        )