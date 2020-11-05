import argparse
import os
from collections import Counter

import torch as T
import yaml
from pyarabic.araby import tokenize
from tqdm import tqdm

from torch.utils.data import DataLoader

from model_d2   import DiacritizerD2
from data_utils import DatasetUtils
from dataloader import DataRetriever

DEVICE = 'cuda'
TEST_FILE = "test"

class Predictor: 
    def __init__(self, config):

        self.data_utils = DatasetUtils(config)
        vocab_size = len(self.data_utils.letter_list)
        word_embeddings = self.data_utils.embeddings

        self.mapping = self.data_utils.load_mapping_v3(TEST_FILE)
        self.original_lines = self.data_utils.load_file_clean(TEST_FILE, strip=True)

        self.model = DiacritizerD2(config, device=DEVICE)
        self.model.build(word_embeddings, vocab_size)
        state_dict = T.load(config["paths"]["load"], map_location=T.device(DEVICE))['state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        self.data_loader = DataLoader(
            DataRetriever(TEST_FILE, self.data_utils, is_test=True),
            batch_size=min(config["predictor"]["batch-size"], 128),
            shuffle=False,
            num_workers=16
        )

class PredictTri(Predictor):
    def __init__(self, config):
        super().__init__(config) 
        self.diacritics = {
            "FATHA": 1,
            "KASRA": 2,
            "DAMMA": 3,
            "SUKUN": 4
        }

    def shakkel_char(self, diac: int, tanween: bool, shadda: bool) -> str:
        returned_text = ""
        if shadda and diac != self.diacritics["SUKUN"]:
            returned_text += "\u0651"

        if diac == self.diacritics["FATHA"]:
            returned_text += "\u064E" if not tanween else "\u064B"
        elif diac == self.diacritics["KASRA"]:
            returned_text += "\u0650" if not tanween else "\u064D"
        elif diac == self.diacritics["DAMMA"]:
            returned_text += "\u064F" if not tanween else "\u064C"
        elif diac == self.diacritics["SUKUN"]:
            returned_text += "\u0652"

        return returned_text

    def predict_mv(self):

        y_gen_diac, y_gen_tanween, y_gen_shadda = self.model.predict(self.data_loader)
        
        diacritized_lines = []
        for sent_idx, line in tqdm(enumerate(self.original_lines), total=len(self.original_lines)):
            diacritized_line = ""
            line = ' '.join(tokenize(line))
            for char_idx, char in enumerate(line):
                diacritized_line += char
                char_vote_haraka, char_vote_shadda, char_vote_tanween = [], [], []
                if sent_idx not in self.mapping: continue
                for seg_idx in self.mapping[sent_idx]:
                    for t_idx in self.mapping[sent_idx][seg_idx]:                        
                        if char_idx in self.mapping[sent_idx][seg_idx][t_idx]:
                            c_idx = self.mapping[sent_idx][seg_idx][t_idx].index(char_idx)
                            char_vote_haraka  += [y_gen_diac[seg_idx][t_idx][c_idx]]
                            char_vote_shadda  += [y_gen_shadda[seg_idx][t_idx][c_idx]]
                            char_vote_tanween += [y_gen_tanween[seg_idx][t_idx][c_idx]]

                if len(char_vote_haraka) > 0:
                    char_mv_diac = Counter(char_vote_haraka).most_common()[0][0]
                    char_mv_shadda = Counter(char_vote_shadda).most_common()[0][0]
                    char_mv_tanween = Counter(char_vote_tanween).most_common()[0][0]
                    diacritized_line += self.shakkel_char(char_mv_diac, char_mv_tanween, char_mv_shadda)
            
            diacritized_lines += [diacritized_line.strip()]
        return diacritized_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config', type=str,
                        default="configs/config_d2.yaml", help='path of config file')
    args = parser.parse_args()


    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["train"]["max-sent-len"] = config["predictor"]["window"]
        
    predictor = PredictTri(config)
    diacritized_lines = predictor.predict_mv()

    exp_id = config["run-title"].split("-")[-1].lower()

    with open(os.path.join(config["paths"]["base"], 'preds', f'predictions_{exp_id}.txt'), 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(diacritized_lines))
