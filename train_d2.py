import argparse
import os
import sys

import yaml
import numpy as np

import torch as T

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_d2 import DiacritizerD2
from data_utils import DatasetUtils
from dataloader import DataRetriever

SEED = 1337
T.random.manual_seed(SEED)
T.manual_seed(SEED)
np.random.seed(SEED)

class Trainer:
    def __init__(self, config, device=T.device('cuda')):
        self.device = device
        self.config = config

        self.debug = config["debug"]
        self.run_title = config["run-title"]
        self.save_path = config["paths"]["save"]
        self.embs_path = config["paths"]["word-embs"]

        self.lr_factor = config["train"]["lr-factor"]
        self.lr_patience = config["train"]["lr-patience"]
        self.min_lr = config["train"]["lr-min"]
        self.init_lr = config["train"]["lr-init"]
        self.optimizer_name = config['train'].get('optimizer', 'adam').lower()
        self.weight_decay_ = config["train"]["weight-decay"]
        self.best_loss = np.inf
        self.stopping_counter = 0

        self.stopping_delta = config["train"]["stopping-delta"]
        self.stopping_patience = config["train"]["stopping-patience"]

        self.data_utils = DatasetUtils(config)
        vocab_size = len(self.data_utils.letter_list)
        word_embeddings = self.data_utils.embeddings

        self.model = DiacritizerD2(config, self.device)
        self.model.build(word_embeddings, vocab_size)
        self.resume = config['train']['resume']
        self.start_epoch = 0
        self.epochs = config["train"]["epochs"]

        if self.resume:
            model_data = T.load(config["paths"]["resume"], map_location=T.device(self.device))
            state_dict = model_data['state_dict']
            self.start_epoch = model_data['last_epoch'] + 1
            if config["train"]["resume-lr"]:
                 self.init_lr = model_data["last_lr"]
                 self.best_loss = model_data["val_loss"]

            self.model.load_state_dict(state_dict)
            print(f"> Loading Checkpoint {config['paths']['resume']} with Best Val-Loss: {self.best_loss:.4f} and Stopping Counter {self.stopping_counter}")
        
        print(f"> Optimizer {self.optimizer_name} with initial LR {self.init_lr}")
        if self.optimizer_name == 'adamw':
            self.optimizer = T.optim.AdamW(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay_)
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = T.optim.RMSprop(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay_)
        else:
            self.optimizer = T.optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay_)
            
        print("> Creating Dataloaders")

        train_set = "train-small" if "small" in self.embs_path else "train"
        self.train_loader = DataLoader(
            DataRetriever(train_set, self.data_utils),
            batch_size=config["train"]["batch-size"],
            shuffle=True,
            num_workers=config["loader"]["num-workers"]
        )

        self.val_loader = DataLoader(
            DataRetriever("val", self.data_utils),            
            batch_size=min(config["train"]["batch-size"], 128),
            shuffle=False,
            num_workers=config["loader"]["num-workers"]
        )

        self.model.to(self.device)
        print(self.model)

    def train(self, epoch):
        n_batches = len(self.train_loader)
        train_loss = np.zeros(n_batches)
        progress = tqdm(range(n_batches), ascii=False, dynamic_ncols=True, unit_scale=True)
        self.model.train()
        for idx, (inputs, outputs) in enumerate(self.train_loader):
    
            self.optimizer.zero_grad()
            loss = self.model.step(inputs, outputs, None)
            loss.backward()
            self.optimizer.step() 

            train_loss[idx] = loss.item()

            progress.update()
            progress.set_description(
                    f"Training: Epoch [{epoch}/{self.epochs}], Loss: {train_loss[:idx+1].mean():.4f} | ({idx}/{n_batches})")

        return train_loss.mean()

    def validate(self, epoch):
        n_batches = len(self.val_loader)
        val_loss = np.zeros(n_batches)
        progress = tqdm(range(n_batches), ascii=False, dynamic_ncols=True, unit_scale=True)
        self.model.eval()
        for idx, (inputs, outputs) in enumerate(self.val_loader):
            
            loss = self.model.step(inputs, outputs, None)
            val_loss[idx] = loss.item()

            progress.update()
            progress.set_description(
                    f"Validating: Epoch [{epoch}/{self.epochs}], Loss: {val_loss[:idx+1].mean():.4f} | ({idx}/{n_batches})")

        return val_loss.mean()

    def run(self):
       
        filepath = os.path.join(self.save_path, self.run_title, self.run_title+"_{epoch:02d}_{val_loss:.4f}.pt")
        best_filepath = os.path.join(self.save_path, self.run_title, self.run_title+".best.pt")

        reduce_lr = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.lr_factor, patience=self.lr_patience, min_lr=self.min_lr)
        writer = SummaryWriter(log_dir=os.path.join("logs", self.run_title))

        print("[INFO] Training Model:", self.run_title)
        
        for epi in range(self.start_epoch, self.epochs):

            train_loss = self.train(epi)
            writer.add_scalar("loss/train", train_loss, epi)

            val_loss = self.validate(epi)
            writer.add_scalar("loss/val", val_loss, epi)

            reduce_lr.step(val_loss)
            last_lr = reduce_lr._last_lr[0] if hasattr(reduce_lr, "_last_lr") else self.init_lr
            writer.add_scalar("lr", last_lr, epi)

            if (val_loss < self.best_loss) and self.save_path and not self.debug:
                model_data = {
                    'last_epoch': epi,
                    'last_lr': last_lr,
                    'val_loss': val_loss,
                    'stopping_counter': self.stopping_counter,
                    'state_dict': self.model.state_dict()
                }
                T.save(model_data, filepath.format(epoch=epi, val_loss=val_loss))
                if os.path.isfile(best_filepath):
                    os.remove(best_filepath)
                T.save(model_data, best_filepath)
                self.best_loss = val_loss
                self.stopping_counter = 0
            else:
                self.stopping_counter += 1


            if self.stopping_counter > self.stopping_patience:
                break
                

            print(f"\nEpoch [{epi}/{self.epochs}] | Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {self.best_loss:.4f} | LR: {last_lr} | DDO: {self.model.diac_dropout_p}")
            writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config.yaml", help='path of config file')
    parser.add_argument('--skip',  action='store_true',
                        help='Skip existing run')
    parser.add_argument('--force',  action='store_true',
                        help='Force reruns')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run_title = config["run-title"]
    exp_path = os.path.join(config["paths"]["save"], config["run-title"])
    if not os.path.isdir(exp_path): os.mkdir(exp_path)

    if not args.force:
        if os.path.isfile(config["paths"]["load"]) and not config["train"]["resume"] and not config["debug"]:
            if not args.skip:
                print('[WARNING] Found existing run with name: ', run_title)
                cont = input('Continue?')
                if cont.lower() not in ['y', 'yes']:
                    sys.exit(0)
            else:
                sys.exit(0)

    with open(os.path.join(exp_path, "config.yaml"), 'w', encoding="utf-8") as fout:
        yaml.dump(config, fout)

    model = Trainer(config)
    model.run()
