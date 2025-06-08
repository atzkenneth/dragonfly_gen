#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (c) 2023 Kenneth Atz, Clemens Isert & Gisbert Schneider (ETH Zurich)
from rdkit import Chem, RDLogger
from rdkit.rdBase import DisableLog

for level in RDLogger._levels:
    DisableLog(level)

import argparse
import configparser
import os
import random

import h5py
import numpy as np
import pandas as pd
import selfies
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import trange

from dragonfly_gen.drugtargetgraph.utils import (
    ACT,
    CONFIG_PATH,
    MODEL_PATH,
    clean_molecule,
    clean_selfies,
    decode_ids,
    decode_selfies,
    get_vocab,
    selfies_vocab,
)
from dragonfly_gen.genfromstructure.net import EGNN, LSTM, GraphTransformer
from dragonfly_gen.gml.pygdataset import Dataset

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load args
parser = argparse.ArgumentParser()
parser.add_argument("-config", type=int, default=701)
parser.add_argument("-epoch", type=int, default=151)
parser.add_argument("-T", type=float, default=0.5)
parser.add_argument("-pdb", type=str, default="3g8i")
parser.add_argument("-num_mols", type=int, default=100)
args = parser.parse_args()

# Load config file
config = configparser.ConfigParser()
CONFIG_NAME = f"config_{args.config}.ini"
config.read(CONFIG_PATH + CONFIG_NAME)

# Load hyperparameters from config file and define globle variables
DROPOUT = float(config["PARAMS"]["DROPOUT"])
SMILES_LENGTH = int(config["PARAMS"]["SMILES_LENGTH"])
D_MODEL = int(config["PARAMS"]["D_MODEL"])
D_INNER = int(config["PARAMS"]["D_INNER"])
N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
PROP_DIM = int(config["PARAMS"]["PROP_DIM"])
RUN_TYPE = int(config["PARAMS"]["RUN_TYPE"])
POOLING_HEADS = int(config["PARAMS"]["POOLING_HEADS"])
N_LAYERS = 2

try:
    SELFIES = int(config["PARAMS"]["SELFIES"])
except:
    SELFIES = None

if SELFIES is not None:
    VOCAB = selfies_vocab()
else:
    VOCAB = get_vocab()

end_token = len(VOCAB) - 1
start_token = len(VOCAB) - 2
padding_token = len(VOCAB) - 3
vocab_i2c = {i: x for i, x in enumerate(VOCAB)}
vocab_c2i = {vocab_i2c[i]: i for i in vocab_i2c}
ALPHABET = end_token + 1

# Define paths
MODEL_PATH = os.path.join(MODEL_PATH, CONFIG_NAME)
CONFIG_NAME = CONFIG_NAME + "/"


def temperature_sampling(egnn, lstm, T, sample_loader, num_mols):
    egnn.eval()
    lstm.eval()

    smiles_list = []
    score_list = []

    for g in sample_loader:
        g = g.to(DEVICE)

        for _ in trange(num_mols):
            hiddens = egnn(g)
            score = 0

            with torch.no_grad():
                pred_smiles_list = [start_token]
                pred_smiles = torch.from_numpy(np.asarray([pred_smiles_list])).to(
                    DEVICE
                )

                for _ in range(SMILES_LENGTH - 1):
                    if isinstance(T, list):
                        temp = random.choice(T)
                    else:
                        temp = T

                    pred, hiddens = lstm(pred_smiles, hiddens)

                    # calculate propabilities
                    prob = ACT(pred)
                    prob = np.squeeze(pred.cpu().detach().numpy())
                    prob = prob.astype("float64")

                    # token selection using temperature T
                    pred = np.squeeze(pred.cpu().detach().numpy())
                    pred = pred.astype("float64")
                    pred = np.exp(pred / temp) / np.sum(np.exp(pred / temp))
                    pred = np.random.multinomial(1, pred, size=1)
                    pred = np.argmax(pred)
                    pred_smiles_list.append(pred)
                    pred_smiles = torch.LongTensor([[pred]]).to(DEVICE)

                    # calculate score (the higher the %, the smaller the log-likelihood)
                    # print(pred, prob[pred], max(prob), -np.log(prob[pred]))
                    score += +(-np.log(prob[pred]))

            try:
                if SELFIES is not None:
                    selfie = clean_selfies(decode_selfies(pred_smiles_list))
                    smiles_j = selfies.decoder(selfie)
                else:
                    smiles_j = clean_molecule(decode_ids(pred_smiles_list[1:]))
                smiles_j = Chem.CanonSmiles(smiles_j)
                mj = Chem.AddHs(Chem.MolFromSmiles(smiles_j))
                # print(smiles_j)
                smiles_list.append(smiles_j)
                score_list.append(score)
            except:
                pass

    novels = []
    probs_abs = []
    for idx, smil in enumerate(smiles_list):
        if smiles_list[idx] not in novels:
            novels.append(smiles_list[idx])
            probs_abs.append(score_list[idx])

    print(f"Number of valid, unique and novel molecules: {len(novels)}")

    return novels, probs_abs


class PocketLoader(Dataset):
    def __init__(self, pdbid, h5data):
        self.pdbid = pdbid

        data_tid = f"input/{h5data}"
        self.h5fs = h5py.File(data_tid, "r")

    def __getitem__(self, idx):
        if self.pdbid:
            pdb = self.pdbid
        else:
            pdb = self.pdb_list[0]

        # get graph from input pdb
        p_num_nodes = torch.LongTensor(self.h5fs[str(pdb)]["atomids"]).size(0)
        p_atomids = np.array(self.h5fs[str(pdb)]["atomids"])
        p_xyzs = np.array(self.h5fs[str(pdb)]["xyzs"])
        p_edge_index = np.array(self.h5fs[str(pdb)]["edge_5"])
        p_distances = np.array(self.h5fs[str(pdb)]["distances"])
        p_embeddings = np.array(self.h5fs[str(pdb)]["embeddings"])
        p_b_factors = np.array(self.h5fs[str(pdb)]["b_factors"])
        properties = np.array(self.h5fs[str(pdb)]["properties"])

        gp = Data(
            atomids=torch.LongTensor(p_atomids),
            residueids=torch.LongTensor(p_embeddings),
            coords=torch.FloatTensor(p_xyzs),
            b_factors=torch.FloatTensor(p_b_factors),
            distances=torch.FloatTensor(p_distances),
            edge_index=torch.LongTensor(p_edge_index),
            properties=torch.FloatTensor(properties),
            num_nodes=p_num_nodes,
        )

        return gp

    def __len__(self):
        return len(self.h5fs)


if __name__ == "__main__":
    # Define PDB id and args
    pdbid = args.pdb
    h5data = f"{args.pdb}.h5"

    epoch = str(args.epoch)
    num_mols = int(args.num_mols)
    T = args.T

    # Load models
    if RUN_TYPE == 0:
        lstm = LSTM(ALPHABET, D_INNER, N_LAYERS, DROPOUT, D_MODEL)
        egnn = EGNN(N_KERNELS, D_INNER, PROP_DIM, POOLING_HEADS)
    elif RUN_TYPE == 1:
        lstm = LSTM(ALPHABET, D_INNER, N_LAYERS, DROPOUT, D_MODEL)
        egnn = GraphTransformer(N_KERNELS, D_INNER, PROP_DIM, POOLING_HEADS)

    egnn_path = os.path.join(MODEL_PATH, f"egnn_{epoch}.pt")
    lstm_path = os.path.join(MODEL_PATH, f"lstm_{epoch}.pt")

    egnn.load_state_dict(torch.load(egnn_path, map_location="cpu"))
    lstm.load_state_dict(torch.load(lstm_path, map_location="cpu"))

    egnn = egnn.to(DEVICE)
    lstm = lstm.to(DEVICE)

    # Load PDB
    g_protein = PocketLoader(pdbid=pdbid, h5data=h5data)
    sample_loader = DataLoader(g_protein, batch_size=1, shuffle=True, num_workers=0)

    # Sample molecules
    print(f"Sampling {num_mols} molecules:")
    novels, probs_abs = temperature_sampling(
        egnn=egnn,
        lstm=lstm,
        T=T,
        sample_loader=sample_loader,
        num_mols=num_mols,
    )

    # Save predictions
    df = pd.DataFrame({"SMILES": novels, "log-likelihood": probs_abs})
    df_sorted = df.sort_values(by=["log-likelihood"], ascending=False)
    os.makedirs("output/", exist_ok=True)
    df_sorted.to_csv(f"output/{args.pdb}.csv", index=False)
    print(f"Saved to output/{args.pdb}.csv")
    df_smi = pd.DataFrame({"SMILES": list(df_sorted["SMILES"])})
    df_smi.to_csv(f"output/{args.pdb}_smiles_only.csv", index=False)
    print(f"Saved to output/{args.pdb}_smiles_only.csv")
