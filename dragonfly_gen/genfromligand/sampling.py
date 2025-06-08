#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (c) 2023 Kenneth Atz, Clemens Isert & Gisbert Schneider (ETH Zurich)
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors
from rdkit.rdBase import DisableLog

for level in RDLogger._levels:
    DisableLog(level)

import argparse
import configparser
import os
import random

import numpy as np
import pandas as pd
import selfies
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import trange

from dragonfly_gen.drugtargetgraph.utils import (
    ACT,
    AROMDICT,
    ATOMDICT,
    CONFIG_PATH,
    HYBDICT,
    MODEL_PATH,
    RINGDICT,
    clean_molecule,
    clean_selfies,
    decode_ids,
    decode_selfies,
    get_vocab,
    selfies_vocab,
)
from dragonfly_gen.genfromligand.net import EGNN, LSTM, GraphTransformer
from dragonfly_gen.gml.pygdataset import Dataset

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load args
parser = argparse.ArgumentParser()
parser.add_argument("-config", type=int, default=603)
parser.add_argument("-epoch", type=int, default=305)
parser.add_argument("-T", type=float, default=0.5)
parser.add_argument("-smi_id", type=str, default="rosiglitazone")
parser.add_argument(
    "-smi",
    type=str,
    default="CN(CCOC1=CC=C(C=C1)CC2C(=O)NC(=O)S2)C3=CC=CC=N3",
)
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
                    prob = np.squeeze(prob.cpu().detach().numpy())
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


class MolLoader(Dataset):
    def __init__(self, smi):
        self.smi = smi
        print(f"Here is your template SMILES: {smi}")

    def __getitem__(self, idx):
        # generate graph from input mol
        mol = Chem.AddHs(Chem.MolFromSmiles(self.smi))

        m_atomids = []
        m_is_ring = []
        m_hyb = []
        m_arom = []

        for i in mol.GetAtoms():
            try:
                m_atomids.append(ATOMDICT[i.GetSymbol()])
                m_is_ring.append(RINGDICT[str(i.IsInRing())])
                m_hyb.append(HYBDICT[str(i.GetHybridization())])
                m_arom.append(AROMDICT[str(i.GetIsAromatic())])
            except:
                print(f"Failed to get info from {Chem.MolToSmiles(mol)}")

        m_atomids = np.array(m_atomids)
        m_is_ring = np.array(m_is_ring)
        m_hyb = np.array(m_hyb)
        m_arom = np.array(m_arom)

        edge_dir1 = []
        edge_dir2 = []
        for idx, bond in enumerate(mol.GetBonds()):
            a2 = bond.GetEndAtomIdx()
            a1 = bond.GetBeginAtomIdx()
            edge_dir1.append(a1)
            edge_dir1.append(a2)
            edge_dir2.append(a2)
            edge_dir2.append(a1)

        m_edge_index = torch.from_numpy(np.array([edge_dir1, edge_dir2]))
        m_num_nodes = torch.LongTensor(m_atomids).size(0)

        weight = rdMolDescriptors.CalcExactMolWt(mol) / 610.0
        num_rot_bond = rdMolDescriptors.CalcNumRotatableBonds(mol) / 17.0
        hba = rdMolDescriptors.CalcNumHBA(mol) / 10.0
        hbd = rdMolDescriptors.CalcNumHBD(mol) / 5.0
        tpsa = rdMolDescriptors.CalcTPSA(mol) / 173
        logp = Crippen.MolLogP(mol) / 7.5
        properties = [weight, num_rot_bond, hba, hbd, tpsa, logp]
        properties = np.array([properties])

        print(f"weight      : {weight*610:.4f}")
        print(f"num_rot_bond: {num_rot_bond*17:.4f}")
        print(f"HBA         : {hba*10:.4f}")
        print(f"HBD         : {hbd*5:.4f}")
        print(f"TPSA        : {tpsa*173:.4f}")
        print(f"logP        : {logp*7.5:.4f}")

        g_mol = Data(
            atomids=torch.LongTensor(m_atomids),
            is_ring=torch.LongTensor(m_is_ring),
            hyb=torch.LongTensor(m_hyb),
            arom=torch.LongTensor(m_arom),
            edge_index=torch.LongTensor(m_edge_index),
            properties=torch.FloatTensor(properties),
            num_nodes=m_num_nodes,
        )

        return g_mol

    def __len__(self):
        return len([0])


if __name__ == "__main__":
    # Define PDB id and args
    smi = args.smi
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
    g_mol = MolLoader(smi=smi)
    sample_loader = DataLoader(g_mol, batch_size=1, shuffle=True, num_workers=0)

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
    df = pd.DataFrame({"SMILES": novels, "log-likelihoodog": probs_abs})
    df_sorted = df.sort_values(by=["log-likelihoodog"], ascending=False)
    os.makedirs("output/", exist_ok=True)
    df_sorted.to_csv(f"output/{args.smi_id}.csv", index=False)
    print(f"Saved to output/{args.smi_id}.csv")
    df_smi = pd.DataFrame({"SMILES": list(df_sorted["SMILES"])})
    df_smi.to_csv(f"output/{args.smi_id}_smiles_only.csv", index=False)
    print(f"Saved to output/{args.smi_id}_smiles_only.csv")