#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (c) 2023 Kenneth Atz, Clemens Isert & Gisbert Schneider (ETH Zurich)
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics.pairwise import manhattan_distances

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT_PATH, "config/")
MODEL_PATH = os.path.join(ROOT_PATH, "models/")


ACT = nn.Softmax(dim=2)


def get_vocab():
    vocab_list = [
        "x",
        "y",
        "z",
        "C",
        "c",
        "N",
        "n",
        "S",
        "s",
        "P",
        "J",
        "O",
        "o",
        "F",
        "I",
        "/",
        "\\",
        "Q",
        "R",
        "r",
        "D",
        "d",
        "G",
        "g",
        "M",
        "m",
        "A",
        "a",
        "T",
        "t",
        "E",
        "K",
        "k",
        "L",
        "l",
        "U",
        "V",
        "v",
        "w",
        "W",
        "X",
        "Y",
        "Z",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "#",
        "=",
        "-",
        "(",
        ")",
    ]

    return sorted(vocab_list)


def selfies_vocab():
    HERE = os.path.abspath(os.path.dirname(__file__))
    vocab_list = torch.load(os.path.join(HERE, "data/selfies_vocab.pt"))

    return sorted(vocab_list)


ATOMDICT = {
    "Cl": 8,
    "Br": 9,
    "H": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "S": 7,
    "P": 6,
    "I": 10,
}

AROMDICT = {
    "True": 1,
    "False": 0,
}

RINGDICT = {
    "True": 1,
    "False": 0,
}

HYBDICT = {
    "SP3": 3,
    "SP2": 2,
    "SP": 1,
    "UNSPECIFIED": 0,
    "S": 0,
}

REPLACEMENTS = {
    "Cl": "X",
    "[nH]": "Y",
    "Br": "Z",
    "[N+](=O)[O-]": "V",
    "O=[N+][O-]": "v",
    "[O-][N+](=O)": "w",
    "S(=O)(=O)": "W",
    "[C@@H]": "U",
    "[C@H]": "Q",
    "[C@@]": "R",
    "[C@]": "T",
    "[N-]=[N+]=N": "J",
    "[O-]": "K",
    "[n+]": "L",
    "[n-]": "l",
    "[N+]": "M",
    "[N-]": "m",
    "[s+]": "G",
    "[S-]": "g",
    "[S+]": "r",
    "[S@@]": "E",
    "[S@]": "E",
    "=O": "D",
    "O=": "d",
    "C#N": "A",
    "N#C": "a",
    "[H]": "k",
    "[C-]": "t",
}

REP_SORTED = sorted(REPLACEMENTS, key=len, reverse=True)
REP_ESCAPED = map(re.escape, REP_SORTED)
RE_PATTERN = re.compile("|".join(REP_ESCAPED))

inv_map = {v: k for k, v in REPLACEMENTS.items()}
INV_REP_SORTED = sorted(inv_map, key=len, reverse=True)
INV_REP_ESCAPED = map(re.escape, INV_REP_SORTED)
INV_RE_PATTERN = re.compile("|".join(INV_REP_ESCAPED))


def functional_group_embedding(smiles):
    return RE_PATTERN.sub(lambda match: REPLACEMENTS[match.group(0)], smiles)


def functional_group_decoding(smiles):
    return INV_RE_PATTERN.sub(lambda match: inv_map[match.group(0)], smiles)


def randomSmiles(m1):
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


def clean_molecule(m):
    end_ind = m.find("z")
    m = [m[:end_ind]]
    m = np.array([d.replace("y", "") for d in m])
    return m[0]


def decode_ids(
    nums=[
        16,
        27,
        16,
        1,
        17,
        2,
        41,
        5,
        41,
        41,
        41,
        6,
        41,
        1,
        41,
        5,
        2,
        16,
    ]
):
    tokens = get_vocab()
    string_list = [tokens[i] for i in nums]
    string_list = "".join(string_list)
    string_list = functional_group_decoding(string_list)
    return string_list


def clean_selfies(m):
    end_ind = m.find("[\\Z]")
    m = [m[:end_ind]]
    m = np.array([d.replace("[\\Y]", "") for d in m])
    return m[0]


def decode_selfies(
    nums=[
        16,
        27,
        16,
        1,
        17,
        2,
        41,
        5,
        41,
        41,
        41,
        6,
        41,
        1,
        41,
        5,
        2,
        16,
    ]
):
    tokens = selfies_vocab()
    string_list = [tokens[i] for i in nums]
    string_list = "".join(string_list)
    return string_list


def print_model_info(model):
    # Calculate model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    sum_params = sum([np.prod(e.size()) for e in model_parameters])
    print("\nModel architecture: ", model)
    print("\nNum model parameters: ", sum_params)
    print("\nLoop over model-weights:")
    for name, e in model.state_dict().items():
        print(name, e.shape)


def jaccard_binary(x, y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity


def get_laplacian(train, test, sig):
    return np.exp(-(manhattan_distances(train, test) / sig))


def get_novel_molecules(gens, train):
    novels = [v for v in gens if v not in train]
    return novels


def get_murcko(gens, train):
    muckro = [
        Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmilesFromSmiles(i) for i in gens
    ]
    num_scaffold_in_train = [train.count(i) for i in muckro]
    return muckro, num_scaffold_in_train


def get_murcko_generic(gens, train):
    murcko_generic = []
    for i in gens:
        try:
            murcko_generic.append(
                Chem.MolToSmiles(
                    Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric(
                        Chem.MolFromSmiles(i)
                    )
                )
            )
        except:
            murcko_generic.append("O")

    num_scaffold_in_train = [train.count(i) for i in murcko_generic]

    return murcko_generic, num_scaffold_in_train


def get_diversity(smiles_list, sim_dist):
    ecfps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        ecfps.append(np.array(ecfp))

    num_sims = []
    total_similarities = []
    for i in ecfps:
        similarities = []
        for j in ecfps:
            similarities.append(jaccard_binary(i, j))
        num_sims.append((np.array(similarities) > sim_dist).sum())
        total_similarities.append((np.sum(similarities) - 1) / len(ecfps))

    return 1 - np.mean(total_similarities), 1 - (np.mean(num_sims) / len(smiles_list))