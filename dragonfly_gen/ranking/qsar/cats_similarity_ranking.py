#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (c) 2023 Kenneth Atz, Clemens Isert & Gisbert Schneider (ETH Zurich)
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from dragonfly_gen.ranking.qsar.cats2d import CATS2D

cats2d = CATS2D()


def get_cats_sim(smiles_list, query, pbar=False):
    ###  calculate descriptors of input list
    cats_dist = []
    smiles = []

    query_mol = Chem.MolFromSmiles(query)
    query_cats = np.array(cats2d.getCATs2D(mol=query_mol))

    if pbar:
        smiles_list = tqdm(smiles_list)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        try:
            cat = np.array(cats2d.getCATs2D(mol=mol))
            dist = np.linalg.norm(cat - query_cats)
            dist = np.linalg.norm(query_cats - cat)
            cats_dist.append(dist)
            smiles.append(smi)
        except:
            pass

    return smiles, cats_dist


if __name__ == "__main__":

    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-smi_file", type=str, default="../../genfromligand/output/rosiglitazone.csv"
    )
    parser.add_argument(
        "-query",
        type=str,
        default="CN(CCOC1=CC=C(C=C1)CC2C(=O)NC(=O)S2)C3=CC=CC=N3",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.smi_file)
    smiles_list = list(df["SMILES"])

    smiles, cats_dist = get_cats_sim(
        smiles_list=smiles_list,
        query=args.query,
        pbar=True,
    )

    # Save predictions
    df = pd.DataFrame({"SMILES": smiles, "cats_euclidean_dist": cats_dist})
    df_sorted = df.sort_values(by=["cats_euclidean_dist"], ascending=True)
    output_file = f"{args.smi_file[:-4]}_cats.csv"
    print(f"The ranked molecule are stored here: {output_file}")
    df_sorted.to_csv(output_file, index=False)
