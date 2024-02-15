#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (c) 2023 Kenneth Atz, Clemens Isert & Gisbert Schneider (ETH Zurich)
import numpy as np
import argparse
import pandas as pd
import torch
from dragonfly_gen.drugtargetgraph.utils import jaccard_binary
from dragonfly_gen.ranking.qsar.cats_similarity_ranking import get_cats_sim
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm


def get_novelty(smiles_list, training):

    training_set = torch.load(training)

    train_ids = []
    train_smis = []
    train_ecfps = []

    for m in training_set:
        smi, ecfp = training_set[m][0], training_set[m][1]
        train_ids.append(m)
        train_smis.append(smi)
        train_ecfps.append(ecfp)

    closest_struc = []
    closest_id = []
    distances = []

    train_ids = train_ids 
    train_smis = train_smis 
    train_ecfps = train_ecfps 

    print("Calculating jaccard distances: ")
    for smi in tqdm(smiles_list):
        dist = 0
        mol = Chem.MolFromSmiles(smi)
        ecfp = np.array(
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        )

        for i, tid in enumerate(train_ids):
            dist_ij = jaccard_binary(train_ecfps[i], ecfp)
            if dist_ij >= dist:
                dist = dist_ij

                selected_id = tid
                selected_struc = train_smis[i]

        closest_struc.append(selected_struc)
        closest_id.append(selected_id)
        distances.append(dist)
        # print(selected_struc, selected_id, 1 - dist)

    return (
        smiles_list,
        np.array(closest_struc),
        np.array(closest_id),
        np.array(distances),
    )


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

    # Calculate CATS similarity
    smiles, cats_dist = get_cats_sim(
        smiles_list=smiles_list,
        query=args.query,
        pbar=True,
    )
    
    # Calculate novelty
    train_fps_path = "../../drugtargetgraph/data/novelity4ranking_genfromligand.pt"
    
    smiles_list, closest_struc, closest_id, distances = get_novelty(
        smiles_list=smiles, training=train_fps_path,
    )

    # Save predictions
    df = pd.DataFrame(
        {
            "SMILES": smiles, 
            "cats_euclidean_dist": cats_dist,
            "closest_struc": closest_struc,
            "closest_id": closest_id,
            "distances": distances,
        }
    )
    df_sorted = df.sort_values(by=["cats_euclidean_dist"], ascending=True)
    output_file = f"{args.smi_file[:-4]}_cats_novelty.csv"
    print(f"The ranked molecule are stored here: {output_file}")
    df_sorted.to_csv(output_file, index=False)