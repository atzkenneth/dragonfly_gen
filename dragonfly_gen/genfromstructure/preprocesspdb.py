#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (c) 2023 Kenneth Atz, Clemens Isert & Gisbert Schneider (ETH Zurich)
import argparse
import warnings

import h5py
import numpy as np
import torch
from Bio import PDB, BiopythonWarning
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

warnings.simplefilter("ignore", BiopythonWarning)

def get_xyzs_from_sdf(sdf_path):
    print(f"Loading ligand from SDF: {sdf_path}.")
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    mol = next(supplier)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {sdf_path}.")

    print("Extracting 3D coordinates from ligand.")
    xyzs = []
    for idx in tqdm(range(mol.GetNumAtoms()),
                    desc="Ligand atoms", unit="atom"):
        pos = mol.GetConformer().GetAtomPosition(idx)
        xyzs.append([pos.x, pos.y, pos.z])
    xyzs = np.array(xyzs)

    print("Computing ligand properties.")
    mol_noHs = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol_noHs)
    mol_base = Chem.MolFromSmiles(smiles)
    props = [
        rdMolDescriptors.CalcExactMolWt(mol_base) / 610.0,
        rdMolDescriptors.CalcNumRotatableBonds(mol_base) / 17.0,
        rdMolDescriptors.CalcNumHBA(mol_base) / 10.0,
        rdMolDescriptors.CalcNumHBD(mol_base) / 5.0,
        rdMolDescriptors.CalcTPSA(mol_base) / 173,
        Crippen.MolLogP(mol_base) / 7.5,
    ]
    properties = np.array([props])

    # Print out the computed properties for logging
    print("Ligand properties:")
    print(f"  - ExactMolWt: {props[0]*610:.4f}")
    print(f"  - NumRotatableBonds: {props[1]*17:.4f}")
    print(f"  - NumHBA: {props[2]*5:.4f}")
    print(f"  - NumHBD: {props[3]*10:.4f}")
    print(f"  - TPSA: {props[4]*173:.4f}")
    print(f"  - LogP: {props[5]*7.5:.4f}")

    print("Finished reading ligand.")
    return xyzs, properties

def get_info_from_pdb(
    pdb_path,
    mol_path,
    radius,
    pocket_radius,
    pdb_atom_dict,
    pdb_res_dict,
):
    print(f"Parsing PDB structure: {pdb_path}.")
    parser = PDB.PDBParser()
    struct = parser.get_structure(pdb_path[-15:-11], pdb_path)

    print("Gathering PDB atoms into list.")
    all_atoms = []
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    all_atoms.append((residue, atom))
    print(f"Total atoms to process: {len(all_atoms)}.")

    xyzs, resnames, atomids, atomnames, b_factors = [], [], [], [], []
    for residue, atom in tqdm(all_atoms, desc="PDB atoms", unit="atom"):
        resnames.append(residue.get_resname())
        coord = atom.get_coord()
        xyzs.append(coord.tolist())
        b_factors.append(atom.get_bfactor())
        atomids.append(atom.element)
        atomnames.append(atom.get_name())

    embeddings = [f"{resnames[i]}{atomnames[i]}" for i in range(len(resnames))]

    mol_xyzs, properties = get_xyzs_from_sdf(mol_path)

    print("Computing distances to ligand atoms.")
    distances = []
    for p in tqdm(xyzs, desc="Distance calc", unit="atom"):
        dists = [np.linalg.norm(p - m) for m in mol_xyzs]
        distances.append(min(dists))

    print(f"Filtering atoms within pocket radius = {pocket_radius} Å.")
    mask = np.array(distances) <= pocket_radius
    xyzs = np.array(xyzs)[mask]
    atomids = np.array(atomids)[mask]
    embeddings = np.array(embeddings)[mask]
    b_factors = np.array(b_factors)[mask]
    distances = 1 / np.array(distances)[mask]

    print("Mapping atom and residue names to indices.")
    xyzs_passed, atomids_passed, embeddings_passed, b_factors_passed, distances_passed = ([] for _ in range(5))
    for emb, aid, coord, bf, dist in zip(embeddings, atomids, xyzs, b_factors, distances):
        try:
            emb_idx = pdb_res_dict.get(emb, pdb_res_dict.get(emb[:4]))
            aid_idx = pdb_atom_dict[aid]
            xyzs_passed.append(coord)
            atomids_passed.append(aid_idx)
            embeddings_passed.append(emb_idx)
            b_factors_passed.append(bf)
            distances_passed.append(dist)
        except KeyError:
            continue

    xyzs = np.array(xyzs_passed)
    atomids = np.array(atomids_passed)
    embeddings = np.array(embeddings_passed)
    b_factors = np.array(b_factors_passed)
    distances = np.array(distances_passed)

    print("Building graph edges based on radius.")
    dm = squareform(pdist(xyzs))
    np.fill_diagonal(dm, np.inf)
    edge_tmp = np.vstack(np.where(dm <= radius))
    edge1, edge2 = edge_tmp.tolist()

    print("Adding centroid nodes.")
    num_atoms = len(atomids)
    for cent in range(num_atoms, num_atoms + 8):
        for i in range(num_atoms):
            edge1.append(i)
            edge2.append(cent)

    edge_5 = torch.tensor([edge1, edge2])

    centroid_nodes = [f"Cent{i+1}" for i in range(8)]
    atomids = np.concatenate([atomids, [pdb_atom_dict[c] for c in centroid_nodes]])
    embeddings = np.concatenate([embeddings, [pdb_res_dict[c] for c in centroid_nodes]])
    b_factors = np.concatenate([b_factors, np.zeros(8)])
    distances = np.concatenate([distances, np.full(8, 0.5)])

    centroid_coord = xyzs.mean(axis=0)
    xyzs = np.vstack([xyzs, np.tile(centroid_coord, (8, 1))])

    print(f"Graph built: {xyzs.shape[0]} nodes, {edge_5.shape[1]} edges.")
    return atomids, xyzs, edge_5, embeddings, distances, b_factors, properties


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein-ligand pocket graph with progress bars.")
    parser.add_argument("-pdb_file", type=str, default="3g8i_protein")
    parser.add_argument("-mol_file", type=str, default="3g8i_ligand")
    parser.add_argument("-pdb_key", type=str, default="3g8i")
    args = parser.parse_args()

    pdb_path = f"input/{args.pdb_file}.pdb"
    mol_path = f"input/{args.mol_file}.sdf"
    pdb_key = args.pdb_key

    print("Loading dictionary mappings.")
    pdb_atom_dict = torch.load("../drugtargetgraph/data/pdb_atom_dict.pt")
    pdb_res_dict = torch.load("../drugtargetgraph/data/pdb_res_dict.pt")

    print("Starting pocket graph extraction.")
    atomids, xyzs, edge_5, embeddings, distances, b_factors, properties = get_info_from_pdb(pdb_path, mol_path, radius=5.0, pocket_radius=6.0, pdb_atom_dict=pdb_atom_dict, pdb_res_dict=pdb_res_dict)

    print(f"Result: {len(embeddings)} residues in pocket.")
    output_file = f"input/{pdb_key}.h5"
    print(f"Writing results to {output_file}.")
    with h5py.File(output_file, "w") as container:
        grp = container.create_group(pdb_key)
        grp.create_dataset("embeddings", data=embeddings)
        grp.create_dataset("b_factors", data=b_factors)
        grp.create_dataset("atomids", data=atomids)
        grp.create_dataset("xyzs", data=xyzs)
        grp.create_dataset("edge_5", data=edge_5)
        grp.create_dataset("distances", data=distances)
        grp.create_dataset("properties", data=properties)