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

warnings.simplefilter("ignore", BiopythonWarning)


def get_xyzs_from_sdf(sdf_path):
    mol = next(Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False))
    xyzs = []

    for idx, i in enumerate(mol.GetAtoms()):
        xyzs.append(list(mol.GetConformer().GetAtomPosition(idx)))

    xyzs = np.array(xyzs)

    mol_noHs = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol_noHs)
    mol = Chem.MolFromSmiles(smiles)

    weight = rdMolDescriptors.CalcExactMolWt(mol) / 610.0
    num_rot_bond = rdMolDescriptors.CalcNumRotatableBonds(mol) / 17.0
    hba = rdMolDescriptors.CalcNumHBA(mol) / 5.0
    hbd = rdMolDescriptors.CalcNumHBD(mol) / 10.0
    tpsa = rdMolDescriptors.CalcTPSA(mol) / 173
    logp = Crippen.MolLogP(mol) / 7.5
    properties = [weight, num_rot_bond, hba, hbd, tpsa, logp]
    properties = np.array([properties])

    return xyzs, properties


def get_info_from_pdb(
    pdb_path,
    mol_path,
    radius,
    pocket_radius,
    pdb_atom_dict,
    pdb_res_dict,
):
    atomids = []
    xyzs = []
    res = []
    ids = []
    b_factors = []  # https://academic.oup.com/peds/article/27/11/457/1520829

    parser = PDB.PDBParser()
    struct = parser.get_structure(pdb_path[-15:-11], pdb_path)
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    res.append(residue.get_resname())
                    x, y, z = atom.get_coord()
                    xyzs.append([x, y, z])
                    b_factors.append(atom.get_bfactor())
                    atomids.append(atom.element)
                    ids.append(atom.get_name())

    embeddings = [str(res[i] + ids[i]) for i in range(len(res))]

    mol_xyzs, properties = get_xyzs_from_sdf(mol_path)

    distances = []
    for p in xyzs:
        dists = []
        for m in mol_xyzs:
            dist = np.linalg.norm(p - m)
            dists.append(float(dist))
        distances.append(min(dists))

    xyzs = [
        x for (x, d) in zip(xyzs, distances) if d <= pocket_radius
    ]  # 6.35012 = 12 a0
    atomids = [x for (x, d) in zip(atomids, distances) if d <= pocket_radius]
    embeddings = [x for (x, d) in zip(embeddings, distances) if d <= pocket_radius]
    b_factors = [x for (x, d) in zip(b_factors, distances) if d <= pocket_radius]
    distances = [1 / x for (x, d) in zip(distances, distances) if d <= pocket_radius]

    xyzs_passed = []
    atomids_passed = []
    embeddings_passed = []
    b_factors_passed = []
    distances_passed = []

    for i, x in enumerate(embeddings):
        try:
            try:
                em = pdb_res_dict[embeddings[i]]
                ai = pdb_atom_dict[atomids[i]]
                embeddings_passed.append(em)
                atomids_passed.append(ai)
                xyzs_passed.append(xyzs[i])
                b_factors_passed.append(b_factors[i])
                distances_passed.append(distances[i])
            except:
                em = pdb_res_dict[embeddings[i][:4]]
                ai = pdb_atom_dict[atomids[i]]
                embeddings_passed.append(em)
                atomids_passed.append(ai)
                xyzs_passed.append(xyzs[i])
                b_factors_passed.append(b_factors[i])
                distances_passed.append(distances_passed[i])
        except:
            pass

    xyzs = xyzs_passed
    atomids = atomids_passed
    embeddings = embeddings_passed
    b_factors = b_factors_passed
    distances = distances_passed

    xyzs = np.array(xyzs)  # 5.29177 = 10 a0

    # Get edges for 3d graph
    distance_matrix = squareform(pdist(xyzs))
    np.fill_diagonal(distance_matrix, float("inf"))  # to remove self-loops
    edge_tmp = np.vstack(np.where(distance_matrix <= radius))  # 5.29177 = 10 a0
    edge1, edge2 = list(edge_tmp[0]), list(edge_tmp[1])

    # Add centroid features into nodes, edges, coords, b_factor and dist
    # source_to_target: messages are passed from source to target.
    centroid_edges = np.arange(len(atomids), len(atomids) + 8, 1)
    for centroid in centroid_edges:
        for i in range(len(atomids)):
            edge1.append(i)  # source
            edge2.append(centroid)  # target

    edge_5 = torch.from_numpy(np.array([edge1, edge2]))

    centroid_nodes = [
        "Cent1",
        "Cent2",
        "Cent3",
        "Cent4",
        "Cent5",
        "Cent6",
        "Cent7",
        "Cent8",
    ]
    centroid_nodes_atom = [pdb_atom_dict[x] for x in centroid_nodes]
    centroid_nodes_resi = [pdb_res_dict[x] for x in centroid_nodes]

    atomids = np.array(atomids + centroid_nodes_atom)
    embeddings = np.array(embeddings + centroid_nodes_resi)

    centroid_bfactor = np.zeros(8)
    b_factors = np.concatenate((b_factors, centroid_bfactor), axis=0)

    centroid_distances = np.repeat(1 / 2, 8)
    distances = np.concatenate((distances, centroid_distances), axis=0)

    centroid_coords = xyzs.mean(axis=0)

    centroid_coords = [centroid_coords for x in range(8)]
    centroid_coords = np.array(centroid_coords)
    xyzs = np.concatenate((xyzs, centroid_coords), axis=0)

    return (
        atomids,
        xyzs,
        edge_5,
        embeddings,
        distances,
        b_factors,
        properties,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdb_file", type=str, default="3g8i_protein")
    parser.add_argument("-mol_file", type=str, default="3g8i_ligand")
    parser.add_argument("-pdb_key", type=str, default="3g8i")
    args = parser.parse_args()

    pdb_file = f"input/{args.pdb_file}.pdb"
    mol_file = f"input/{args.mol_file}.sdf"
    pdb_key = args.pdb_key

    pdb_atom_dict = torch.load("../drugtargetgraph/data/pdb_atom_dict.pt")
    pdb_res_dict = torch.load("../drugtargetgraph/data/pdb_res_dict.pt")

    (
        atomids,
        xyzs,
        edge_5,
        embeddings,
        distances,
        b_factors,
        properties,
    ) = get_info_from_pdb(pdb_file, mol_file, 5, 6, pdb_atom_dict, pdb_res_dict)

    print(f"Number of embedded atoms: {len(embeddings)} / {len(distances)}")

    print(f"Writing input/{pdb_key}.h5")
    with h5py.File(f"input/{pdb_key}.h5", "w") as container:
        if (None not in embeddings) and (None not in atomids):
            # print(atomids.shape, xyzs.shape, edge_5.shape, embeddings.shape, distances.shape, b_factors.shape)

            # Create group in h5 for this id
            container.create_group(str(pdb_key))

            # Add all parameters as datasets to the created group
            container[str(pdb_key)].create_dataset("embeddings", data=embeddings)
            container[str(pdb_key)].create_dataset("b_factors", data=b_factors)
            container[str(pdb_key)].create_dataset("atomids", data=atomids)
            container[str(pdb_key)].create_dataset("xyzs", data=xyzs)
            container[str(pdb_key)].create_dataset("edge_5", data=edge_5)
            container[str(pdb_key)].create_dataset("distances", data=distances)
            container[str(pdb_key)].create_dataset("properties", data=properties)
