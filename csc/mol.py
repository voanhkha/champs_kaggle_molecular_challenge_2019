import traceback
from dataclasses import dataclass
from functools import reduce
from itertools import product
from pprint import pprint
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, Get3DDistanceMatrix, SanitizeMol, SanitizeFlags, rdMolDescriptors, BondType, \
    GetDistanceMatrix, GetShortestPath
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdMolTransforms import GetDihedralRad, GetAngleRad
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdmolfiles import MolFromMolFile, MolFromMol2File

from csc.const import ARTIFACTS_DIR


@dataclass
class NamedNode:
    name: str
    n: int


def read_mol(m_id):
    m = MolFromMolFile('{}/mol/{}.mol'.format(ARTIFACTS_DIR, m_id), removeHs=False)
    if m is not None:
        return m

    m = MolFromMol2File('{}/mol2/{}.mol2'.format(ARTIFACTS_DIR, m_id), removeHs=False)
    if m is not None:
        return m

    m = MolFromMolFile('{}/mol/{}.mol'.format(ARTIFACTS_DIR, m_id), removeHs=False, sanitize=False)
    return m


def determine_surrounding_nodes_1jhx(G: nx.Graph, path) -> List[NamedNode]:
    n0, n1 = path

    nodes = []

    # Neighbouring to n1
    n1_s_nodes = np.setdiff1d(list(G.neighbors(n1)), path)
    if len(n1_s_nodes) == 0:
        pass
    elif len(n1_s_nodes) == 1:
        nodes.append(NamedNode('1f', n1_s_nodes[0]))
    elif len(n1_s_nodes) == 2:
        if G.edges[n1, n1_s_nodes[0]]['bond_type'] == BondType.AROMATIC:
            # Assignment is arbitrary.
            nodes.append(NamedNode('1d', n1_s_nodes[0]))
            nodes.append(NamedNode('1e', n1_s_nodes[1]))
        elif G.edges[n1, n1_s_nodes[0]]['bond_type'] == BondType.DOUBLE:
            nodes.append(NamedNode('1d', n1_s_nodes[0]))
            nodes.append(NamedNode('1e', n1_s_nodes[1]))
        elif G.edges[n1, n1_s_nodes[1]]['bond_type'] == BondType.DOUBLE:
            nodes.append(NamedNode('1d', n1_s_nodes[1]))
            nodes.append(NamedNode('1e', n1_s_nodes[0]))
        else:
            # Only single bonds such as NH3 and CH3-.
            # TODO Should I assign different node names such as 1g and 1h?
            pos0 = G.nodes[n0]['pos']
            pos1 = G.nodes[n1]['pos']
            pos_a = G.nodes[n1_s_nodes[0]]['pos']
            pos_b = G.nodes[n1_s_nodes[1]]['pos']
            dp = (pos0 - pos1).DotProduct(pos_a - pos_b)
            if dp > 0:
                nodes.append(NamedNode('1d', n1_s_nodes[0]))
                nodes.append(NamedNode('1e', n1_s_nodes[1]))
            else:
                nodes.append(NamedNode('1d', n1_s_nodes[1]))
                nodes.append(NamedNode('1e', n1_s_nodes[0]))
    elif len(n1_s_nodes) == 3:
        # Assignment is arbitrary.
        nodes.append(NamedNode('1a', n1_s_nodes[0]))
        nodes.append(NamedNode('1b', n1_s_nodes[1]))
        nodes.append(NamedNode('1c', n1_s_nodes[2]))

    return nodes


def determine_surrounding_nodes_2jhx(G, path) -> List[NamedNode]:
    n0, n1, n2 = path

    nodes = []

    # Neighbouring to n1
    n1_s_nodes = np.setdiff1d(list(G.neighbors(n1)), path)
    if len(n1_s_nodes) == 0:
        pass
    elif len(n1_s_nodes) == 1:
        nodes.append(NamedNode('1c', n1_s_nodes[0]))
    else:
        pos0 = G.nodes[n0]['pos']
        pos2 = G.nodes[n2]['pos']
        pos_a = G.nodes[n1_s_nodes[0]]['pos']
        pos_b = G.nodes[n1_s_nodes[1]]['pos']
        dp = (pos0 - pos2).DotProduct(pos_a - pos_b)

        if dp > 0:
            nodes.append(NamedNode('1a', n1_s_nodes[0]))
            nodes.append(NamedNode('1b', n1_s_nodes[1]))
        else:
            nodes.append(NamedNode('1a', n1_s_nodes[1]))
            nodes.append(NamedNode('1b', n1_s_nodes[0]))

    # Neighbouring to n2
    n2_s_nodes = np.setdiff1d(list(G.neighbors(n2)), path)
    if len(n2_s_nodes) == 0:
        pass
    elif len(n2_s_nodes) == 1:
        nodes.append(NamedNode('2f', n2_s_nodes[0]))
    elif len(n2_s_nodes) == 2:
        pos2 = G.nodes[n2]['pos']
        pos1 = G.nodes[n1]['pos']
        pos_a = G.nodes[n2_s_nodes[0]]['pos']
        pos_b = G.nodes[n2_s_nodes[1]]['pos']
        dp = (pos2 - pos1).DotProduct(pos_a - pos_b)

        if dp > 0:
            nodes.append(NamedNode('2d', n2_s_nodes[0]))
            nodes.append(NamedNode('2e', n2_s_nodes[1]))
        else:
            nodes.append(NamedNode('2d', n2_s_nodes[1]))
            nodes.append(NamedNode('2e', n2_s_nodes[0]))
    else:
        # [Important] Assignment is arbitrary, so they must be aggregated later.
        nodes.append(NamedNode('2a', n2_s_nodes[0]))
        nodes.append(NamedNode('2b', n2_s_nodes[1]))
        nodes.append(NamedNode('2c', n2_s_nodes[2]))

    return nodes


def determine_surrounding_nodes_3jhx(G, path) -> List[NamedNode]:
    n0, n1, n2, n3 = path

    nodes = []

    # Neighbouring to n1
    n1_s_nodes = np.setdiff1d(list(G.neighbors(n1)), path)
    if len(n1_s_nodes) == 0:
        pass
    elif len(n1_s_nodes) == 1:
        nodes.append(NamedNode('1c', n1_s_nodes[0]))
    else:
        pos0 = G.nodes[n0]['pos']
        pos2 = G.nodes[n2]['pos']
        pos_a = G.nodes[n1_s_nodes[0]]['pos']
        pos_b = G.nodes[n1_s_nodes[1]]['pos']
        dp = (pos0 - pos2).DotProduct(pos_a - pos_b)

        if dp > 0:
            nodes.append(NamedNode('1a', n1_s_nodes[0]))
            nodes.append(NamedNode('1b', n1_s_nodes[1]))
        else:
            nodes.append(NamedNode('1a', n1_s_nodes[1]))
            nodes.append(NamedNode('1b', n1_s_nodes[0]))

    # Neighbouring to n2
    n2_s_nodes = np.setdiff1d(list(G.neighbors(n2)), path)
    if len(n2_s_nodes) == 0:
        pass
    elif len(n2_s_nodes) == 1:
        nodes.append(NamedNode('2c', n2_s_nodes[0]))
    else:
        pos3 = G.nodes[n3]['pos']
        pos1 = G.nodes[n1]['pos']
        pos_a = G.nodes[n2_s_nodes[0]]['pos']
        pos_b = G.nodes[n2_s_nodes[1]]['pos']
        dp = (pos3 - pos1).DotProduct(pos_a - pos_b)

        if dp > 0:
            nodes.append(NamedNode('2a', n2_s_nodes[0]))
            nodes.append(NamedNode('2b', n2_s_nodes[1]))
        else:
            nodes.append(NamedNode('2a', n2_s_nodes[1]))
            nodes.append(NamedNode('2b', n2_s_nodes[0]))

    # Neighbouring to n3. It does not exist in the case of xJxH.
    n3_s_nodes = np.setdiff1d(list(G.neighbors(n3)), path)
    if len(n3_s_nodes) == 0:
        pass
    elif len(n3_s_nodes) == 1:
        nodes.append(NamedNode('3f', n3_s_nodes[0]))
    elif len(n3_s_nodes) == 2:
        pos3 = G.nodes[n3]['pos']
        pos2 = G.nodes[n2]['pos']
        pos_a = G.nodes[n3_s_nodes[0]]['pos']
        pos_b = G.nodes[n3_s_nodes[1]]['pos']
        dp = (pos3 - pos2).DotProduct(pos_a - pos_b)
        if dp > 0:
            nodes.append(NamedNode('3d', n3_s_nodes[0]))
            nodes.append(NamedNode('3e', n3_s_nodes[1]))
        else:
            nodes.append(NamedNode('3d', n3_s_nodes[1]))
            nodes.append(NamedNode('3e', n3_s_nodes[0]))
    else:
        # [Important] Assignment is arbitrary, so they must be aggregated later.
        nodes.append(NamedNode('3a', n3_s_nodes[0]))
        nodes.append(NamedNode('3b', n3_s_nodes[1]))
        nodes.append(NamedNode('3c', n3_s_nodes[2]))

    return nodes


# noinspection PyProtectedMember
def mol_to_nx(mol) -> nx.Graph:
    G = nx.Graph()
    conf = mol.GetConformer()

    SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES)

    ComputeGasteigerCharges(mol)
    ring_info = mol.GetRingInfo()
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        # if atom.GetSymbol() == 'N' and atom.GetTotalValence() == 2:
        #     formal_charge = -1
        # elif atom.GetSymbol() == 'N' and atom.GetTotalValence() == 4:
        #     formal_charge = 1
        # elif atom.GetSymbol() == 'O' and atom.GetTotalValence() == 1:
        #     formal_charge = -1
        # else:
        #     formal_charge = atom.GetFormalCharge()
        formal_charge = atom.GetFormalCharge()

        G.add_node(idx,
                   pos=conf.GetAtomPosition(idx),
                   formal_charge=formal_charge,
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   # num_explicit_hs=atom.GetNumExplicitHs(),  # All same
                   is_aromatic=atom.GetIsAromatic(),
                   num_atom_rings=ring_info.NumAtomRings(idx),
                   is_in_ring_size3=atom.IsInRingSize(3),
                   is_in_ring_size4=atom.IsInRingSize(4),
                   is_in_ring_size5=atom.IsInRingSize(5),
                   is_in_ring_size6=atom.IsInRingSize(6),
                   symbol=atom.GetSymbol(),
                   total_valence=atom.GetTotalValence(),
                   gasteiger_charge=atom.GetProp('_GasteigerCharge'),
                   num_implicit_hs=atom.GetNumImplicitHs(),
                   total_degree=atom.GetTotalDegree(),
                   crippen_logp=crippen_contribs[idx][0],
                   crippen_mr=crippen_contribs[idx][1],
                   tpsa=tpsa_contribs[idx],
                   )

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   is_conjugated=bond.GetIsConjugated(),
                   )

    return G


class MyMol:
    def __init__(self, molecule_name):
        self.name = molecule_name
        self.mol = read_mol(molecule_name)
        try:
            self.G = mol_to_nx(self.mol)
        except RuntimeError:
            traceback.print_exc()
            raise 'molecule {} got an error'.format(molecule_name)

    def dihedral(self, n0: int, n1: int, n2: int, n3: int) -> float:
        return GetDihedralRad(self.mol.GetConformer(), int(n0), int(n1), int(n2), int(n3))

    def angle(self, n0: int, n1: int, n2: int) -> float:
        return GetAngleRad(self.mol.GetConformer(), int(n0), int(n1), int(n2))

    def formal_charge(self, node: int):
        return self.G.node[node]['formal_charge']

    def hybridization(self, node: int):
        return self.G.node[node]['hybridization']

    def is_aromatic(self, node: int):
        return self.G.node[node]['is_aromatic']

    def num_atom_rings(self, node: int):
        return self.G.node[node]['num_atom_rings']

    def is_in_ring_size_n(self, node: int, ring_size: int):
        return self.G.node[node]['is_in_ring_size{}'.format(ring_size)]

    def symbol(self, node: int):
        return self.G.node[node]['symbol']

    def gasteiger_charge(self, node: int):
        return self.G.node[node]['gasteiger_charge']

    def total_degree(self, node: int):
        return self.G.node[node]['total_degree']

    def total_valence(self, node: int):
        return self.G.node[node]['total_valence']

    def chiral_tag(self, node: int):
        return self.G.node[node]['chiral_tag']

    def num_implicit_hs(self, node: int):
        return self.G.node[node]['num_implicit_hs']

    def crippen_logp(self, node: int):
        return self.G.node[node]['crippen_logp']

    def crippen_mr(self, node: int):
        return self.G.node[node]['crippen_mr']

    def tpsa(self, node: int):
        return self.G.node[node]['tpsa']

    def is_on_conjugated_edge(self, node: int) -> bool:
        for n in self.G.neighbors(node):
            if self.G.edges[n, node]['is_conjugated']:
                return True
        return False

    def n_atom_on_conjugated_path(self, node: int, atom_symbol: str) -> int:
        target_nodes = self.get_nodes_by_atom(atom_symbol)

        cnt = 0
        for n in target_nodes:
            for path in nx.all_simple_paths(self.G, source=node, target=n):
                is_all_conjugated = np.all([
                    self.G.edges[n0, n1]['is_conjugated']
                    for n0, n1 in zip(path, path[1:])
                ])
                if is_all_conjugated:
                    cnt += 1
                    break

        return cnt

    def bond_type(self, node0: int, node1: int):
        return self.G.edges[node0, node1]['bond_type']

    def find_carboxylic_acid_from(self, node):
        return self.find_ptn_from(node, 'C(=O)O')

    def find_ammonium_from(self, node):
        return self.find_ptn_from(node, '[NH4+]')

    def find_ptn_from(self, node, ptn_smiles):
        ptn = MolFromSmiles(ptn_smiles)
        matches = self.mol.GetSubstructMatches(ptn)

        for sub in matches:
            if node in sub:
                continue
            for n1 in sub:
                if self.G.has_edge(node, n1):
                    return sub

        return None

    def path(self, src: int, dst: int) -> List[int]:
        return nx.shortest_path(self.G, src, dst)

    def get_nodes_by_atom(self, atom_symbol: str):
        assert atom_symbol in ['H', 'C', 'N', 'O', 'F'], 'non supported atom {}'.format(atom_symbol)
        return [x for x, y in self.G.nodes(data=True) if y['symbol'] == atom_symbol]


def node_to_feat(mol: MyMol, n, suf):
    if n is None:
        return {}

    # TODO case of mixing carbocation and carbanion such as dsgdb9nsd_037519
    # total_charge = sum([fc for n, fc in mol.G.nodes.data('formal_charge')])
    # if total_charge != 0 and mol.symbol(n) == 'C' and mol.total_valence(n) == 3:
    #     formal_charge = -total_charge
    #     if formal_charge < 0:
    #         hybridization = HybridizationType.SP3
    #     else:
    #         hybridization = HybridizationType.SP2
    # else:
    #     formal_charge = mol.formal_charge(n)
    #     hybridization = mol.hybridization(n)
    formal_charge = mol.formal_charge(n)
    hybridization = mol.hybridization(n)

    return {
        'formal_charge_{}'.format(suf): formal_charge,
        'chiral_tag_{}'.format(suf): mol.chiral_tag(n),
        'hybridization_{}'.format(suf): hybridization,
        'num_implicit_hs_{}'.format(suf): mol.num_implicit_hs(n),
        'is_aromatic_{}'.format(suf): mol.is_aromatic(n),
        'num_atom_rings_{}'.format(suf): mol.num_atom_rings(n),
        'is_in_ring_size3_{}'.format(suf): mol.is_in_ring_size_n(n, 3),
        'is_in_ring_size4_{}'.format(suf): mol.is_in_ring_size_n(n, 4),
        'is_in_ring_size5_{}'.format(suf): mol.is_in_ring_size_n(n, 5),
        'is_in_ring_size6_{}'.format(suf): mol.is_in_ring_size_n(n, 6),
        'symbol_{}'.format(suf): mol.symbol(n),
        'gasteiger_charge_{}'.format(suf): mol.gasteiger_charge(n),
        'total_degree_{}'.format(suf): mol.total_degree(n),
        'total_valence_{}'.format(suf): mol.total_valence(n),
        'n_neighbors_{}'.format(suf): len(list(mol.G.neighbors(n))),
        'is_on_conjugated_edge_{}'.format(suf): mol.is_on_conjugated_edge(n),
        'n_O_on_conjugated_path_{}'.format(suf): mol.n_atom_on_conjugated_path(n, 'O'),
        'n_N_on_conjugated_path_{}'.format(suf): mol.n_atom_on_conjugated_path(n, 'N'),
        'crippen_logp_{}'.format(suf): mol.crippen_logp(n),
        # 'crippen_mr_{}'.format(suf): mol.crippen_mr(n),
        # 'tpsa_{}'.format(suf): mol.tpsa(n),
    }


def edge_to_feat(mol: MyMol, n0, n1, suf):
    return {
        'bond_type_{}'.format(suf): mol.bond_type(n0, n1),
    }


def calc_all_3d_dist(mol, named_nodes: List[NamedNode]):
    nodes = [n.n for n in named_nodes]
    D = Get3DDistanceMatrix(mol.mol)
    sub_D = D[nodes, :][:, nodes]

    dists = sub_D[np.triu_indices(len(named_nodes), 1)]
    res = {}
    for i, (a, b) in enumerate(zip(*np.triu_indices(len(named_nodes), 1))):
        res['dist_{}_{}'.format(named_nodes[a].name, named_nodes[b].name)] = dists[i]

    return res


def calc_all_angle_feats(mol: MyMol, named_nodes: List[NamedNode]):
    # Build a graph of interest
    G = nx.Graph()
    for i, nn in enumerate(named_nodes):
        if nn.name == '1':
            G.add_edge(0, 1)
        if nn.name == '2':
            G.add_edge(1, 2)
        if nn.name == '3':
            G.add_edge(2, 3)
        if nn.name.startswith('1'):
            G.add_edge(1, i)
        if nn.name.startswith('2'):
            G.add_edge(2, i)
        if nn.name.startswith('3'):
            G.add_edge(3, i)
    # import matplotlib.pyplot as plt
    # nx.draw(G)
    # plt.show()

    length = dict(nx.all_pairs_shortest_path_length(G))
    res = {}

    for a, b in zip(*np.triu_indices(len(named_nodes), 1)):
        node_a, node_b = named_nodes[a], named_nodes[b]
        if length[a][b] == 2:
            path_interest = nx.shortest_path(G, a, b)
            path = [named_nodes[n].n for n in path_interest]
            angle = mol.angle(*path)
            res['angle_{}_{}'.format(node_a.name, node_b.name)] = angle
        if length[a][b] == 3:
            path_interest = nx.shortest_path(G, a, b)
            path = [named_nodes[n].n for n in path_interest]
            dihedral = mol.dihedral(*path)
            res['dihedral_{}_{}'.format(node_a.name, node_b.name)] = dihedral
    return res


def make_dist_feats(dists, suf):
    if len(dists) == 0:
        return {}

    return {
        'dist_mean_{}'.format(suf): np.mean(dists),
        'dist_min_{}'.format(suf): np.min(dists),
        'dist_max_{}'.format(suf): np.max(dists),
        'dist_std_{}'.format(suf): np.std(dists),
    }


def calc_dist_stats_feats(mol: MyMol, named_nodes: List[NamedNode]):
    D = Get3DDistanceMatrix(mol.mol)
    dist_stats_feat = {}

    # for atom_sym in ['H', 'C', 'N', 'O', 'F']:
    for atom_sym in ['H', 'C', 'N', 'O']:
        for nn in named_nodes:
            if len(mol.get_nodes_by_atom(atom_sym)) > 0:
                feats = make_dist_feats(
                    D[nn.n, np.setdiff1d(mol.get_nodes_by_atom(atom_sym), nn.n)],
                    suf='{}_{}'.format(nn.name, atom_sym)
                )
                dist_stats_feat = {**dist_stats_feat, **feats}

    return dist_stats_feat


def calc_usrcat(mol: MyMol, path):
    path_ = np.array(path) + 1
    results = rdMolDescriptors.GetUSRCAT(mol.mol, atomSelections=[path_.tolist()])
    keys = ['usrcat_{}'.format(n) for n in range(len(results))]
    return dict(zip(keys, results))


def calc_node_to_atom_dihedral_stats(mol: MyMol, named_node: NamedNode, atom_sym: str):
    default = {
        'cos_dihedral_cnt_{}_{}'.format(named_node.name, atom_sym): 0,
    }

    atom_nodes = np.array(mol.get_nodes_by_atom(atom_sym))
    if len(atom_nodes) == 0:
        return default

    idx_to_use = np.where(GetDistanceMatrix(mol.mol)[named_node.n, atom_nodes] == 3)
    if len(idx_to_use[0]) == 0:
        return default

    radians = []

    for n in atom_nodes[idx_to_use[0]]:
        path = GetShortestPath(mol.mol, int(named_node.n), int(n))
        dihedral = mol.dihedral(*path)
        radians.append(dihedral)

    return {
        'cos_dihedral_sum_{}_{}'.format(named_node.name, atom_sym): np.cos(radians).sum(),
        'cos2_dihedral_sum_{}_{}'.format(named_node.name, atom_sym): np.cos(np.array(radians) * 2).sum(),
        'cos_dihedral_cnt_{}_{}'.format(named_node.name, atom_sym): len(radians),
    }


def calc_node_to_atom_angle_stats(mol: MyMol, named_node: NamedNode, atom_sym: str):
    atom_nodes = np.array(mol.get_nodes_by_atom(atom_sym))
    default = {
        'cos_angle_cnt_{}_{}'.format(named_node.name, atom_sym): 0,
    }
    if len(atom_nodes) == 0:
        return default

    idx_to_use = np.where(GetDistanceMatrix(mol.mol)[named_node.n, atom_nodes] == 2)
    if len(idx_to_use[0]) == 0:
        return default

    radians = []

    for n in atom_nodes[idx_to_use[0]]:
        path = GetShortestPath(mol.mol, int(named_node.n), int(n))
        angle = mol.angle(*path)
        radians.append(angle)

    return {
        'cos_angle_mean_{}_{}'.format(named_node.name, atom_sym): np.mean(np.cos(radians)),
        'cos_angle_std_{}_{}'.format(named_node.name, atom_sym): np.std(np.cos(radians)),
        'cos_angle_max_{}_{}'.format(named_node.name, atom_sym): np.max(np.cos(radians)),
        'cos_angle_min_{}_{}'.format(named_node.name, atom_sym): np.min(np.cos(radians)),
        'cos_angle_cnt_{}_{}'.format(named_node.name, atom_sym): len(radians),
    }


def load_mulliken_charge():
    return pd.read_csv('{}/mulliken_charge_all.csv'.format(ARTIFACTS_DIR), index_col=['molecule_name', 'atom_index'])


def mol_to_3jxx_feats(mol: MyMol, atom0: int, atom1: int):
    path = mol.path(atom0, atom1)
    assert path[0] == atom0, 'wrong path'
    n0, n1, n2, n3 = path

    s_nodes = determine_surrounding_nodes_3jhx(mol.G, path)
    nodes_in_interest = [NamedNode(name=str(i), n=n) for i, n in enumerate(path)] + s_nodes

    node_feats = reduce(lambda x, y: {**x, **y}, [
        node_to_feat(mol, n.n, n.name)
        for n in nodes_in_interest
    ])

    e12_feats = edge_to_feat(mol, n1, n2, '12')

    all_3d_dist_feats = calc_all_3d_dist(mol, nodes_in_interest)
    all_angle_feats = calc_all_angle_feats(mol, nodes_in_interest)
    dist_stats_feats = calc_dist_stats_feats(mol, nodes_in_interest[:4])

    dihedral_stats_feats = reduce(lambda a, b: {**a, **b}, [
        calc_node_to_atom_dihedral_stats(mol, nn, atom_sym)
        for nn, atom_sym in product(nodes_in_interest[:4], ['H', 'C', 'N', 'O'])
    ])

    angle_stats_feats = reduce(lambda a, b: {**a, **b}, [
        calc_node_to_atom_angle_stats(mol, nn, atom_sym)
        for nn, atom_sym in product(nodes_in_interest[:4], ['H', 'C', 'N', 'O'])
    ])

    return {
        **node_feats,
        **e12_feats,
        **all_3d_dist_feats,
        **all_angle_feats,
        **dist_stats_feats,
        **dihedral_stats_feats,
        **angle_stats_feats,
        # Non features
        **{
            'molecule_name': mol.name,
            'n0': n0,
            'n1': n1,
            'n2': n2,
            'n3': n3,
        }
    }


def mol_to_2jxx_feats(mol: MyMol, atom0: int, atom1: int):
    path = mol.path(atom0, atom1)
    assert path[0] == atom0, 'wrong path'
    n0, n1, n2 = path

    s_nodes = determine_surrounding_nodes_2jhx(mol.G, path)
    nodes_in_interest = [NamedNode(name=str(i), n=n) for i, n in enumerate(path)] + s_nodes

    node_feats = reduce(lambda x, y: {**x, **y}, [
        node_to_feat(mol, n.n, n.name)
        for n in nodes_in_interest
    ])

    e12_feats = edge_to_feat(mol, n1, n2, '12')

    all_3d_dist_feats = calc_all_3d_dist(mol, nodes_in_interest)
    all_angle_feats = calc_all_angle_feats(mol, nodes_in_interest)
    dist_stats_feats = calc_dist_stats_feats(mol, nodes_in_interest)

    dihedral_stats_feats = reduce(lambda a, b: {**a, **b}, [
        calc_node_to_atom_dihedral_stats(mol, nn, atom_sym)
        for nn, atom_sym in product(nodes_in_interest[:4], ['H', 'C', 'N', 'O'])
    ])

    angle_stats_feats = reduce(lambda a, b: {**a, **b}, [
        calc_node_to_atom_angle_stats(mol, nn, atom_sym)
        for nn, atom_sym in product(nodes_in_interest[:4], ['H', 'C', 'N', 'O'])
    ])

    return {
        **node_feats,
        **e12_feats,
        **all_3d_dist_feats,
        **all_angle_feats,
        **dist_stats_feats,
        **dihedral_stats_feats,
        **angle_stats_feats,
        **{
            'molecule_name': mol.name,
            'n0': n0,
            'n1': n1,
            'n2': n2,
        },
    }


def mol_to_1jxx_feats(mol: MyMol, atom0: int, atom1: int):
    path = mol.path(atom0, atom1)
    assert path[0] == atom0, 'wrong path'

    s_nodes = determine_surrounding_nodes_1jhx(mol.G, path)
    nodes_in_interest = [NamedNode(name=str(i), n=n) for i, n in enumerate(path)] + s_nodes

    node_feats = reduce(lambda x, y: {**x, **y}, [
        node_to_feat(mol, n.n, n.name)
        for n in nodes_in_interest
    ])

    all_3d_dist_feats = calc_all_3d_dist(mol, nodes_in_interest)
    all_angle_feats = calc_all_angle_feats(mol, nodes_in_interest)
    dist_stats_feats = calc_dist_stats_feats(mol, nodes_in_interest)

    dihedral_stats_feats = reduce(lambda a, b: {**a, **b}, [
        calc_node_to_atom_dihedral_stats(mol, nn, atom_sym)
        for nn, atom_sym in product(nodes_in_interest, ['H', 'C', 'N', 'O'])
    ])

    angle_stats_feats = reduce(lambda a, b: {**a, **b}, [
        calc_node_to_atom_angle_stats(mol, nn, atom_sym)
        for nn, atom_sym in product(nodes_in_interest, ['H', 'C', 'N', 'O'])
    ])

    fp = GetMorganFingerprintAsBitVect(mol.mol, 2, fromAtoms=path)

    return {
        **node_feats,
        **all_3d_dist_feats,
        **all_angle_feats,
        **dist_stats_feats,
        **dihedral_stats_feats,
        **angle_stats_feats,
        **{
            'fp': fp,
        },
        # Not feature
        **{
            'molecule_name': mol.name,
        },
        **{
            'n{}'.format(nn.name): nn.n
            for nn in nodes_in_interest
        }
    }


if __name__ == '__main__':
    # my_mol = MyMol('dsgdb9nsd_000214')  # Benzine
    # my_mol = MyMol('dsgdb9nsd_000271')
    # my_mol = MyMol('dsgdb9nsd_045541')
    # my_mol = MyMol('dsgdb9nsd_000004')
    my_mol = MyMol('dsgdb9nsd_055944')
    # D = my_mol.get_3d_dist_mat()[[1, 3, 4, 5], :][:, [1, 3, 4, 5]]
    # print(D)
    # print(np.triu_indices(4, 1))
    # for a, b in zip(*np.triu_indices(4, 1)):
    #     print(a, b)
    # print(D[np.triu_indices(4, 1)])
    # print(mol_to_3jhh_feats(my_mol, 6, 7))
    # pprint(mol_to_3jxx_feats(my_mol, 5, 12))
    # print(len(mol_to_3jxx_feats(my_mol, 5, 12)))
    pprint(mol_to_1jxx_feats(my_mol, 2, 1))

    # pass

# %%
# my_mol = MyMol('dsgdb9nsd_045298')
# my_mol = MyMol('dsgdb9nsd_000271')
# my_mol = MyMol('dsgdb9nsd_132702')  # Sanitize error for Explicit valence for atom # 1 C, 5, is greater than permitted
# my_mol = MyMol('dsgdb9nsd_037519')
# my_mol = MyMol('dsgdb9nsd_100504')
# my_mol = MyMol('dsgdb9nsd_059592')
# my_mol = MyMol('dsgdb9nsd_093054')
# my_mol = MyMol('dsgdb9nsd_000028')
# my_mol = MyMol('dsgdb9nsd_054796')
# my_mol = MyMol('dsgdb9nsd_129053')
# my_mol = MyMol('dsgdb9nsd_108155')
# mol = my_mol.mol
#
# print(nx.cycle_basis(my_mol.G))

# my_mol.edge_to_feat(1, 7, '')
# node_to_feat(my_mol, 5, '')

# %%
# mol = read_mol('dsgdb9nsd_000785')
# mol = read_mol('dsgdb9nsd_066602')
# # mol = read_mol('dsgdb9nsd_132702')
# res = SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES)
# # contribs = rdMolDescriptors._CalcCrippenContribs(mol)
# # contribs = rdMolDescriptors._CalcLabuteASAContribs(mol)
# # contribs = rdMolDescriptors._CalcTPSAContribs(mol)
# # contribs = rdMolDescriptors.CalcAUTOCORR2D(mol)
# # contribs = rdMolDescriptors.GetUSR(mol)
# contribs = rdMolDescriptors.GetUSRCAT(mol, atomSelections=[[1, 2, 3, 4]])
# print(contribs)

# %%
# mol = read_mol('dsgdb9nsd_129053')
# smiles = MolToSmiles(mol)
#
# mol2 = MolFromSmiles(smiles)
# for atom in mol2.GetAtoms():
#     idx = atom.GetIdx()
#
#     print(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalValence())
# my_mol = MyMol('dsgdb9nsd_055944')
# h_nodes = np.array(my_mol.get_nodes_by_atom('H'))
#
# idx_to_use = np.where(GetDistanceMatrix(my_mol.mol)[16, h_nodes] == 3)
# sum_cos_dihedral = 0
# sum_cos2_dihedral = 0
#
# for n in h_nodes[idx_to_use[0]]:
#     path = GetShortestPath(my_mol.mol, 16, int(n))
#     dihedral = my_mol.dihedral(*path)
#     sum_cos_dihedral += np.cos(dihedral)
#     sum_cos2_dihedral += np.cos(dihedral * 2)
