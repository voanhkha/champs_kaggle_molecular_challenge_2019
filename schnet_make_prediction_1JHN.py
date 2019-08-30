TYPE = '1JHN'
FILEDIR = '1jhn_all/'
OUTPUT = '1JHN_seed333_8fold.joblib'
KFOLD_SPLIT = 8
KFOLD_SEED = 333

N_ATOM_BASIS = 256
N_INTERACTIONS = 3
PAIRWISE_LAYERS = 2
ATOMWISE_LAYERS = 2


import dataclasses
from glob import glob
from multiprocessing import cpu_count
from time import time
from typing import List

import joblib
import pandas as pd
import schnetpack
import schnetpack.nn.blocks
import torch
import torch.nn as nn
from schnetpack.data import AtomsLoader
from schnetpack.datasets import *
from schnetpack.nn import shifted_softplus, Dense
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_scatter import scatter_add

from csc import util, const

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.width = 999


# %%
@dataclasses.dataclass
class Conf:
    val_batch: int = 256 + 128

    n_atom_basis: int = 128
    n_interactions: int = 1
    pairwise_layers: int = 2
    atomwise_layers: int = 2

    types: List[str] = None
    train_db_path: str = None
    test_db_path: str = None

    seed: int = 3

    device: str = device

    # exp_name: str = 'schnet-edge_update'
    # exp_time: float = time()

    logger_epoch = None

    @staticmethod
    def create_logger(name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            logger.addHandler(logging.FileHandler(filename))
        return logger

    def __post_init__(self):
        self.type_encoder = util.get_type_encoder()

        global device
        device = self.device

    @property
    def out_dir(self):
        return FILEDIR
        #return 'data/weights/model2-np/{}'.format(self.types[0])


class AtomsData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = row.to_dict()
        return {
            **data,
        }


def _compute_stacked_offsets(sizes, repeats):
    """ Computes offsets to add to indices of stacked np arrays.
    When a set of np arrays are stacked, the indices of those from the second on
    must be offset in order to be able to index into the stacked np array. This
    computes those offsets.
    Args:
        sizes: A 1D sequence of np arrays of the sizes per graph.
        repeats: A 1D sequence of np arrays of the number of repeats per graph.
    Returns:
        The index offset per graph.
    """
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def _concat(to_stack):
    """ function to stack (or concatentate) depending on dimensions """
    if np.asarray(to_stack[0]).ndim >= 2:
        return np.concatenate(to_stack)

    else:
        return np.hstack(to_stack)


def rbf_expansion(distances, mu=0, delta=0.1, kmax=150):
    k = np.arange(0, kmax)
    logits = -(np.atleast_2d(distances).T - (-mu + delta * k)) ** 2 / delta
    return np.exp(logits)


def make_target_mask(batch_data):
    n_total_atom = sum(batch_data['n_atom'])
    target_pairs = batch_data['target_pairs']
    connectivity = batch_data['connectivity']

    mat0 = np.zeros((n_total_atom, n_total_atom))
    mat0[target_pairs[:, 0], target_pairs[:, 1]] = 1
    mask0 = mat0[connectivity[:, 0], connectivity[:, 1]].astype(bool)

    mat1 = np.zeros((n_total_atom, n_total_atom))
    mat1[target_pairs[:, 1], target_pairs[:, 0]] = 1
    mask1 = mat1[connectivity[:, 0], connectivity[:, 1]].astype(bool)

    return mask0, mask1


def to_undirected(graph_index, graph_attr):
    graph_index = np.concatenate((graph_index, np.fliplr(graph_index)))
    graph_attr = np.concatenate((graph_attr, graph_attr))
    return graph_index, graph_attr


def convert_edge_feat(edge_index, edge_attr, n_atoms, n_feat):
    mat = np.zeros((n_atoms, n_atoms, n_feat))
    mat[
        edge_index[:, 0],
        edge_index[:, 1]
    ] = edge_attr
    return mat


def collate_fn(examples):
    keys = list(examples[0].keys())
    batch_data = {
        k: _concat([examples[n][k] for n in range(len(examples))])
        for k in keys
    }

    offset = _compute_stacked_offsets(batch_data['n_atom'], batch_data['n_bond'])
    batch_data['connectivity'] += offset[:, np.newaxis]

    offset = _compute_stacked_offsets(batch_data['n_atom'], batch_data['n_target_pairs'])
    batch_data['target_pairs'] += offset[:, np.newaxis]
    batch_data['target_mask0'], batch_data['target_mask1'] = make_target_mask(batch_data)

    offset = _compute_stacked_offsets(batch_data['n_atom'], batch_data['n_dihedral_edges'])
    batch_data['dihedral_edge_index'] += offset[:, np.newaxis]
    # print(batch_data['dihedral_edge_attr'].shape)
    batch_data['dihedral_edge_attr'][:, 2] = 1  # remove cos3 and one hot at the same time
    batch_data['dihedral_edge_index'], batch_data['dihedral_edge_attr'] = to_undirected(
        batch_data['dihedral_edge_index'], batch_data['dihedral_edge_attr']
    )

    offset = _compute_stacked_offsets(batch_data['n_atom'], batch_data['n_angle_edges'])
    batch_data['angle_edge_index'] += offset[:, np.newaxis]
    batch_data['angle_edge_index'][:, 1] = 1  # remove cos(rad) and one hot at the same time
    # batch_data['angle_edge_index'][:, 0] = 1  # remove rad and one hot at the same time
    # batch_data['angle_edge_attr'] = np.pad(batch_data['angle_edge_attr'], [(0, 0), (0, 1)], 'constant',
    #                                        constant_values=1)
    batch_data['angle_edge_index'], batch_data['angle_edge_attr'] = to_undirected(
        batch_data['angle_edge_index'], batch_data['angle_edge_attr']
    )

    offset = _compute_stacked_offsets(batch_data['n_atom'], batch_data['n_coupling_edges'])
    batch_data['coupling_edge_index'] += offset[:, np.newaxis]
    batch_data['coupling_edge_index'], batch_data['coupling_edge_attr'] = to_undirected(
        batch_data['coupling_edge_index'], batch_data['coupling_edge_attr']
    )

    offset = _compute_stacked_offsets(batch_data['n_atom'], batch_data['n_bond_edges'])
    batch_data['bond_edge_index'] += offset[:, np.newaxis]
    batch_data['bond_edge_index'], batch_data['bond_edge_attr'] = to_undirected(
        batch_data['bond_edge_index'], batch_data['bond_edge_attr']
    )
    # print(batch_data['connectivity'])
    # print(batch_data['bond_edge_index'])

    n_graphs = len(examples)
    batch_data['node_graph_indices'] = np.repeat(np.arange(n_graphs), batch_data['n_atom'])
    batch_data['bond_graph_indices'] = np.repeat(np.arange(n_graphs), batch_data['n_bond'])

    batch_data['distance_rbf'] = rbf_expansion(batch_data['distance']).astype(np.float32)

    # Merge coupling feats into distance_rbf
    n_total_atom = sum(batch_data['n_atom'])
    coupling_feat = convert_edge_feat(
        batch_data['coupling_edge_index'],
        batch_data['coupling_edge_attr'],
        n_atoms=n_total_atom,
        n_feat=13,
    )[
        batch_data['connectivity'][:, 0],
        batch_data['connectivity'][:, 1]
    ]
    bond_feat = convert_edge_feat(
        batch_data['bond_edge_index'],
        batch_data['bond_edge_attr'],
        n_atoms=n_total_atom,
        n_feat=1,
    )[
        batch_data['connectivity'][:, 0],
        batch_data['connectivity'][:, 1]
    ]
    dihedral_feat = convert_edge_feat(
        batch_data['dihedral_edge_index'],
        batch_data['dihedral_edge_attr'],
        n_atoms=n_total_atom,
        n_feat=3,
    )[
        batch_data['connectivity'][:, 0],
        batch_data['connectivity'][:, 1]
    ]
    angle_feat = convert_edge_feat(
        batch_data['angle_edge_index'],
        batch_data['angle_edge_attr'],
        n_atoms=n_total_atom,
        n_feat=2,
    )[
        batch_data['connectivity'][:, 0],
        batch_data['connectivity'][:, 1]
    ]
    batch_data['distance_rbf'] = np.concatenate((
        batch_data['distance_rbf'],
        coupling_feat,
        bond_feat,
        dihedral_feat,
        angle_feat,
    ), axis=1).astype(np.float32)
    # print(tmp.shape, batch_data['distance_rbf'].shape)

    del batch_data['n_atom']
    del batch_data['n_bond']
    del batch_data['n_target_pairs']
    del batch_data['n_dihedral_edges']
    del batch_data['n_angle_edges']
    del batch_data['n_coupling_edges']
    del batch_data['n_bond_edges']
    del batch_data['distance']
    del batch_data['dihedral_edge_index']
    del batch_data['dihedral_edge_attr']
    del batch_data['coupling_edge_index']
    del batch_data['coupling_edge_attr']
    del batch_data['bond_edge_index']
    del batch_data['bond_edge_attr']

    batch_data = {
        k: torch.from_numpy(v)
        for k, v in batch_data.items()
    }
    return batch_data


class SchnetWithEdgeUpdate(nn.Module):
    def __init__(self, n_atom_basis=128, max_z=100, kmax=150, n_interactions=1, activation=shifted_softplus):
        super(SchnetWithEdgeUpdate, self).__init__()
        self.n_interactions = n_interactions
        self.embedding = nn.Embedding(max_z, n_atom_basis - 1)
        self.edge_update_net = nn.Sequential(
            Dense(3 * n_atom_basis, 2 * n_atom_basis, activation=activation),
            Dense(2 * n_atom_basis, n_atom_basis),
        )
        self.msg_edge_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=activation),
        )
        self.msg_atom_fc = Dense(n_atom_basis, n_atom_basis)
        self.state_trans_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis),
        )

        # self.n_dihedral_edge_attrs = 2
        # n_dihedral_edge_feats = 16
        # self.dihedral_net = gnn.NNConv(n_atom_basis, n_dihedral_edge_feats, nn.Sequential(
        #     Dense(self.n_dihedral_edge_attrs, n_atom_basis, activation=F.relu),
        #     Dense(n_atom_basis, n_atom_basis * n_dihedral_edge_feats),
        # ))

        # n_angle_edge_attrs = 1
        # n_angle_edge_feats = 8
        # self.angle_net = gnn.NNConv(n_atom_basis, n_angle_edge_feats, nn.Sequential(
        #     Dense(n_angle_edge_attrs, n_atom_basis, activation=F.relu),
        #     Dense(n_atom_basis, n_atom_basis * n_angle_edge_feats),
        # ))

        # self.init_atom_fc = Dense(n_atom_basis + n_dihedral_edge_feats, n_atom_basis, activation=activation)
        self.init_edge_fc = Dense(kmax, n_atom_basis, activation=activation)

    # noinspection PyCallingNonCallable
    def forward(self, inputs):
        x_atom = self.embedding(inputs['atom'])
        x_atom = torch.cat((
            x_atom,
            inputs['mulliken_charges'],
            # inputs['gasteiger_charges'],
        ), dim=1)

        # assert inputs['dihedral_edge_attr'].shape[1] == 3, 'n dihedral feats is not 3.'
        # dihedral_feat = self.dihedral_net(x_atom,
        #                                   inputs['dihedral_edge_index'].t(),
        #                                   inputs['dihedral_edge_attr'][:, :self.n_dihedral_edge_attrs])
        # dihedral_feat = F.relu(dihedral_feat)

        # angle_feat = self.angle_net(x_atom, inputs['angle_edge_index'].t(), inputs['angle_edge_attr'])
        # angle_feat = F.relu(angle_feat)

        # x_atom = torch.cat((x_atom, dihedral_feat), dim=1)
        # x_atom = self.init_atom_fc(x_atom)

        x_bond = inputs['distance_rbf']
        x_bond = self.init_edge_fc(x_bond)

        src_idx = inputs['connectivity'][:, 0]
        dst_idx = inputs['connectivity'][:, 1]

        for n in range(self.n_interactions):
            # Update edge
            x_src_atom = x_atom[src_idx]
            x_dst_atom = x_atom[dst_idx]
            x_bond = torch.cat((x_src_atom, x_dst_atom, x_bond), dim=1)
            x_bond = self.edge_update_net(x_bond)

            # message function
            bond_msg = self.msg_edge_net(x_bond)
            src_atom_msg = self.msg_atom_fc(x_src_atom)
            messages = torch.mul(bond_msg, src_atom_msg)
            messages = scatter_add(messages, dst_idx, dim=0)

            # state transition function
            messages = self.state_trans_net(messages)
            x_atom = x_atom + messages
            # print('x_atom', n, torch.isnan(x_atom).sum())

        return x_atom, x_bond


class ScalarCoupling(nn.Module):
    def __init__(self, n_atom_in=128, n_bond_in=128, n_layers=2, activation=shifted_softplus):
        super(ScalarCoupling, self).__init__()
        self.out_net = nn.Sequential(
            schnetpack.nn.blocks.MLP(n_atom_in * 2 + n_bond_in, 4, n_layers=n_layers, activation=activation),
        )
        self.contributions_to_scalar = nn.Linear(4, 1)
        self.contributions_to_scalar.weight.data = torch.Tensor([[1., 1., 1., 1.]])
        self.contributions_to_scalar.bias.data = torch.Tensor([0.])

    def forward(self, inputs, x_atom, x_bond):
        idx0 = inputs['target_pairs'][:, 0]
        idx1 = inputs['target_pairs'][:, 1]

        x_atom_idx0 = x_atom[idx0]
        x_atom_idx1 = x_atom[idx1]
        x_target_bond0 = x_bond[inputs['target_mask0']]
        # x_target_bond1 = x_bond[inputs['target_mask1']]
        x_bond = torch.cat((x_atom_idx0, x_atom_idx1, x_target_bond0), dim=1)

        contributions = self.out_net(x_bond)
        scc = self.contributions_to_scalar(contributions)
        out = torch.cat((scc, contributions), dim=1)

        return out


class Atomwise(nn.Module):
    def __init__(self, n_in=128, n_out=1, n_layers=2, activation=shifted_softplus):
        super(Atomwise, self).__init__()
        # n_bond_in = 128
        # n_bond_out = 32
        # self.bond_to_atom = gnn.NNConv(n_in, n_bond_out, nn.Sequential(
        #     Dense(n_bond_in, n_in, activation=F.relu),
        #     Dense(n_in, n_in * n_bond_out),
        # ))
        self.out_net = nn.Sequential(
            schnetpack.nn.blocks.MLP(n_in, n_out, n_layers=n_layers, activation=activation),
        )

    # noinspection PyCallingNonCallable
    def forward(self, inputs, x_atom):
        # x_bond_to_atom = self.bond_to_atom(x_atom, inputs['connectivity'].t(), x_bond)
        # x_bond_to_atom = F.relu(x_bond_to_atom)
        # x_atom = torch.cat((x_atom, x_bond_to_atom), dim=1)
        out = self.out_net(x_atom)
        return out


class Net(nn.Module):
    def __init__(self, schnet, coupling, atomwise, energy=None):
        super(Net, self).__init__()
        self.schnet = schnet
        self.coupling = coupling
        self.atomwise = atomwise
        # self.energy = energy

    def forward(self, inputs):
        x_atom, x_bond = self.schnet(inputs)
        out_coupling = self.coupling(inputs, x_atom, x_bond)
        out_atomwise = self.atomwise(inputs, x_atom)
        # out_energy = self.energy(inputs, x_atom, x_bond)
        return {
            'coupling': out_coupling,
            'atomwise': out_atomwise,
            # 'energy': out_energy,
        }


def make_coupling_result(batch, result, is_train=True):
    if is_train:
        scc = batch['scalar_coupling_constants'].view(-1, 7)  # idx0, idx1, cc, fc, sd, pso, dso
    else:
        scc = batch['scalar_coupling_constants'].view(-1, 2)  # idx0, idx1

    mask = scc[:, 0] != scc[:, 1]

    y_pred = result['y'].view(-1, 5)
    y_pred = y_pred[mask, 0]

    if is_train:
        y_true = scc[mask, 2]  # cc
        return y_pred, y_true
    else:
        return y_pred


def pred(conf: Conf):
    train_df = pd.read_pickle(conf.train_db_path)
    test_df = pd.read_pickle(conf.test_db_path)
    test_loader = DataLoader(AtomsData(test_df),
                             batch_size=conf.val_batch,
                             num_workers=os.cpu_count() - 1,
                             collate_fn=collate_fn)

    folds = KFold(n_splits=KFOLD_SPLIT, random_state=KFOLD_SEED, shuffle=True)

    y_pred_all = []
    y_true_all = []
    y_pred_test_all = []

    for cv, (_, val_idx) in enumerate(folds.split(train_df)):
        val_loader = AtomsLoader(AtomsData(train_df.iloc[val_idx]),
                                 batch_size=conf.val_batch,
                                 num_workers=cpu_count() - 1,
                                 collate_fn=collate_fn)
        y_pred_cv = []
        y_true_cv = []
        y_pred_test_cv = []

        schnet = SchnetWithEdgeUpdate(
            n_atom_basis=conf.n_atom_basis,
            n_interactions=conf.n_interactions,
            kmax=150 + 13 + 1 + 3 + 2,
        )
        atomwise = Atomwise(
            n_in=conf.n_atom_basis,
            n_layers=conf.atomwise_layers,
            n_out=3,
        )
        model = Net(
            schnet=schnet,
            coupling=ScalarCoupling(
                n_atom_in=conf.n_atom_basis,
                n_bond_in=conf.n_atom_basis,
                n_layers=conf.pairwise_layers,
            ),
            atomwise=atomwise,
        )

        snapshot_paths = sorted(glob('{}/{}-*.ckpt'.format(conf.out_dir, cv)))

        for s in snapshot_paths:
            print(cv, s)
            y_pred_snapshot = np.array([]).reshape(-1, 1)  # do not save contributions
            y_true_snapshot = np.array([]).reshape(-1, 1)
            y_pred_test_snapshot = np.array([]).reshape(-1, 1)

            ckpt = torch.load(s, map_location=device)
            model.load_state_dict(ckpt['model'])
            model.eval()
            model.to(conf.device)

            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    output = model(batch)
                    y_pred = output['coupling'][:, 0]
                    y_true = batch['coupling_values'][:, 0]
                    y_pred_snapshot = np.concatenate((y_pred_snapshot, y_pred.cpu().numpy().reshape(-1, 1)))
                    y_true_snapshot = np.concatenate((y_true_snapshot, y_true.cpu().numpy().reshape(-1, 1)))

            for batch in test_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    output = model(batch)
                    y_pred = output['coupling'][:, 0]
                    y_pred_test_snapshot = np.concatenate((y_pred_test_snapshot, y_pred.cpu().numpy().reshape(-1, 1)))

            y_pred_cv.append(y_pred_snapshot)
            y_true_cv.append(y_true_snapshot)
            y_pred_test_cv.append(y_pred_test_snapshot)

        y_pred_all.append(np.array(y_pred_cv))
        y_true_all.append(np.array(y_true_cv))
        y_pred_test_all.append(np.array(y_pred_test_cv))

    return y_pred_all, y_true_all, y_pred_test_all


# %%


# (n_cv, n_snapshots, pred)
y_pred_all, y_true_all, y_pred_test_all = pred(Conf(
    device='cuda',

    n_atom_basis=N_ATOM_BASIS,
    n_interactions=N_INTERACTIONS,
    pairwise_layers=PAIRWISE_LAYERS,
    atomwise_layers=ATOMWISE_LAYERS,

    types=[TYPE],
    train_db_path='CHAMPS_train_{}-np.pkl'.format(TYPE),
    test_db_path='CHAMPS_test_{}-np.pkl'.format(TYPE),
))

# %%
joblib.dump({
    'y_pred_all': y_pred_all,
    'y_true_all': y_true_all,
    'y_pred_test_all': y_pred_test_all,
}, OUTPUT)