FOLD_TO_TRAIN = [4,5]
KFOLD_SPLITS = 8
KFOLD_SEED = 333

import dataclasses
import logging
import os
from pprint import pformat
from time import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import schnetpack
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from schnetpack.nn import Dense, shifted_softplus
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_add, scatter_mean

from csc import const, util, optim
from csc.avg_meter import AverageMeterSet
from csc.clr import CyclicLR
from csc.dnn import l1_loss
from csc.early_stopping import EarlyStopping

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
    lr: float = 1e-4
    weight_decay: float = 1e-4

    clr_max_lr: float = 3e-3
    clr_base_lr: float = 3e-6
    clr_gamma: float = 0.99994

    train_batch: int = 32
    val_batch: int = 256

    n_atom_basis: int = 128
    n_interactions: int = 1

    pairwise_layers: int = 2
    atomwise_layers: int = 2

    atomwise_weight: float = 1.

    pre_trained_path: str = None

    optim: str = 'adam'
    loss_fn: Callable = l1_loss

    epochs: int = 400
    is_save_epoch_fn: Callable = None
    resume_from: Dict[str, int] = None

    types: List[str] = None
    db_path: str = None

    is_one_cv: bool = True

    seed: int = 2

    device: str = device

    exp_name: str = 'schnet-edge_update'
    exp_time: float = time()

    logger_epoch = None

    @staticmethod
    def create_logger(name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            logger.addHandler(logging.FileHandler(filename))
        return logger

    def __post_init__(self):
        if self.resume_from is not None:
            assert os.path.exists(self.out_dir), '{} does not exist.'.format(self.out_dir)

        util.ensure_dir(self.out_dir)

        self.logger_epoch = self.create_logger('epoch_logger_{}'.format(self.exp_time),
                                               '{}/epoch.log'.format(self.out_dir))
        self.type_encoder = util.get_type_encoder()

        with open('{}/conf.txt'.format(self.out_dir), 'w') as f:
            f.write(str(self))

        global device
        device = self.device

    @property
    def out_dir(self):
        return 'data/experiments/{}/{}'.format(self.exp_name, self.exp_time)

    def __str__(self):
        return pformat(dataclasses.asdict(self))


class AtomsData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        atomwise_values = np.concatenate((
            row.sigma_iso,
            row.log_omega,
            row.kappa,
        ), axis=1)
        energy_values = np.concatenate((
            [row.homo],
            [row.lumo],
            [row.U0],
        )).reshape(1, 3).astype(np.float32)

        data = row.to_dict()
        del data['sigma_iso']
        del data['log_omega']
        del data['kappa']
        del data['homo']
        del data['lumo']
        del data['U0']

        return {
            **data,
            'atomwise_values': atomwise_values,
            'energy_values': energy_values,
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


class Energy(nn.Module):
    def __init__(self, n_in=128, n_out=1, n_layers=2, activation=shifted_softplus):
        super(Energy, self).__init__()
        self.out_net = nn.Sequential(
            schnetpack.nn.blocks.MLP(n_in, n_out, n_layers=n_layers, activation=activation),
        )

    def forward(self, inputs, x_atom, x_bond):
        out = self.out_net(x_atom)
        out = scatter_mean(out, inputs['node_graph_indices'], dim=0)

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


def calc_loss_atomwise_detail(y_pred, y_true):
    with torch.no_grad():
        loss = torch.abs(y_pred - y_true)
        loss = loss.mean(dim=0)
        # loss = torch.log(loss)
        loss = loss.detach().cpu().numpy()
    detail = {
        'sigma_iso': loss[0],
        'log_omega': loss[1],
        'kappa': loss[2],
    }
    return detail


def calc_loss_contribution_detail(y_pred, y_true):
    with torch.no_grad():
        loss = torch.abs(y_pred - y_true)
        loss = loss.mean(dim=0)
        # loss = torch.log(loss)
        loss = loss.detach().cpu().numpy()
    detail = {
        'cc': loss[0],
        'fc': loss[1],
        'sd': loss[2],
        'pso': loss[3],
        'dso': loss[4],
    }
    return detail


def calc_loss_energy_detail(y_pred, y_true):
    with torch.no_grad():
        loss = torch.abs(y_pred - y_true)
        loss = loss.mean(dim=0)
        # loss = torch.log(loss)
        loss = loss.detach().cpu().numpy()
    detail = {
        # 'homo': loss[0],
        # 'lumo': loss[1],
        # 'U0': loss[2],
        'U0': loss[0],
    }
    return detail


def calc_loss_type_detail(y_pred, y_true, types):
    abs_errs = torch.abs(y_pred - y_true)
    abs_errs_cc = abs_errs.detach().cpu().numpy()[:, 0]
    types = types.cpu().numpy().astype(int)

    maes = pd.DataFrame({
        'errs': abs_errs_cc,
        'types': types,
    }).groupby('types').agg({
        'errs': [np.mean, 'size']
    })
    maes = maes.reset_index()
    maes.columns = ['type', 'mae', 'n_data']
    # maes['log_mae'] = np.log(maes['log_mae'])

    return maes


def train(loader, model: Net, optimizer, scheduler, conf: Conf):
    meters = AverageMeterSet()
    model.train()

    for i, batch in enumerate(loader):
        scheduler.step()
        meters.update('lr', optimizer.param_groups[0]['lr'])

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)
        coupling_pred = result['coupling']
        atomwise_pred = result['atomwise']
        # energy_pred = result['energy']

        loss_fn = conf.loss_fn

        # ----- Coupling Loss -----
        n_pairs = len(coupling_pred)
        coupling_loss = loss_fn(coupling_pred, batch['coupling_values'])
        meters.update('loss_coupling', coupling_loss.item(), n_pairs)

        # ----- Atomwise Loss -----
        n_atoms = len(atomwise_pred)
        atomwise_loss = loss_fn(atomwise_pred, batch['atomwise_values'])
        meters.update('loss_atomwise', atomwise_loss.item(), n_atoms)

        # ----- Energy Loss -----
        # n_mols = len(energy_pred)
        # # energy_loss = loss_fn(energy_pred, batch['energy_values'])
        # energy_loss = loss_fn(energy_pred, batch['energy_values'][:, [2]])  # Only U0
        # meters.update('loss_energy', energy_loss.item(), n_mols)

        # ----- Metric of coupling contributions -----
        contribution_detail = calc_loss_contribution_detail(coupling_pred, batch['coupling_values'])
        meters.update('loss_cc', contribution_detail['cc'], n_pairs)
        meters.update('loss_fc', contribution_detail['fc'], n_pairs)
        meters.update('loss_sd', contribution_detail['sd'], n_pairs)
        meters.update('loss_pso', contribution_detail['pso'], n_pairs)
        meters.update('loss_dso', contribution_detail['dso'], n_pairs)

        # ----- Metric of atomwise -----
        atomwise_detail = calc_loss_atomwise_detail(atomwise_pred, batch['atomwise_values'])
        meters.update('loss_sigma_iso', atomwise_detail['sigma_iso'], n_atoms)
        meters.update('loss_log_omega', atomwise_detail['log_omega'], n_atoms)
        meters.update('loss_kappa', atomwise_detail['kappa'], n_atoms)

        # ----- Metric of energy -----
        # energy_detail = calc_loss_energy_detail(energy_pred, batch['energy_values'])
        # energy_detail = calc_loss_energy_detail(energy_pred, batch['energy_values'][:, [2]])  # Only U0
        # meters.update('loss_homo', energy_detail['homo'], n_mols)
        # meters.update('loss_lumo', energy_detail['lumo'], n_mols)
        # meters.update('loss_U0', energy_detail['U0'], n_mols)

        # ----- Metric for each types -----
        type_detail = calc_loss_type_detail(coupling_pred, batch['coupling_values'], batch['coupling_types'])
        type_detail['type_name'] = conf.type_encoder.inverse_transform(type_detail.type.values)
        for _, row in type_detail.iterrows():
            meters.update('loss_{}'.format(row.type_name), row.mae, row.n_data)

        # ----- Total Loss -----
        loss = coupling_loss + atomwise_loss * conf.atomwise_weight
        # loss = coupling_loss + atomwise_loss * 0.5 + energy_loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        'lr': meters['lr'].avg,
        'train_loss_cc': np.log(meters['loss_cc'].avg),
        'train_loss_fc': np.log(meters['loss_fc'].avg),
        'train_loss_sd': np.log(meters['loss_sd'].avg),
        'train_loss_pso': np.log(meters['loss_pso'].avg),
        'train_loss_dso': np.log(meters['loss_dso'].avg),
        'train_loss_coupling': meters['loss_coupling'].avg,

        'train_loss_sigma_iso': np.log(meters['loss_sigma_iso'].avg),
        'train_loss_log_omega': np.log(meters['loss_log_omega'].avg),
        'train_loss_kappa': np.log(meters['loss_kappa'].avg),
        'train_loss_atomwise': meters['loss_atomwise'].avg,

        # 'train_loss_homo': np.log(meters['loss_homo'].avg),
        # 'train_loss_lumo': np.log(meters['loss_lumo'].avg),
        # 'train_loss_U0': np.log(meters['loss_U0'].avg),
        # 'train_loss_energy': meters['loss_energy'].avg,

        **{
            'train_loss_{}'.format(t): np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        },
        'train_loss_total': np.mean([
            np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        ]),
    }


def validate(loader, model, conf: Conf):
    meters = AverageMeterSet()
    model.eval()

    for i, batch in enumerate(loader):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        loss_fn = conf.loss_fn

        with torch.no_grad():
            result = model(batch)
            coupling_pred = result['coupling']
            atomwise_pred = result['atomwise']
            # energy_pred = result['energy']

            # ----- Coupling Loss -----
            n_pairs = len(coupling_pred)
            coupling_loss = loss_fn(coupling_pred, batch['coupling_values'])
            meters.update('loss_coupling', coupling_loss.item(), n_pairs)

            # ----- Atomwise Loss -----
            n_atoms = len(atomwise_pred)
            atomwise_loss = loss_fn(atomwise_pred, batch['atomwise_values'])
            meters.update('loss_atomwise', atomwise_loss.item(), n_atoms)

            # ----- Energy Loss -----
            # n_mols = len(energy_pred)
            # energy_loss = loss_fn(energy_pred, batch['energy_values'])
            # energy_loss = loss_fn(energy_pred, batch['energy_values'][:, [2]])  # Only U0
            # meters.update('loss_energy', energy_loss.item(), n_mols)

            # ----- Metric of coupling contributions -----
            contribution_detail = calc_loss_contribution_detail(coupling_pred, batch['coupling_values'])
            meters.update('loss_cc', contribution_detail['cc'], n_pairs)
            meters.update('loss_fc', contribution_detail['fc'], n_pairs)
            meters.update('loss_sd', contribution_detail['sd'], n_pairs)
            meters.update('loss_pso', contribution_detail['pso'], n_pairs)
            meters.update('loss_dso', contribution_detail['dso'], n_pairs)

            # ----- Metric of atomwise -----
            atomwise_detail = calc_loss_atomwise_detail(atomwise_pred, batch['atomwise_values'])
            meters.update('loss_sigma_iso', atomwise_detail['sigma_iso'], n_atoms)
            meters.update('loss_log_omega', atomwise_detail['log_omega'], n_atoms)
            meters.update('loss_kappa', atomwise_detail['kappa'], n_atoms)

            # ----- Metric of energy -----
            # energy_detail = calc_loss_energy_detail(energy_pred, batch['energy_values'])
            # energy_detail = calc_loss_energy_detail(energy_pred, batch['energy_values'][:, [2]])  # Only U0
            # meters.update('loss_homo', energy_detail['homo'], n_mols)
            # meters.update('loss_lumo', energy_detail['lumo'], n_mols)
            # meters.update('loss_U0', energy_detail['U0'], n_mols)

            # ----- Metric for each types -----
            type_detail = calc_loss_type_detail(coupling_pred, batch['coupling_values'], batch['coupling_types'])
            type_detail['type_name'] = conf.type_encoder.inverse_transform(type_detail.type.values)
            for _, row in type_detail.iterrows():
                meters.update('loss_{}'.format(row.type_name), row.mae, row.n_data)

    return {
        'val_loss_cc': np.log(meters['loss_cc'].avg),
        'val_loss_fc': np.log(meters['loss_fc'].avg),
        'val_loss_sd': np.log(meters['loss_sd'].avg),
        'val_loss_pso': np.log(meters['loss_pso'].avg),
        'val_loss_dso': np.log(meters['loss_dso'].avg),
        'val_loss_coupling': meters['loss_coupling'].avg,

        'val_loss_sigma_iso': np.log(meters['loss_sigma_iso'].avg),
        'val_loss_log_omega': np.log(meters['loss_log_omega'].avg),
        'val_loss_kappa': np.log(meters['loss_kappa'].avg),
        'val_loss_atomwise': meters['loss_atomwise'].avg,

        # 'val_loss_homo': np.log(meters['loss_homo'].avg),
        # 'val_loss_lumo': np.log(meters['loss_lumo'].avg),
        # 'val_loss_U0': np.log(meters['loss_U0'].avg),
        # 'val_loss_energy': meters['loss_energy'].avg,

        **{
            'val_loss_{}'.format(t): np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        },
        'val_loss_total': np.mean([
            np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        ]),
    }


def log_hist(df_hist, logger: logging.Logger, types):
    last = df_hist.tail(1)
    best = df_hist.sort_values('val_loss_cc', ascending=True).head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary['name'] = ['Last', 'Best']
    logger.debug(summary[[
                             'name',
                             'epoch',
                             'train_loss_coupling',
                             'train_loss_atomwise',
                             # 'train_loss_energy',
                             'val_loss_coupling',
                             'val_loss_atomwise',
                             # 'val_loss_energy',
                         ] + [
                             'train_loss_{}'.format(t) for t in types
                         ] + [
                             'val_loss_{}'.format(t) for t in types
                         ]])
    logger.debug('')


def write_on_board(df_hist, writer, conf: Conf):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('{}/lr'.format(conf.exp_name), {
        '{}'.format(conf.exp_time): row.lr,
    }, row.epoch)

    for tag in ['cc', 'fc', 'sd', 'pso', 'dso']:
        writer.add_scalars('{}/loss/coupling/{}'.format(conf.exp_name, tag), {
            '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
            '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
        }, row.epoch)
    writer.add_scalars('{}/loss/coupling/total'.format(conf.exp_name), {
        '{}_train'.format(conf.exp_time): row.train_loss_coupling,
        '{}_val'.format(conf.exp_time): row.val_loss_coupling,
    }, row.epoch)

    for tag in ['sigma_iso', 'log_omega', 'kappa']:
        writer.add_scalars('{}/loss/atomwise/{}'.format(conf.exp_name, tag), {
            '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
            '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
        }, row.epoch)
    writer.add_scalars('{}/loss/atomwise/total'.format(conf.exp_name), {
        '{}_train'.format(conf.exp_time): row.train_loss_atomwise,
        '{}_val'.format(conf.exp_time): row.val_loss_atomwise,
    }, row.epoch)

    # for tag in ['homo', 'lumo', 'U0']:
    # for tag in ['U0']:
    #     writer.add_scalars('{}/loss/energy/{}'.format(conf.exp_name, tag), {
    #         '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
    #         '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
    #     }, row.epoch)
    # writer.add_scalars('{}/loss/energy/total'.format(conf.exp_name), {
    #     '{}_train'.format(conf.exp_time): row.train_loss_atomwise,
    #     '{}_val'.format(conf.exp_time): row.val_loss_atomwise,
    # }, row.epoch)

    for tag in conf.types:
        writer.add_scalars('{}/loss/type/{}'.format(conf.exp_name, tag), {
            '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
            '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
        }, row.epoch)
    writer.add_scalars('{}/loss/type/total'.format(conf.exp_name), {
        '{}_train'.format(conf.exp_time): row['train_loss_total'],
        '{}_val'.format(conf.exp_time): row['val_loss_total'],
    }, row.epoch)


def load_pre_trained_dict(pre_trained_path: str, net: nn.Module, prefix: str):
    ckpt = torch.load(pre_trained_path, map_location=device)

    # Use only schnet part
    dst_dict = net.state_dict()
    src_dict = ckpt['model']  # it contains other weights
    keys = src_dict.keys()

    for k in keys:
        if not k.startswith(prefix):
            continue
        dst_key = k[len(prefix) + 1:]
        dst_dict[dst_key] = src_dict[k]

    return dst_dict


def main(conf: Conf):
    print(conf)
    print('less +F {}/epoch.log'.format(conf.out_dir))

    df = pd.read_pickle(conf.db_path)
    folds = KFold(n_splits=KFOLD_SPLITS, random_state=KFOLD_SEED, shuffle=True)

    for cv, (train_idx, val_idx) in enumerate(folds.split(df)):
        if cv not in FOLD_TO_TRAIN: continue
        print('Training fold', cv)
        train_data = AtomsData(df.iloc[train_idx])
        val_data = AtomsData(df.iloc[val_idx])
        print(cv, len(train_data), len(val_data))

        train_loader = DataLoader(train_data,
                                  batch_size=conf.train_batch,
                                  shuffle=True,
                                  num_workers=os.cpu_count() - 1,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_data,
                                batch_size=conf.val_batch,
                                num_workers=os.cpu_count() - 1,
                                collate_fn=collate_fn)

        schnet = SchnetWithEdgeUpdate(
            n_atom_basis=conf.n_atom_basis,
            n_interactions=conf.n_interactions,
            kmax=150 + 13 + 1 + 3 + 2,
        )
        if conf.pre_trained_path is not None:
            state_dict = load_pre_trained_dict(conf.pre_trained_path, schnet, 'schnet')
            schnet.load_state_dict(state_dict)

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

        if conf.optim == 'adam':
            opt = Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        elif conf.optim == 'adamw':
            opt = optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        elif conf.optim == 'radam':
            opt = optim.RAdam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        else:
            raise Exception("Not supported optim {}".format(conf.optim))
        scheduler = CyclicLR(
            opt,
            base_lr=conf.clr_base_lr,
            max_lr=conf.clr_max_lr,
            step_size_up=len(train_loader) * 10,
            mode="exp_range",
            gamma=conf.clr_gamma,
            cycle_momentum=False,
        )
        early_stopping = EarlyStopping(patience=120)

        if conf.resume_from is not None:
            cv_resume = conf.resume_from['cv']
            start_epoch = conf.resume_from['epoch']
            if cv_resume > cv:
                continue
            ckpt = torch.load('{}/{}-{:03d}.ckpt'.format(conf.out_dir, cv, start_epoch))

            model.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

            writer = SummaryWriter(logdir=ckpt['writer_logdir'], purge_step=start_epoch)
            hist = pd.read_csv('{}/{}.csv'.format(conf.out_dir, cv)).to_dict('records')

            print('Loaded checkpoint cv {}, epoch {} from {}'.format(cv, start_epoch, conf.out_dir))
        else:
            hist = []
            writer = SummaryWriter()
            start_epoch = 0

        model = model.to(device)

        for epoch in range(start_epoch, conf.epochs):
            train_result = train(train_loader, model, opt, scheduler, conf)
            val_result = validate(val_loader, model, conf)

            hist.append({
                'epoch': epoch,
                **train_result,
                **val_result,
            })
            df_hist = pd.DataFrame(hist)
            log_hist(df_hist, conf.logger_epoch, conf.types)
            write_on_board(df_hist, writer, conf)

            if epoch % 10 == 9:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            if conf.is_save_epoch_fn is not None and conf.is_save_epoch_fn(epoch):
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'writer_logdir': writer.logdir,
                }, '{}/{}-{:03d}.ckpt'.format(conf.out_dir, cv, epoch + 1))
                df_hist.to_csv('{}/{}.csv'.format(conf.out_dir, cv))
                print('Saved checkpoint {}/{}-{:03d}.ckpt'.format(conf.out_dir, cv, epoch + 1))

            should_stop = early_stopping.step(val_result['val_loss_total'])
            if should_stop:
                print('Early stopping at {}'.format(epoch))
                break

        df_hist = pd.DataFrame(hist)
        best = df_hist.sort_values('val_loss_cc', ascending=True).head(1).iloc[0]
        print(best)

        writer.close()
        if conf.is_one_cv:
            break


def test_model():
    # df = pd.read_pickle('{}/CHAMPS_train_1JHN.pkl'.format(const.ARTIFACTS_DIR))
    df = pd.read_pickle('{}CHAMPS_train_1JHN-np.pkl'.format(const.ARTIFACTS_DIR))
    df = df.head(3)

    data = AtomsData(df)
    loader = DataLoader(data,
                        batch_size=2,
                        shuffle=False,
                        num_workers=os.cpu_count() - 1,
                        collate_fn=collate_fn)
    schnet = SchnetWithEdgeUpdate(kmax=150 + 13 + 1)
    # state_dict = load_pre_trained_dict(
    #     'data/experiments/schnet-edge_update-qm9/1564667098.348393/0-280.ckpt',
    #     schnet, 'schnet')
    # schnet.load_state_dict(state_dict)

    atomwise = Atomwise(n_out=3)
    # state_dict = load_pre_trained_dict(
    #     'data/experiments/schnet-edge_update-qm9/1564667098.348393/0-280.ckpt',
    #     atomwise, 'atomwise')
    # atomwise.load_state_dict(state_dict)
    # atomwise.out_net[0].out_net[1] = Dense(64, 3)
    # print(atomwise)

    model = Net(
        schnet=schnet,
        coupling=ScalarCoupling(),
        atomwise=atomwise,
        energy=Energy(),
    )
    model = model.to(device)

    for i, batch in enumerate(loader):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        # print(batch)
        result = model(batch)
        # print(batch['energy_values'][:, [2]].shape)
        # print(result['energy'].shape)
        break


# %%
# test_model()

# %%
main(Conf(
    is_one_cv=False,

    device='cuda',

    train_batch=20,
    # train_batch=64,
    # train_batch=128,
    val_batch=256,

    lr=1e-4,
    clr_max_lr=2e-3,
    clr_base_lr=3e-6,
    # lr=1e-5,
    # clr_max_lr=3e-4,
    # clr_base_lr=3e-7,
    clr_gamma=0.999991,
    weight_decay=1e-4,
    # weight_decay=3e-3,

    n_atom_basis=128,
    n_interactions=3,
    pairwise_layers=2,
    atomwise_layers=2,

    # atomwise_weight=0.3,

    # loss_fn=log_cosh_loss,
    # loss_fn=smooth_l1_loss(),
    loss_fn=l1_loss,
    optim='adam',

    epochs=800,
    is_save_epoch_fn=lambda x: x % 20 == 19,

    # pre_trained_path='data/experiments/schnet-edge_update-qm9/1564667098.348393/0-600.ckpt',  # atomwise=128

    types=['1JHN'],
    db_path='{}CHAMPS_train_1JHN-np.pkl'.format(const.ARTIFACTS_DIR),

    exp_time=time(),
    # exp_time=1565247280.7239242,
    # resume_from={
    #     'cv': 0,
    #     'epoch': 26,
    # },
))
