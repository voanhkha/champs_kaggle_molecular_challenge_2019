import os

import feather
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from csc.util import get_type_onehot_encoder


def get_main_train_df(coupling_type=None):
    df = pd.read_csv('data/input/train.csv')
    if coupling_type is None:
        return df
    if isinstance(coupling_type, str):
        coupling_type = [coupling_type]

    return df[df.type.isin(coupling_type)].reset_index(drop=True)


def get_main_test_df(coupling_type=None):
    df = pd.read_csv('data/input/test.csv')
    if coupling_type is None:
        return df
    if isinstance(coupling_type, str):
        coupling_type = [coupling_type]

    return df[df.type.isin(coupling_type)].reset_index(drop=True)


class CouplingProvider:
    DF_CACHE_PATH = 'tmp/df_CouplingProvider.fth'

    def __init__(self, scaler='std'):
        self.cols = [
            'scalar_coupling_constant',
            'fc',
            'sd',
            'pso',
            'dso',
        ]
        self.scaler = scaler
        df = self.init_df()
        self.grouped = df.groupby('molecule_name')
        self.type_encoder = get_type_onehot_encoder()

    def get_coupling_values(self, mol_name, types):
        g = self.grouped.get_group(mol_name)
        g = g[g.type.isin(types)]

        # Case for H2O
        if len(g.type.values) == 0:
            return np.array([]).reshape(-1, 2), np.array([]).reshape(-1, 13)

        encoded_types = self.type_encoder.transform(g.type.values.reshape(-1, 1))
        feats = np.concatenate((
            g[self.cols].values,
            encoded_types,
        ), axis=1)
        edge_index = g[['atom_index_0', 'atom_index_1']].values

        return edge_index, feats

    def init_df(self):
        if os.path.exists(self.DF_CACHE_PATH):
            print('load df from cache')
            return feather.read_dataframe(self.DF_CACHE_PATH)

        df = get_main_train_df()
        df = df.merge(pd.read_csv('data/input/scalar_coupling_contributions.csv').drop(columns=['type']),
                      on=['molecule_name', 'atom_index_0', 'atom_index_1'])
        types = sorted(df.type.unique())

        # Scale in for each types
        for t in types:
            df_tmp = df[df.type == t]
            if self.scaler == 'std':
                scaled = StandardScaler().fit_transform(df_tmp[self.cols])
            elif self.scaler == 'minmax':
                scaled = MinMaxScaler().fit_transform(df_tmp[self.cols])
            else:
                raise Exception('not supported scaler')
            df.loc[df.type == t, self.cols] = scaled

        df[self.cols] = df[self.cols].astype(np.float32)
        df.to_feather(self.DF_CACHE_PATH)
        return df
