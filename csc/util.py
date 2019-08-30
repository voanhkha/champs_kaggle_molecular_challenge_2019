import os
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from csc import const


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def set_mulliken_charge(df, node_cols):
    df_mc = pd.read_csv('data/artifacts/mulliken_charge_all.csv')

    for n in node_cols:
        df = df \
            .merge(df_mc, how='left', left_on=['molecule_name', n], right_on=['molecule_name', 'atom_index']) \
            .drop('atom_index', axis=1) \
            .rename(columns={'mulliken_charge': 'mulliken_charge_{}'.format(n[1:])})

    return df


def swap_cols(df, cols0, cols1):
    df_ = df.copy().reset_index(drop=True)
    df_[cols0 + cols1] = df_[cols1 + cols0]
    return df_


def make_swapped_abc(df):
    all_cols = df.columns.values
    cols_1a = sorted([c for c in all_cols if re.search(r"_1a", c)])
    cols_1b = sorted([c for c in all_cols if re.search(r"_1b", c)])
    cols_1c = sorted([c for c in all_cols if re.search(r"_1c", c)])

    return pd.concat((
        swap_cols(df, cols_1a, cols_1b),
        swap_cols(df, cols_1a, cols_1c),
        swap_cols(df, cols_1b, cols_1c),
    ), ignore_index=True)


def bit_vect_to_int(bitvector):
    bitstring = bitvector.ToBitString()
    return list(map(np.int8, bitstring))


def get_type_encoder() -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(const.TYPES)
    return enc


def get_type_onehot_encoder() -> OneHotEncoder:
    enc = OneHotEncoder(sparse=False, dtype=int)
    enc.fit(np.array(const.TYPES).reshape(-1, 1))
    return enc


if __name__ == '__main__':
    enc = get_type_onehot_encoder()
    # print(enc.categories_)
    # result = enc.inverse_transform([1, 4])
    result = enc.transform([['2JHN']])
    print(result)
