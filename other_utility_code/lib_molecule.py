import numpy as np, pandas as pd, os
from tqdm import tqdm_notebook as tqdm
import scipy.io
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error as mae
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import catboost
import xgboost as xgb

# --------------------------------  Specification ------------------------------------- #

CATEGORICAL_FEATURES = np.load('info_data/CATEGORICAL_FEATURES.npy')
USELESS_FEATURES = [l for l in np.load('info_data/FAIL_FEATURES.npy')]
REMOVED_FEATURES = ['id', 'y', 'molecule_id', 'atom1', 'atom2', 'bondtype', 'scalar_coupling_constant']

PARAMS = {'num_leaves': [50, 50, 50, 50, 50, 50, 50, 50],
             'min_data_in_leaf': [1, 1, 1, 1, 1, 1, 1, 1],
             'max_depth': [15, 15, 15, 15, 15, 15, 15, 15],
             'learning_rate': [.1, .1, .1, .1, .1, .1, .1, .1],
             'objective': ['huber', 'huber', 'huber', 'huber', 'huber', 'huber', 'huber', 'huber'],
             'n_estimators': 10000 * np.array([3, 4, 2, 4, 4, 2, 4, 4])
             }


# PARAMS = {'num_leaves': [100, 100, 100, 100, 100, 100, 100, 100],
#              'min_data_in_leaf': [50, 50, 50, 50, 50, 50, 50, 50],
#              'max_depth': [15,15,15,15,15,15,15,15],
#              'learning_rate': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#              'objective': ['huber', 'huber', 'huber', 'huber', 'huber', 'huber', 'huber', 'huber'],
#              'n_estimators': 10000 * np.array([3, 4, 2, 4, 4, 2, 4, 4])
#              }


def get_datapath(computer):
    if computer=='desktop': DATAPATH = 'I:/Molecule_Kaggle/'
    elif computer=='laptop': DATAPATH = '/home/khahuras/Desktop/Molecule/'
    elif computer=='macbook': DATAPATH = '/Users/voanhkha/Desktop/Molecule_Kaggle/'
    return DATAPATH

class LogMAEMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers
        # (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.   
        # weight parameter can be None.
        # Returns pair (error, weights sum)

        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in xrange(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * (target[i] * approx[i] - math.log(1 + math.exp(approx[i])))

        return error_sum, weight_sum

def LMAE_XGB(preds, dtrain):
    labels = dtrain.get_label()
    return 'LMAE_XGB', np.log(mae(preds, labels))

# --------------------------------------------------------------------- #

def get_params(bondtype):
    LGB_PARAMS = {'num_leaves': PARAMS['num_leaves'][bondtype],
             'min_data_in_leaf': PARAMS['min_data_in_leaf'][bondtype], 
             'objective': PARAMS['objective'][bondtype],
             'max_depth': PARAMS['max_depth'][bondtype],
             'learning_rate': PARAMS['learning_rate'][bondtype],
             "boosting": "gbdt",
             "feature_fraction": 1,
             "bagging_fraction": 0.85,
             "random_state": 240691,
              "num_threads": 4,
              "n_estimators": PARAMS['n_estimators'][bondtype]
             }
    return LGB_PARAMS


def read_file(path, filename):
    #print(filename)
    files = os.listdir(path)
    for f in files:
        if filename == f[:len(filename)]: 
            extension = f.split('.')[-1]
            break
    if extension == 'csv': return pd.read_csv(path  + filename + '.csv')
    elif extension == 'parquet': return pd.read_parquet(path + filename + '.parquet')
    elif extension == 'p': return pd.read_pickle(path + filename + '.p')


def LMAE_LGB(preds, labels):
    return 'LMAE_LGB', np.log(mae(preds, labels)), False

def LMAE_LGB_group(labels, preds, bondtype):
    types = np.unique(bondtype)
    LMAE_types = np.zeros(len(types))
    for i, t in enumerate(types): LMAE_types[i] = np.log(mae(preds[bondtype==t], labels[bondtype==t]))
    return 'LMAE_LGB_group', np.mean(LMAE_types), False

def LMAE(preds, labels):
    return np.log(mae(preds, labels))

class Dataset:
    
    def __init__(self, bondtype, feature_sets=None, oof_features=None, computer='desktop', use_fc=False):
        self.bondtype = bondtype
        DATAPATH = get_datapath(computer)
        
        train_info = pd.read_csv('info_data/info_train_type_'+str(bondtype)+'.csv')
        test_info = pd.read_csv('info_data/info_test_type_'+str(bondtype)+'.csv')
        self.X_id = train_info['id'].values.astype(int)
        self.Xt_id = test_info['id'].values.astype(int)
        self.y = train_info['scalar_coupling_constant'].values
        if use_fc: self.y = train_info['fc'].values
        self.molecule_train = train_info['molecule_name'].values
        self.molecule_test = test_info['molecule_name'].values
        self.unique_molecule_train = pd.unique(self.molecule_train)
        self.unique_molecule_test = pd.unique(self.molecule_test)

        for j, s in enumerate(feature_sets):
            temp_train = read_file(DATAPATH + 'features/' + s + '/', s+'_train_'+str(bondtype))
            temp_test = read_file(DATAPATH + 'features/' + s + '/', s+'_test_'+str(bondtype))
            if j == 0:
                train_df = temp_train
                test_df = temp_test
            else:
                train_df = pd.concat( [train_df , temp_train], axis=1 )
                test_df = pd.concat( [test_df , temp_test], axis=1 )

        useful_features = [ f for f in train_df.columns if f not in USELESS_FEATURES[bondtype-1] + REMOVED_FEATURES ]
        self.feature_names = useful_features
        self.categorical_features = [ f for f in useful_features if f in CATEGORICAL_FEATURES[bondtype-1] ]

        # for col in self.categorical_features:
        #     LE = LabelEncoder().fit(np.concatenate([train_df[col], test_df[col]]))
        #     train_df[col] = LE.transform(train_df[col])
        #     test_df[col] = LE.transform(test_df[col])

        # train_df = train_df.loc[:,~train_df.columns.duplicated()]
        # test_df = test_df.loc[:,~test_df.columns.duplicated()]
        # train_df.fillna(0, inplace=True)
        # test_df.fillna(0, inplace=True)

        self.X = train_df[useful_features].values
        self.Xt = test_df[useful_features].values

        if oof_features is not None:
            oof_features = [s+'_'+str(bondtype) for s in oof_features]
            for f in oof_features:
                df_oof = pd.read_csv('oof_preds/oof_'+f+'.csv')
                self.X = np.concatenate([self.X, df_oof['scalar_coupling_constant'].values[:,None]], axis=1)
                df_oof = pd.read_csv('test_preds/test_'+f+'.csv')
                self.Xt = np.concatenate([self.Xt, df_oof['scalar_coupling_constant'].values[:,None]], axis=1)                
                self.feature_names += [f]
        
class Method:
    
    def __init__(self, name, n_folds=5, params=None, random_state=2019, cv_strategy=None):
        self.name = name
        self.params = params
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_strategy = cv_strategy

    def fit(self, dataset, fit_once):
        X, y, Xt, X_id, Xt_id = dataset.X, dataset.y, dataset.Xt, dataset.X_id, dataset.Xt_id

        self.valid_preds = np.zeros(X.shape[0])
        self.test_preds = np.zeros(Xt.shape[0])
        self.oof_scores = []
        self.feature_importances = pd.DataFrame(data={'feature':dataset.feature_names, 'importance':0})
        
        print('Fitting', self.name, '...')
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        kf_group = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state).split(dataset.unique_molecule_train))
        
        for i, (fold_index_tr, fold_index_va) in enumerate(kf.split(X)):
            print('\n\n')
            
            if self.cv_strategy == 'group':
                print('Group Molecule Split Activated')
                idx_molecule_tr, idx_molecule_va = kf_group[i][0], kf_group[i][1]
                fold_index_tr = np.isin(dataset.molecule_train, dataset.unique_molecule_train[idx_molecule_tr])
                fold_index_va = np.isin(dataset.molecule_train, dataset.unique_molecule_train[idx_molecule_va])
            else: print('Group Molecule Split Not Used')

            print(fold_index_tr[:10], fold_index_va[:10])

            print('Training fold', i+1, '/', self.n_folds, 'type', dataset.bondtype, '...')
            X_tr, y_tr = X[fold_index_tr], y[fold_index_tr]
            X_va, y_va = X[fold_index_va], y[fold_index_va]
            print('True train size:', X_tr.shape, 'Valid size:', X_va.shape)

            if self.name == 'LGB': 
                model = lgb.LGBMRegressor(**self.params, n_jobs = -1) #self.params['n_estimators']
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=LMAE_LGB, \
                        verbose=200, early_stopping_rounds=50, feature_name=dataset.feature_names, \
                            categorical_feature=dataset.categorical_features)
                oof_preds = model.predict(X_va)#model.evals_result_['valid_0']['LMAE_LGB'][-1]
                self.valid_preds[fold_index_va] = oof_preds
                test_preds_this_fold = model.predict(Xt)
                self.test_preds += test_preds_this_fold/self.n_folds
                self.feature_importances["importance"] += model.feature_importances_


            if self.name == 'CatBoost':
                model = catboost.CatBoostRegressor(loss_function='RMSE', n_estimators = 20000, eval_metric="MAE") 
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose_eval=50, early_stopping_rounds=50)
                oof_preds = model.predict(X_va)
                self.valid_preds[fold_index_va] = oof_preds
                test_preds_this_fold = model.predict(Xt)
                self.test_preds += test_preds_this_fold/self.n_folds


            if self.name == 'XGB':
                XGB_PARAMS = { "learning_rate":0.1, "n_estimators":500, "max_depth":12, 
                       "min_child_weight": 40, "subsample":0.7, "objective":'reg:linear', 
                       "nthread":4, "scale_pos_weight":1, "seed":777, "base_score":np.mean(y_tr)}
                model = xgb.XGBRegressor(**XGB_PARAMS) 
                model.fit(X_tr, y_tr, verbose=10, early_stopping_rounds=30, \
                eval_set=[(X_va, y_va)], eval_metric=LMAE_XGB)

                oof_preds = model.predict(X_va)
                self.valid_preds[fold_index_va] = oof_preds
                test_preds_this_fold = model.predict(Xt)
                self.test_preds += test_preds_this_fold/self.n_folds
                #best_iter = int(model.best_ntree_limit*1.2)

                if fit_once:
                    print('Refitting on all data using', 600, 'trees...')
                    XGB_PARAMS = { "learning_rate":0.1, "n_estimators":600, "max_depth":15, 
                       "min_child_weight": 40, "subsample":0.7, "objective":'reg:linear', 
                       "nthread":4, "scale_pos_weight":1, "seed":27, "base_score":np.mean(y_tr)}
                    model = xgb.XGBRegressor(**XGB_PARAMS) 
                    model.fit(X, y, verbose=10)
                    self.test_preds = model.predict(Xt)
                    break
                
            LMAE_fold = LMAE(y_va, oof_preds)
            self.oof_scores.append(LMAE_fold)
            print('OOF LMAE fold', i+1, ':', round4(LMAE_fold))
            self.feature_importances.sort_values('importance', ascending=False).reset_index(drop=True).to_csv('importance_lgb/feature_importance_type_'+ str(dataset.bondtype) +'.csv',index=None)     
            
        print('Avaraged OOF LMAE:', round4(LMAE(y, self.valid_preds)))
        print('----------------------------------------------------\n\n')
   
def round4(x):
    return np.round(x, 4)

