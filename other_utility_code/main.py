import numpy as np
import pandas as pd
from lib_molecule import Dataset, Method, get_params

# -------------------------------- Main Run Specification ------------------------------------- #

KFOLD_SEED = 546
TRAIN_TYPES = [3]
FEATURE_SETS = ['lgb01']
COMPUTER = 'macbook'  
OUTFILE = 'schnet_stack'#stack_baseline_seed111'
OOF_TO_USE = ['schnet']
USE_FC = False 
METHOD = 'LGB'
CV_STRATEGY = None  #'group'
FIT_ONCE = False

# ---------------------------------------- Main ------------------------------------------------ #

for bondtype in TRAIN_TYPES:    
    print('Bond types', bondtype)
    D = Dataset(bondtype, feature_sets=FEATURE_SETS, oof_features=OOF_TO_USE, computer=COMPUTER, use_fc=USE_FC)
    LGB_model = Method(METHOD, params=get_params(bondtype-1), random_state=KFOLD_SEED, cv_strategy=CV_STRATEGY)
    LGB_model.fit(D, FIT_ONCE)

    sub = pd.DataFrame(data={'id':D.Xt_id, 'scalar_coupling_constant':LGB_model.test_preds})
    sub.to_csv('test_preds/test_'+ OUTFILE +'_'+str(bondtype)+'.csv', index=None)
    
    sub = pd.DataFrame(data={'id':D.X_id, 'scalar_coupling_constant':LGB_model.valid_preds})
    sub.to_csv('oof_preds/oof_' + OUTFILE + '_'+str(bondtype)+'.csv', index=None)