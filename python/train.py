import os
import joblib
import pandas as pd
import lightgbm as lgb
from openfe import OpenFE, transform
from sklearn.model_selection import train_test_split
from impute import impute_file
from data_process import prepare_training_data
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feature_list_new = [
    'phastConsElements100way', 'phyloP100way_vertebrate', 'phyloP20way_mammalian',
    'phastCons100way_vertebrate', 'phastCons20way_mammalian', 'SiPhy_29way_logOdds',
    'phyloP30way_mammalian', 'phastCons30way_mammalian', 'AF', 'AF_raw', 'AF_male',
    'AF_female', 'AF_afr', 'AF_ami', 'AF_amr', 'AF_asj', 'AF_eas', 'AF_fin', 'AF_nfe',
    'AF_oth', 'gdi', 'gdi_phred', 'rvis1', 'rvis2', 'lof_score', 'molecular_weight',
    'equipotential_point', 'hydrophilic', 'hydrophobic', 'amphipathic', 'cyclic',
    'essential', 'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic',
    'basic', 'sulfur', 'pka_cooh', 'pka_nh3', 'blosum100', 'DS_AG', 'DS_AL',
    'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL', 'Gm12878',
    'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
    'func_frameshift', 'func_nonframeshift', 'func_nonsynonymous SNV',
    'func_startloss', 'func_stopgain', 'func_stoploss',
    'omim_Autosomal_dominant', 'omim_Autosomal_recessive',
    'omim_X_linked_dominant', 'omim_X_linked_recessive', 'omim_other'
]

selection_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': 40,
    'max_depth': 6,
    'max_bin': 255,
    'min_data_in_leaf': 101,
    'learning_rate': 0.01,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 45,
    'lambda_l1': 0.001,
    'lambda_l2': 0.4,
    'min_split_gain': 0.0,
    'verbose': -1,
    'is_unbalance': True
}

def autofe(data):
    print('Memulai tahapan OpenFE')
    train_info = data.iloc[:, :7].copy()
    X_train = data[feature_list_new].astype('float64')
    Y_train = data['CLASS']
    
    ofe = OpenFE()
    features = ofe.fit(data=X_train, label=Y_train, n_jobs=4)
    
    features_path = os.path.join(root, 'data/result/openFE.features')
    joblib.dump(features, features_path)
    
    X_train_tr, _ = transform(X_train, X_train.iloc[:2], features, n_jobs=4)
    X_train_tr.index = list(range(X_train_tr.shape[0]))
    train_info.index = list(range(train_info.shape[0]))
    
    selection_path = os.path.join(root, 'data/result/selection.csv')
    pd.DataFrame({'feature': X_train_tr.columns}).to_csv(selection_path, index=False)
    
    return pd.concat([train_info, X_train_tr], axis=1)

def train(train_file):
    print('Membaca data pelatihan dinamis')
    data_raw = pd.read_csv(train_file, low_memory=False)
    data_prep = prepare_training_data(data_raw)
    
    temp_file = os.path.join(root, 'data/temp/train_prep.csv')
    data_prep.to_csv(temp_file, index=False)
    
    imputed_file = impute_file(temp_file)
    data_bpca = pd.read_csv(imputed_file, low_memory=False)
    
    for col in feature_list_new:
        if col in data_bpca.columns:
            data_prep[col] = data_bpca[col]
        else:
            data_prep[col] = 0.0
            
    cols_front = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'CLASS', 'gene']
    cols_other = [c for c in data_prep.columns if c not in cols_front]
    data_final = data_prep[cols_front + cols_other]
    
    data_autofe = autofe(data_final)
    
    X_train = data_autofe.iloc[:, 7:]
    Y_train = data_autofe['CLASS']
    
    X, val_X, y, val_y = train_test_split(X_train, Y_train, test_size=0.1, random_state=1, stratify=Y_train)
    lgb_train = lgb.Dataset(X, y)
    lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train)

    print('Melatih model LightGBM multikelas')
    gbm = lgb.train(
        selection_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    model_path = os.path.join(root, 'data/result/MAGPIE.model')
    joblib.dump(gbm, model_path)
    print('Model sukses disimpan')
