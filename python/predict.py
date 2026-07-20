import joblib
import pandas as pd
import numpy as np
import os
from openfe import transform
from impute import impute_file
from data_process import prepare_training_data

target_dir = '/kaggle/working/data/output/visualization'
os.makedirs(target_dir, exist_ok=True)
print('Direktori visualisasi siap')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predict(test_file, autoFE_features, selection, model_file, filename, file_state):
    feature_list = [
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

    test = pd.read_csv(test_file, low_memory=False)
    test = prepare_training_data(test)
    
    temp_file = os.path.join(root, f'data/temp/{filename}_pred_prep.csv')
    test.to_csv(temp_file, index=False)
    
    imputed_file = impute_file(temp_file)
    data_iterative_imputer = pd.read_csv(imputed_file, low_memory=False)
    
    for col in feature_list:
        if col in data_iterative_imputer.columns:
            test[col] = data_iterative_imputer[col]
        else:
            test[col] = 0.0

    X_test = test[feature_list].astype('float64')
    features = joblib.load(autoFE_features)
    
    train_dummy = pd.DataFrame(columns=feature_list)
    train_dummy.loc[0] = 0.0
    train_dummy.loc[1] = 1.0
    
    _, X_test_tr = transform(train_dummy, X_test, features, n_jobs=4)
    
    if os.path.exists(selection):
        feature_list_sel = pd.read_csv(selection)['feature'].tolist()
        valid_cols = [c for c in feature_list_sel if c in X_test_tr.columns]
        X_matrix = X_test_tr[valid_cols].astype('float64').values
    else:
        X_matrix = X_test_tr.astype('float64').values

    print('Memulai perhitungan probabilitas')
    model = joblib.load(model_file)
    test_pred_prob = model.predict(X_matrix)
    
    test_pred_label = np.argmax(test_pred_prob, axis=1)
    
    result_df = pd.concat([test.reset_index(drop=True), pd.DataFrame(test_pred_label, columns=['MAGPIE_pred'])], axis=1)
    result_df.to_csv(os.path.join(root, f'data/result/{filename}.csv'), index=False)
    
    return result_df
