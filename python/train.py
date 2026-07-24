import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from openfe import OpenFE, transform
from impute import impute_file
from data_process import prepare_training_data
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

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
    'metric': ['multi_logloss', 'multi_error'],
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

def average_history(history_list):
    max_len = max(len(h) for h in history_list)
    padded = []
    for h in history_list:
        if len(h) < max_len:
            h_list = list(h)
            padded.append(h_list + [h_list[-1]] * (max_len - len(h)))
        else:
            padded.append(list(h))
    return np.mean(padded, axis=0)

def train(train_file):
    print('Membaca data pelatihan dinamis')
    data_raw = pd.read_csv(train_file, low_memory=False)
    data_prep = prepare_training_data(data_raw)
    
    temp_file = os.path.join(root, 'data/temp/train_prep.csv')
    data_prep.to_csv(temp_file, index=False)
    
    imputed_file = impute_file(temp_file)
    data_iterative_imputer = pd.read_csv(imputed_file, low_memory=False)
    
    for col in feature_list_new:
        if col in data_iterative_imputer.columns:
            data_prep[col] = data_iterative_imputer[col]
        else:
            data_prep[col] = 0.0
            
    cols_front = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'CLASS', 'gene']
    cols_other = [c for c in data_prep.columns if c not in cols_front]
    data_final = data_prep[cols_front + cols_other]
    
    data_autofe = autofe(data_final)
    
    X_train = data_autofe.iloc[:, 7:]
    Y_train = data_autofe['CLASS']
    
    # Map classes: -1 -> 0 (Benign), 0 -> 1 (VUS), 1 -> 2 (Pathogenic)
    label_map = {-1: 0, 0: 1, 1: 2}
    class_names = ['Benign', 'VUS', 'Pathogenic']
    y_mapped = Y_train.map(label_map).values
    
    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    oof_preds = np.zeros((len(X_train), 3))
    
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accuracies = []
    fold_val_accuracies = []
    
    print('Melatih model LightGBM multikelas dengan Stratified 5-Fold CV')
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_mapped)):
        print(f'Training Fold {fold + 1}...')
        X_tr, y_tr = X_train.iloc[train_idx], y_mapped[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_mapped[val_idx]
        
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_eval = lgb.Dataset(X_va, y_va, reference=lgb_train)
        
        evals_result = {}
        gbm = lgb.train(
            selection_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_eval],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.record_evaluation(evals_result)
            ]
        )
        models.append(gbm)
        
        oof_preds[val_idx] = gbm.predict(X_va)
        
        fold_train_losses.append(evals_result['train']['multi_logloss'])
        fold_val_losses.append(evals_result['valid']['multi_logloss'])
        
        fold_train_accuracies.append([1 - e for e in evals_result['train']['multi_error']])
        fold_val_accuracies.append([1 - e for e in evals_result['valid']['multi_error']])
        
    # Save the ensemble models
    filename = os.path.splitext(os.path.basename(train_file))[0]
    model_path = os.path.join(root, 'data/result/MAGPIE.model')
    joblib.dump(models, model_path)
    model_path_suffix = os.path.join(root, f'data/result/MAGPIE_{filename}.model')
    joblib.dump(models, model_path_suffix)
    print('Model ensemble sukses disimpan')
    
    # ------------------ Visualizations ------------------
    vis_dir = os.path.join(root, 'data/output/visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Loss Curve
    avg_train_loss = average_history(fold_train_losses)
    avg_val_loss = average_history(fold_val_losses)
    plt.figure(figsize=(8, 6))
    plt.plot(avg_train_loss, label='Train Loss', color='teal', lw=2)
    plt.plot(avg_val_loss, label='Val Loss', color='orange', lw=2)
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Boosting Round')
    plt.ylabel('Multi-Logloss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'loss_curve.pdf'), format='pdf', dpi=300)
    plt.savefig(os.path.join(vis_dir, 'loss_curve.svg'), format='svg', dpi=300)
    plt.close()
    
    # 2. Accuracy Curve
    avg_train_acc = average_history(fold_train_accuracies)
    avg_val_acc = average_history(fold_val_accuracies)
    plt.figure(figsize=(8, 6))
    plt.plot(avg_train_acc, label='Train Acc', color='teal', lw=2)
    plt.plot(avg_val_acc, label='Val Acc', color='orange', lw=2)
    plt.title('Training & Validation Accuracy Curve')
    plt.xlabel('Boosting Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accuracy_curve.pdf'), format='pdf', dpi=300)
    plt.savefig(os.path.join(vis_dir, 'accuracy_curve.svg'), format='svg', dpi=300)
    plt.close()
    
    # 3. ROC Curve (One-vs-Rest)
    plt.figure(figsize=(8, 8))
    colors_list = ['teal', 'orange', 'crimson']
    for c in range(3):
        y_true_c = (y_mapped == c).astype(int)
        y_score_c = oof_preds[:, c]
        fpr, tpr, _ = roc_curve(y_true_c, y_score_c)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors_list[c], lw=2, label=f'ROC {class_names[c]} (AUC = {roc_auc:.3f})')
        
    y_onehot = label_binarize(y_mapped, classes=[0, 1, 2])
    fpr_macro, tpr_macro, _ = roc_curve(y_onehot.ravel(), oof_preds.ravel())
    macro_auc = auc(fpr_macro, tpr_macro)
    plt.plot(fpr_macro, tpr_macro, color='navy', linestyle='--', lw=2, label=f'Macro-average ROC (AUC = {macro_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class One-vs-Rest ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'roc_curve.pdf'), format='pdf', dpi=300)
    plt.savefig(os.path.join(vis_dir, 'roc_curve.svg'), format='svg', dpi=300)
    plt.close()
    
    # 4. PRC Curve (One-vs-Rest)
    plt.figure(figsize=(8, 8))
    for c in range(3):
        y_true_c = (y_mapped == c).astype(int)
        y_score_c = oof_preds[:, c]
        precision, recall, _ = precision_recall_curve(y_true_c, y_score_c)
        prc_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors_list[c], lw=2, label=f'PRC {class_names[c]} (AUPRC = {prc_auc:.3f})')
        
    precision_macro, recall_macro, _ = precision_recall_curve(y_onehot.ravel(), oof_preds.ravel())
    macro_prc_auc = auc(recall_macro, precision_macro)
    plt.plot(recall_macro, precision_macro, color='navy', linestyle='--', lw=2, label=f'Macro-average PRC (AUPRC = {macro_prc_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class One-vs-Rest PRC Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'prc_curve.pdf'), format='pdf', dpi=300)
    plt.savefig(os.path.join(vis_dir, 'prc_curve.svg'), format='svg', dpi=300)
    plt.close()
    
    # 5. Confusion Matrix
    pred_labels = np.argmax(oof_preds, axis=1)
    cm = confusion_matrix(y_mapped, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confusion_matrix.pdf'), format='pdf', dpi=300)
    plt.savefig(os.path.join(vis_dir, 'confusion_matrix.svg'), format='svg', dpi=300)
    plt.close()
    
    # 6. Classification Report Table
    report_dict = classification_report(y_mapped, pred_labels, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('Classification Report', fontsize=16, y=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'classification_report.pdf'), format='pdf', dpi=300)
    plt.savefig(os.path.join(vis_dir, 'classification_report.svg'), format='svg', dpi=300)
    plt.close()
    
    print('Visualisasi metrik evaluasi (PDF & SVG) sukses disimpan')
