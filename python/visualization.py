import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from utils import get_prior, cal_pr_values, cal_metrics_visualization

# Set plot style for premium aesthetics
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
sns.set_style('whitegrid')

tick_font = 20
label_font = 24
legend_font = 18

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(root, 'data/output/visualization/')
os.makedirs(save_dir, exist_ok=True)

# Define full list of possible models and their color scheme
all_model_names = ['ClinPred', 'REVEL', 'VARITY', 'MetaSVM', 'MetaLR',
                   'VEST4', 'M-CAP', 'MutPred', 'PrimateAI', 'MutationAssessor',
                   'LIST-S2', 'SIFT4G', 'DANN', 'MutationTaster', 'SenusaBio', 'BIAS-2015']
all_colors = ['rosybrown', 'Teal', 'steelblue', 'DarkSlateGrey', 'OliveDrab',
              'DarkSeaGreen', 'CadetBlue', 'CornflowerBlue', 'peru', 'DarkKhaki',
              'GoldenRod', 'DarkGrey', 'LightBlue', 'royalblue', 'crimson', 'purple']
color_dict = dict(zip(all_model_names, all_colors))

def map_multiclass_labels(series):
    """
    Standardize labels from different sources to:
    -1 (Benign), 0 (VUS), 1 (Pathogenic)
    """
    mapping = {
        'pathogenic': 1,
        'likely pathogenic': 1,
        '1': 1,
        '1.0': 1,
        1: 1,
        1.0: 1,
        
        'uncertain': 0,
        'uncertain significance': 0,
        'vus': 0,
        '0': 0,
        '0.0': 0,
        0: 0,
        0.0: 0,
        
        'benign': -1,
        'likely benign': -1,
        '-1': -1,
        '-1.0': -1,
        -1: -1,
        -1.0: -1
    }
    
    mapped = []
    for val in series:
        if pd.isna(val):
            mapped.append(np.nan)
            continue
        val_str = str(val).strip().lower()
        if val_str in mapping:
            mapped.append(mapping[val_str])
        elif val in mapping:
            mapped.append(mapping[val])
        else:
            mapped.append(np.nan)
    return pd.Series(mapped)

# ROC plot (Binary subset)
def multi_models_roc(names, colors, data, save=None, dpin=500):
    plt.figure(figsize=(10, 10), dpi=dpin)
    plot_list = []
    label_list = []

    sns.set(context=None, style='white', palette='deep', font='Times New Roman', font_scale=1.3, color_codes=True, rc=None)

    order_dict = {}
    order_index = 0
    for (name, colorname) in zip(names, colors):
        df = data.loc[:, [name, 'CLASS']].dropna().astype('float64')
        if df.empty:
            continue
        fpr, tpr, thresholds = roc_curve(df.CLASS, df[name].to_list(), pos_label=1)
        score = auc(fpr, tpr)
        label = '{} ({:.3f})'.format('_'.join(name.split('_')[:-1]) if '_' in name else name, score)
        order_dict[score] = order_index
        order_index += 1
        label_list.append(label)
        plt_temp, = plt.plot(fpr, tpr, lw=5, label=label, color=colorname)
        plot_list.append(plt_temp)
        
    plt.plot([0, 1], [0, 1], '--', lw=5, color='black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=tick_font)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=tick_font)
    plt.xlabel('False Positive Rate', fontsize=label_font)
    plt.ylabel('True Positive Rate', fontsize=label_font)
    
    if order_dict:
        order_list = pd.DataFrame.from_dict({'AUC': order_dict.keys(), 'order': order_dict.values()}).sort_values(by='AUC',
                                                                                                                  ascending=False).order.tolist()
        plt.legend(handles=[plot_list[idx] for idx in order_list],
                   labels=[label_list[idx] for idx in order_list],
                   loc='lower right', fontsize=legend_font)

    if save is not None:
        fmt = os.path.splitext(save)[1][1:]
        plt.savefig(save, dpi=500, bbox_inches='tight', format=fmt)
    plt.close()

# PRC plot (Binary subset)
def multi_models_pr(names, colors, data, mode, save=None, dpin=500, loc='upper left'):
    sns.set(context=None, style='white', palette='deep', font='Times New Roman', font_scale=1.3, color_codes=True, rc=None)

    test_precision, test_recall = [0.9, 0.9]
    plt.figure(figsize=(10, 10), dpi=dpin)
    label_list = []
    plot_list = []
    order_dict = {}
    order_index = 0
    for (name, colorname) in zip(names, colors):
        df = data.loc[:, [name, 'CLASS']].dropna().astype('float64')
        if df.empty:
            continue
        if 'SIFT4G' in name:
            df.SIFT4G_score = [1 - i for i in df.SIFT4G_score.tolist()]
        y = df.CLASS.tolist()
        y_predicted = df[name].to_list()
        prior = get_prior(y)
        precisions, recalls, prc_thresholds = precision_recall_curve(y, y_predicted)
        recalls = np.insert(recalls, 0, 1)
        precisions = np.insert(precisions, 0, prior)
        balanced_precisions = precisions * (1 - prior) / (precisions * (1 - prior) + (1 - precisions) * prior)
        balanced_recalls = recalls
        if mode == 'prc':
            [auprc, up_auprc, rfp, pfr] = cal_pr_values(precisions, recalls, test_precision, test_recall)
            label = '{} ({:.3f})'.format('_'.join(name.split('_')[:-1]) if '_' in name else name, auprc)
            order_dict[auprc] = order_index
            order_index += 1
            label_list.append(label)
            plt_temp, = plt.plot(recalls, precisions, lw=5, label=label, color=colorname)
            plot_list.append(plt_temp)
        elif mode == 'bprc':
            [aubprc, up_aubprc, brfp, bpfr] = cal_pr_values(balanced_precisions, balanced_recalls, test_precision,
                                                            test_recall)
            label = '{} ({:.3f})'.format('_'.join(name.split('_')[:-1]) if '_' in name else name, aubprc)
            order_dict[aubprc] = order_index
            order_index += 1
            label_list.append(label)
            plt_temp, = plt.plot(balanced_recalls, balanced_precisions, lw=5, label=label, color=colorname)
            plot_list.append(plt_temp)

        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(fontsize=tick_font)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=tick_font)
        plt.xlabel('Precision', fontsize=label_font)
        plt.ylabel('Recall', fontsize=label_font)

    if order_dict:
        order_list = pd.DataFrame.from_dict({'AUC': order_dict.keys(), 'order': order_dict.values()}).sort_values(
            by='AUC', ascending=False).order.tolist()
        plt.legend(handles=[plot_list[idx] for idx in order_list],
                   labels=[label_list[idx] for idx in order_list],
                   loc=loc, fontsize=legend_font)

    if save is not None:
        fmt = os.path.splitext(save)[1][1:]
        plt.savefig(save, dpi=500, bbox_inches='tight', format=fmt)
    plt.close()

# 10-metrics plot including Missing rate (Binary subset)
def get_metrics_plot(df, type_name, name_color_dict, save=None, metric_list=None, height=8, n_cols=5):
    sns.set(context=None, style='white', palette='deep', font='Times New Roman', font_scale=3, color_codes=True, rc=None)
    
    if len(metric_list) == 10:
        fig, ax = plt.subplots(nrows=2, ncols=n_cols, sharex=True, sharey=False, figsize=(20, height), dpi=500)
    elif len(metric_list) == n_cols:
        fig, ax = plt.subplots(nrows=1, ncols=n_cols, sharex=True, sharey=False, figsize=(20, height), dpi=500)
        
    subplot_index = 1
    for metric in metric_list:
        if len(metric_list) == 10:
            plt.subplot(2, 5, subplot_index)
        elif len(metric_list) == n_cols:
            plt.subplot(1, n_cols, subplot_index)
        plt.title(metric, fontsize=24)
        plt.xlim(None)
        
        # Filter data to keep only models in the dictionary
        plot_df = df[df.model_name.isin(name_color_dict.keys())]
        
        sns.scatterplot(data=plot_df, x=metric, y='model_name',
                        c=[name_color_dict[i] for i in plot_df.model_name.tolist()], s=100)
        if subplot_index % n_cols == 1:
            plt.ylabel("Model", fontsize=label_font)
            plt.yticks(fontsize=tick_font)
        else:
            plt.ylabel(None)
            plt.yticks([])
        if len(metric_list) == 10:
            if subplot_index >= 6:
                plt.xticks([0, 0.5, 1], fontsize=tick_font)
            else:
                plt.xticks(None)
        elif len(metric_list) == n_cols:
            plt.xticks([0, 0.5, 1], fontsize=tick_font)
        subplot_index += 1
        plt.xlabel(None)
        
    if save:
        fmt = os.path.splitext(save)[1][1:]
        plt.savefig(save, dpi=500, bbox_inches='tight', format=fmt)
    plt.close()

# Confusion Matrices (Multiclass)
def plot_confusion_matrices(df, true_col, pred_cols, col_to_name, labels, class_names, save_path):
    n_cols = len(pred_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5), squeeze=False)
    
    palettes = {
        'SenusaBio': 'Blues',
        'BIAS-2015': 'Purples'
    }
    
    for i, pred_col in enumerate(pred_cols):
        ax = axes[0, i]
        temp_df = df[[true_col, pred_col]].dropna()
        if temp_df.empty:
            ax.text(0.5, 0.5, f"No predictions for\n{col_to_name.get(pred_col, pred_col)}", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue
            
        y_true = temp_df[true_col].astype(int).values
        y_pred = temp_df[pred_col].astype(int).values
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Calculate percentage for annotations
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = np.zeros_like(cm, dtype=float)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                if cm_sum[r, 0] > 0:
                    cm_perc[r, c] = cm[r, c] / cm_sum[r, 0] * 100
                    
        annot = np.empty_like(cm, dtype=object)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm[r, c]}\n({cm_perc[r, c]:.1f}%)"
        
        model_name = col_to_name.get(pred_col, pred_col)
        cmap = palettes.get(model_name, 'Blues')
        
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, 
                    xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False,
                    annot_kws={"size": 13, "weight": "bold"}, linewidths=1, linecolor='gray')
        
        ax.set_title(model_name, fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')
            
    plt.tight_layout()
    fmt = os.path.splitext(save_path)[1][1:]
    plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.close()

# Classification Report Table (Multiclass)
def save_classification_report_table(df, true_col, pred_cols, col_to_name, labels, class_names, save_path):
    for pred_col in pred_cols:
        model_name = col_to_name.get(pred_col, pred_col)
        temp_df = df[[true_col, pred_col]].dropna()
        if temp_df.empty:
            continue
            
        y_true = temp_df[true_col].astype(int).values
        y_pred = temp_df[pred_col].astype(int).values
        
        report_dict = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.axis('off')
        table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title(f'Classification Report ({model_name})', fontsize=16, y=0.95, fontweight='bold')
        plt.tight_layout()
        
        # Add model name suffix to avoid overwriting if there are multiple models
        base, ext = os.path.splitext(save_path)
        fmt = ext[1:]
        plt.savefig(f"{base}_{model_name}{ext}", format=fmt, dpi=300)
        plt.close()

def run_evaluation_pipeline(data_df, filename_suffix):
    """
    Evaluates both multiclass and binary metrics on the provided dataframe.
    """
    # 1. Map labels to standard multiclass
    data_df['CLASS_mapped'] = map_multiclass_labels(data_df['CLASS'])
    
    # Identify Prediction Columns
    pred_cols = []
    col_to_name = {}
    
    senusa_col = None
    if 'SenusaBio_pred' in data_df.columns:
        senusa_col = 'SenusaBio_pred'
    elif 'MAGPIE_pred' in data_df.columns:
        senusa_col = 'MAGPIE_pred'
        
    if senusa_col:
        mapped_col = f'{senusa_col}_mapped'
        data_df[mapped_col] = map_multiclass_labels(data_df[senusa_col])
        pred_cols.append(mapped_col)
        col_to_name[mapped_col] = 'SenusaBio'
        
    bias_col = None
    for col in data_df.columns:
        if 'bias' in col.lower() or 'bias-2015' in col.lower():
            bias_col = col
            break
            
    if bias_col:
        mapped_col = f'{bias_col}_mapped'
        data_df[mapped_col] = map_multiclass_labels(data_df[bias_col])
        pred_cols.append(mapped_col)
        col_to_name[mapped_col] = 'BIAS-2015'

    if not pred_cols:
        print("Warning: No prediction columns found for evaluation.")
        return
        
    labels = [-1, 0, 1]
    class_names = ['Benign', 'VUS', 'Pathogenic']
    
    # ------------------ Multiclass Visualizations (SVG & PDF) ------------------
    # Confusion Matrix
    plot_confusion_matrices(data_df, 'CLASS_mapped', pred_cols, col_to_name, labels, class_names, 
                            save_path=f'{save_dir}{filename_suffix}_confusion_matrix.pdf')
    plot_confusion_matrices(data_df, 'CLASS_mapped', pred_cols, col_to_name, labels, class_names, 
                            save_path=f'{save_dir}{filename_suffix}_confusion_matrix.svg')
    
    # Classification Report
    save_classification_report_table(data_df, 'CLASS_mapped', pred_cols, col_to_name, labels, class_names, 
                                     save_path=f'{save_dir}{filename_suffix}_classification_report.pdf')
    save_classification_report_table(data_df, 'CLASS_mapped', pred_cols, col_to_name, labels, class_names, 
                                     save_path=f'{save_dir}{filename_suffix}_classification_report.svg')

    # ------------------ Binary Visualizations (MAGPIE Ori specs - SVG & PDF) ------------------
    # Filter dataset to binary CLASS values only (exclude true VUS)
    binary_df = data_df[data_df['CLASS_mapped'].isin([-1, 1])].copy()
    
    if not binary_df.empty:
        # Standardize binary target: change -1 to 0, 1 to 1 for ROC/PR computations
        binary_df['CLASS'] = [1 if c == 1 else 0 for c in binary_df['CLASS_mapped']]
        
        # Build active models list dynamically based on what is in test dataset
        orig_model_names = ['ClinPred', 'REVEL', 'VARITY', 'MetaSVM', 'MetaLR',
                            'VEST4', 'M-CAP', 'MutPred', 'PrimateAI', 'MutationAssessor',
                            'LIST-S2', 'SIFT4G', 'DANN', 'MutationTaster']
        orig_model_cols = ['ClinPred_score', 'REVEL_score', 'VARITY_R', 'MetaSVM_score', 'MetaLR_score',
                           'VEST4_score', 'M-CAP_score', 'MutPred_score', 'PrimateAI_score', 'MutationAssessor_score',
                           'LIST-S2_score', 'SIFT4G_score', 'DANN_rankscore', 'MutationTaster_score']
        orig_thresholds = [0.5, 0.5, 0.5, 0.5, 0.5,
                           0.5, 0.025, 0.79, 0.8, 0.8,
                           0.85, 0.95, 0.5, 0.3]
                           
        active_names = []
        active_cols = []
        active_thresholds = []
        
        for name, col, thresh in zip(orig_model_names, orig_model_cols, orig_thresholds):
            if col in binary_df.columns:
                active_names.append(name)
                # Ensure float data type
                binary_df[col] = binary_df[col].astype('float64')
                active_cols.append(col)
                active_thresholds.append(thresh)
                
        # Add SenusaBio mapped predictions
        if senusa_col:
            # Map 0 (VUS predictions) to 0 (Benign) in binary evaluation context
            pred_col_name = 'SenusaBio_binary'
            binary_df[pred_col_name] = [1.0 if val == 1.0 else 0.0 for val in binary_df[f'{senusa_col}_mapped']]
            active_names.append('SenusaBio')
            active_cols.append(pred_col_name)
            active_thresholds.append(0.5)
            
        # Add BIAS-2015 mapped predictions
        if bias_col:
            pred_col_name = 'BIAS_binary'
            binary_df[pred_col_name] = [1.0 if val == 1.0 else 0.0 for val in binary_df[f'{bias_col}_mapped']]
            active_names.append('BIAS-2015')
            active_cols.append(pred_col_name)
            active_thresholds.append(0.5)
            
        if active_cols:
            # SIFT4G score needs inverting
            df_roc = binary_df.copy()
            if 'SIFT4G_score' in df_roc.columns:
                df_roc.SIFT4G_score = [1 - i if str(i) != 'nan' else np.nan for i in df_roc.SIFT4G_score]
                
            active_colors_list = [color_dict.get(n, 'black') for n in active_names]
            
            # ROC Curves
            multi_models_roc(active_cols, active_colors_list, df_roc, save=f'{save_dir}{filename_suffix}_AUC.pdf')
            multi_models_roc(active_cols, active_colors_list, df_roc, save=f'{save_dir}{filename_suffix}_AUC.svg')
            
            # PRC Curves
            multi_models_pr(active_cols, active_colors_list, binary_df, mode='prc', save=f'{save_dir}{filename_suffix}_PRC.pdf')
            multi_models_pr(active_cols, active_colors_list, binary_df, mode='prc', save=f'{save_dir}{filename_suffix}_PRC.svg')
            
            # Performance metrics plot (horizontal dots showing Missing rate, MCC, Accuracy, Precision, Recall, etc.)
            df_metrics = binary_df[active_cols + ['CLASS']]
            df_metrics.columns = active_names + ['CLASS']
            if 'SIFT4G' in df_metrics.columns:
                df_metrics.SIFT4G = [1 - i for i in df_metrics.SIFT4G.tolist()]
                
            # Use original MAGPIE version calculations
            df_performance = cal_metrics_visualization(df_metrics, 'old', active_thresholds, active_names)
            
            # Set up color map for plotting
            active_color_dict = {n: color_dict.get(n, 'black') for n in active_names}
            
            get_metrics_plot(df_performance, filename_suffix, name_color_dict=active_color_dict,
                             metric_list=['Missing rate', 'MCC', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'F_beta-score',
                                          'G-mean', 'AUPRC', 'AUBPRC'],
                             save=f'{save_dir}{filename_suffix}_performance.pdf')
            get_metrics_plot(df_performance, filename_suffix, name_color_dict=active_color_dict,
                             metric_list=['Missing rate', 'MCC', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'F_beta-score',
                                          'G-mean', 'AUPRC', 'AUBPRC'],
                             save=f'{save_dir}{filename_suffix}_performance.svg')

def visualize(test, filename):
    """
    Main visualization entry point:
    1. Evaluates all variants in the test dataset.
    2. Dynamically isolates rare variants (AF <= 0.01) and evaluates them separately.
    3. Outputs all visualizations in PDF and SVG formats.
    """
    print('---' + time.asctime(time.localtime(time.time())) + '--- Starting visualization pipeline.\n')
    
    # Ensure test dataframe is not modified in-place
    test_df = test.copy()
    
    # 1. Evaluate entire test set
    print("Evaluating full dataset...")
    run_evaluation_pipeline(test_df, filename)
    
    # 2. Evaluate rare variants (Allele Frequency <= 0.01)
    if 'AF' in test_df.columns:
        print("Isolating and evaluating rare variants (AF <= 0.01)...")
        # Handle cases where AF contains string values
        test_df['AF_float'] = pd.to_numeric(test_df['AF'].replace('.', np.nan).replace('-', np.nan), errors='coerce')
        rare_df = test_df[test_df['AF_float'] <= 0.01].copy()
        
        if not rare_df.empty:
            run_evaluation_pipeline(rare_df, f"{filename}_rare")
            print(f"Rare variants isolated: {len(rare_df)} variants evaluated.")
        else:
            print("No variants matching AF <= 0.01 found. Skipping rare variant plots.")
    else:
        print("Warning: 'AF' column not found in dataset. Skipping rare variant evaluation.")
        
    print('---' + time.asctime(time.localtime(time.time())) + f'--- Visualizations saved in SVG & PDF format at: {save_dir}\n')
