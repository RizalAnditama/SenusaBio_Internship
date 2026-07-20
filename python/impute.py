
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(ROOT, "data", "temp")

def impute_file(input_file: str) -> str:
    print("[REPRO INFO] Membaca data temp")
    data = pd.read_csv(input_file, low_memory=False)
    
    drop_cols = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'Func.refGene', 'gene', 'GeneDetail.refGene', 'AAChange.refGene', 'CLASS']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    feature_data = data[feature_cols].copy()

    categorical_bounds = {}

    print("[REPRO INFO] Memulai Label Encoding")
    for col in feature_data.columns:
        feature_data[col] = feature_data[col].replace(['.', 'NaN', 'nan', ''], np.nan)
        
        if feature_data[col].dtype == 'object':
            non_nas = feature_data[col].dropna()
            if not non_nas.empty:
                labels, uniques = pd.factorize(non_nas)
                feature_data.loc[non_nas.index, col] = labels
                categorical_bounds[col] = len(uniques) - 1
            else:
                categorical_bounds[col] = 0

    feature_data = feature_data.apply(pd.to_numeric, errors="coerce")

    dead_cols = feature_data.columns[feature_data.isna().all()]
    if len(dead_cols) > 0:
        feature_data[dead_cols] = feature_data[dead_cols].fillna(0)
        for col in dead_cols:
            categorical_bounds[col] = 0

    print("[REPRO INFO] Memulai IterativeImputer")
    imputer = IterativeImputer(random_state=0, sample_posterior=False, max_iter=10)
    filled_array = imputer.fit_transform(feature_data)

    filled_df = pd.DataFrame(filled_array, columns=feature_data.columns)

    for col, max_label in categorical_bounds.items():
        filled_df[col] = filled_df[col].clip(lower=0, upper=max_label)
        filled_df[col] = filled_df[col].round()
        filled_df[col] = filled_df[col].astype(int)

    filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(TEMP_DIR, f"{filename}_bpca.csv")
    
    filled_df.to_csv(output_file, index=False, header=True)
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    args = parser.parse_args()
    impute_file(args.input_file)
