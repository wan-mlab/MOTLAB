#!/usr/bin/env python
# coding: utf-8

"""
MOTLAB execution script for breast-cancer clinical outcome prediction modeling

This script implements the analysis pipeline used in the updated manuscript:
  1. Align mRNA, miRNA, and DNA methylation data at the TCGA patient level.
  2. Apply sample-wise normalization within each omics layer.
  3. Construct a weighted three-omics feature representation by concatenation.
  4. Define prognosis labels from the TCGA Clinical Data Resource.
  5. Use European american patients as the source domain and African american
     patients as the target domain.
  6. Within each target domain CV fold, construct leakage-safe patient-to-anchor
     Pearson-correlation features using source patients plus few-shot target-train
     patients as anchors.
  7. Select features using the source domain training labels only.
  8. Compare transfer learning (TL) with TL plus target domain
     synthetic data augmentation (TLDA).
  9. Save seed-level metrics and patient-level/subgroup reviewer outputs.

Required project-side module:
  Initialization.py

Expected default input files under --dat_path:
  mRNA_filtered.csv
  miRNA_filtered.csv
  methyl_filtered.csv
  Genetic_Ancestry.xlsx
  TCGA-CDR-SupplementalTableS1.xlsx
  BRCA_subtype_stage.csv
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler, SMOTE
from keras import Input, Model
from keras.layers import Activation, Dense, Dropout, Lambda
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=("This script implements a fairness-aware deep learning model designed to mitigate\n"
                     "racial disparities in breast cancer clinical outcome prediction.\n"
                     "It integrates transfer learning and data augmentation with weighted multi-omics representation\n"
                     "to improve predictive performance in underrepresented populations. The model is\n"
                     "pretrained on European American data and fine-tuned on African American data using\n"
                     "domain adaptation called Classification and Contrastive Semantic Alignment (CCSA).\n"
                    ),
        epilog=(
            "Example:\n"
            "  python run_MOTLAB.py \\\n\n"
            "      --dat_path /home/user/MOTLAB/data \\\n\n"
            "      --out_path /home/user/MOTLAB/output \\\n\n"
            "      --cpoint PFI \\\n\n"
            "      --year 3\n\n"
            "      --comb_start 0\n\n"
            "      --comb_end 36\n\n"
            "Run only the first six weight combinations: "
            "  python run_MOTLAB.py --dat_path DATA --out_path OUT --cpoint PFI "
            "--year 3 --comb_start 0 --comb_end 6"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=88),
    )

    parser.add_argument("--dat_path", type=str, required=True,
                        help="Directory containing omics, ancestry, clinical, and metadata files.")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Directory in which model outputs will be written.")
    parser.add_argument("--cpoint", type=str, required=True,
                        choices=["OS", "DSS", "DFI", "PFI"],
                        help="Clinical endpoint used for horizon-specific prognosis labeling.")
    parser.add_argument("--year", type=int, required=True,
                        help="Prediction horizon in years (e.g., 2, 3, 4, or 5).")

    # Input file names
    parser.add_argument("--mrna_file", default="mRNA_filtered.csv")
    parser.add_argument("--mirna_file", default="miRNA_filtered.csv")
    parser.add_argument("--methyl_file", default="methyl_filtered.csv")
    parser.add_argument("--ancestry_file", default="Genetic_Ancestry.xlsx")
    parser.add_argument("--cdr_file", default="TCGA-CDR-SupplementalTableS1.xlsx")
    parser.add_argument("--strata_file", default="BRCA_subtype_stage.csv")
    parser.add_argument("--init_module", default="Initialization",
                        help="Python module implementing CCSA pair creation/model training.")

    # Model / CV settings used in the updated analysis
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=35)
    parser.add_argument("--k_features", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sample_per_class", type=int, default=5)
    parser.add_argument("--repetition", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[100, 50])

    # Weighted three-omics grid
    parser.add_argument("--weight_step", type=float, default=0.1)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--comb_start", type=int, default=0,
                        help="0-based inclusive start index of generated weight combinations.")
    parser.add_argument("--comb_end", type=int, default=36,
                        help="0-based exclusive end index.")
    parser.add_argument(
        "--weights", type=float, nargs=3, default=None, metavar=("W_MRNA", "W_MIRNA", "W_METHYL"),
        help="Run one explicit weight triplet instead of the generated grid; weights must sum to 1.",
    )

    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = Path(ARGS.dat_path).expanduser().resolve()
OUT_ROOT = Path(ARGS.out_path).expanduser().resolve()
RUN_DIR = OUT_ROOT / f"{ARGS.year}yr" / ARGS.cpoint
RUN_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")

sys.path.insert(0, str(DATA_DIR))

# Configure a process-specific Theano compilation directory before importing the
# project-side CCSA implementation (the module may import Theano internally).
theano_cache = RUN_DIR / ".theano" / str(os.getpid())
theano_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("THEANO_FLAGS", f"base_compiledir={theano_cache}")

# Import the project-side CCSA implementation after --dat_path and THEANO_FLAGS are set.
Initialization = __import__(ARGS.init_module)

print("Data path:", DATA_DIR)
print("Output path:", RUN_DIR)
print("Clinical endpoint:", ARGS.cpoint)
print("Prediction horizon:", ARGS.year)


def configure_gpu():
    devices = tf.config.experimental.list_physical_devices("GPU")
    if devices:
        print("GPU is detected.")
        tf.config.experimental.set_memory_growth(devices[0], True)
        print("GPU dynamic memory allocation is activated.")
    else:
        print("GPU is not available; running on CPU.")


configure_gpu()


# -----------------------------------------------------------------------------
# Reproducibility / utility helpers
# -----------------------------------------------------------------------------

def make_seed_fold(seed_base: int, fold_id: int) -> int:
    return int(seed_base) * 1000 + int(fold_id)


def set_all_seeds(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_pairs_dir(run_tag: str, seed_base: int, fold_id: int) -> Path:
    path = RUN_DIR / "CCSA_pairs" / str(run_tag) / f"seed{seed_base}_fold{fold_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def concordance_index_simple(T, risk, E):
    """Harrell-style C-index; higher risk means earlier/worse event."""
    T = np.asarray(T, dtype=float).reshape(-1)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    E = np.asarray(E, dtype=int).reshape(-1)

    concordant = 0.0
    ties = 0.0
    comparable = 0.0

    for i in range(len(T)):
        if E[i] != 1:
            continue
        for j in range(len(T)):
            if T[i] < T[j]:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    ties += 1.0

    if comparable == 0:
        return np.nan
    return (concordant + 0.5 * ties) / comparable


# -----------------------------------------------------------------------------
# Clinical / ancestry data
# -----------------------------------------------------------------------------

def get_race(cancer_type="BRCA"):
    path = DATA_DIR / ARGS.ancestry_file
    df_race = pd.read_excel(
        path,
        sheet_name=cancer_type,
        usecols="A,E",
        index_col="Patient_ID",
        keep_default_na=False,
    )
    df_race = df_race[df_race["EIGENSTRAT"].isin(["EA", "AA", "EAA", "NA", "OA"])]
    df_race["race"] = df_race["EIGENSTRAT"]
    df_race.loc[df_race["EIGENSTRAT"] == "EA", "race"] = "WHITE"
    df_race.loc[df_race["EIGENSTRAT"] == "AA", "race"] = "BLACK"
    df_race.loc[df_race["EIGENSTRAT"] == "EAA", "race"] = "ASIAN"
    df_race.loc[df_race["EIGENSTRAT"] == "NA", "race"] = "NAT_A"
    df_race.loc[df_race["EIGENSTRAT"] == "OA", "race"] = "OTHER"
    return df_race.drop(columns=["EIGENSTRAT"])


def get_CT(target):
    path = DATA_DIR / ARGS.cdr_file

    cols = "B,Z,AA"       # OS
    if target == "DSS":
        cols = "B,AB,AC"
    elif target == "DFI":
        cols = "B,AD,AE"
    elif target == "PFI":
        cols = "B,AF,AG"

    df = pd.read_excel(path, "TCGA-CDR", usecols=cols, index_col="bcr_patient_barcode")
    df.columns = ["E", "T"]
    df = df[df["E"].isin([0, 1])].dropna()
    df["C"] = 1 - df["E"]
    return df.drop(columns=["E"])


def add_race_CT(df, target, meta_df=None, groups=("WHITE", "BLACK")):
    """Attach ancestry, survival outcome, and optional subgroup metadata."""
    df = df.copy()
    df.index = df.index.astype(str)

    df_race = get_race("BRCA")
    df_race.index = df_race.index.astype(str)
    df_race = df_race[df_race["race"].isin(groups)]

    df_CT = get_CT(target)
    df_CT.index = df_CT.index.astype(str)

    df = df.join(df_race, how="inner")
    df = df.dropna(axis="columns")
    df = df.join(df_CT, how="inner")

    meta_cols = []
    if meta_df is not None:
        meta_df = meta_df.copy()
        meta_df.index = meta_df.index.astype(str)
        df = df.join(meta_df, how="inner")
        meta_cols = list(meta_df.columns)

    C = df["C"].to_numpy(dtype=np.int32)
    T = df["T"].to_numpy(dtype=np.float32)
    E = (1 - C).astype(np.int32)
    R = df["race"].astype(str).to_numpy()

    model_cols = [c for c in df.columns if c not in ["C", "race", "T"] + meta_cols]
    X = df[model_cols].to_numpy(dtype=np.float32)

    data = {
        "X": X,
        "T": T,
        "C": C,
        "E": E,
        "R": R,
        "Samples": df.index.to_numpy(dtype=str),
        "FeatureName": list(model_cols),
    }
    for col in meta_cols:
        data[col] = df[col].to_numpy()

    print("After ancestry + outcome + metadata join:", len(data["Samples"]), "patients")
    return data


def get_n_years(dataset, years):
    """
    Construct horizon-specific binary prognosis labels.

    Y=0: event occurred at or before the horizon.
    Y=1: event-free beyond the horizon, including censoring after the horizon.
    Patients censored at or before the horizon are removed.
    """
    T = np.asarray(dataset["T"])
    C = np.asarray(dataset["C"])
    R = np.asarray(dataset["R"])
    samples = np.asarray(dataset["Samples"]).astype(str)

    time_p = 365 * int(years)
    keep = ~((T <= time_p) & (C == 1))

    T_keep = T[keep]
    C_keep = C[keep]
    R_keep = R[keep]
    sample_keep = samples[keep]

    Y = np.ones_like(C_keep, dtype=np.int32)
    Y[(T_keep <= time_p) & (C_keep == 0)] = 0

    y_strat = np.array([str(int(y)) + str(r) for y, r in zip(Y, R_keep)], dtype=object)
    return Y, R_keep, y_strat, sample_keep


# -----------------------------------------------------------------------------
# Multi-omics representation
# -----------------------------------------------------------------------------

def load_omics_table(filename):
    """Load the manuscript-format omics CSV and return patient x molecular-feature data."""
    path = DATA_DIR / filename
    raw = pd.read_csv(path)
    raw = raw.T
    raw.columns = raw.iloc[0]
    raw = raw.iloc[1:].copy()

    raw.index = [str(x)[:12] for x in raw.index]
    raw = raw.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")
    raw.index = raw.index.astype(str)
    return raw


def build_weighted_three_omics_matrix(df_a, df_b, df_c, w1, w2, w3, eps=1e-8):
    """
    Build weighted three-omics sample x feature matrix.

    Important: preprocessing here is sample-wise only. No cohort-level scaling or
    imputation is performed before CV.
    """
    Xa, Xb, Xc = df_a.copy(), df_b.copy(), df_c.copy()
    Xa.index = Xa.index.astype(str)
    Xb.index = Xb.index.astype(str)
    Xc.index = Xc.index.astype(str)

    common = np.array(
        [sid for sid in Xa.index if sid in set(Xb.index) and sid in set(Xc.index)],
        dtype=str,
    )
    if len(common) == 0:
        raise ValueError("No common patient IDs across the three omics datasets.")

    Xa = Xa.loc[common].apply(pd.to_numeric, errors="coerce")
    Xb = Xb.loc[common].apply(pd.to_numeric, errors="coerce")
    Xc = Xc.loc[common].apply(pd.to_numeric, errors="coerce")

    def rowwise_impute_and_zscore(df):
        arr = df.to_numpy(dtype=np.float32)
        row_mean = np.nanmean(arr, axis=1, keepdims=True)
        row_mean = np.where(np.isnan(row_mean), 0.0, row_mean)
        missing = np.where(np.isnan(arr))
        if len(missing[0]) > 0:
            arr[missing] = row_mean[missing[0], 0]
        arr = arr - arr.mean(axis=1, keepdims=True)
        arr = arr / (arr.std(axis=1, ddof=1, keepdims=True) + eps)
        return arr.astype(np.float32)

    Xa = rowwise_impute_and_zscore(Xa)
    Xb = rowwise_impute_and_zscore(Xb)
    Xc = rowwise_impute_and_zscore(Xc)

    X_comb = np.concatenate(
        [float(w1) * Xa, float(w2) * Xb, float(w3) * Xc], axis=1
    ).astype(np.float32)

    return common, X_comb


def pcc_to_anchors(X_query, X_anchor, eps=1e-8):
    """Pearson correlation of each query patient to each anchor patient."""
    Xq = X_query.astype(np.float32)
    Xa = X_anchor.astype(np.float32)

    Xq = Xq - Xq.mean(axis=1, keepdims=True)
    Xa = Xa - Xa.mean(axis=1, keepdims=True)
    Xq = Xq / (Xq.std(axis=1, ddof=1, keepdims=True) + eps)
    Xa = Xa / (Xa.std(axis=1, ddof=1, keepdims=True) + eps)

    p = Xq.shape[1]
    return ((Xq @ Xa.T) / max(p - 1, 1)).astype(np.float32)


# -----------------------------------------------------------------------------
# Prediction / metrics
# -----------------------------------------------------------------------------

def find_best_threshold_by_f1(y_true, score, average="binary"):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    score = np.asarray(score).astype(float).reshape(-1)

    if len(y_true) == 0:
        return 0.5, np.nan, np.nan

    if np.unique(y_true).size < 2:
        pred = (score >= 0.5).astype(int)
        return 0.5, f1_score(y_true, pred, average=average, zero_division=0), accuracy_score(y_true, pred)

    thresholds = np.unique(score)
    if thresholds.size > 200:
        thresholds = np.linspace(score.min(), score.max(), 200)

    best_thr, best_f1, best_acc = 0.5, -1.0, -1.0
    for thr in thresholds:
        pred = (score >= thr).astype(int)
        f1 = f1_score(y_true, pred, average=average, zero_division=0)
        acc = accuracy_score(y_true, pred)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and acc > best_acc):
            best_thr, best_f1, best_acc = float(thr), float(f1), float(acc)

    return best_thr, best_f1, best_acc


def safe_binary_metrics(y_true, y_score, y_pred):
    """Report metrics with event as the positive class."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)  # P(Y=1), non-event
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    y_true_event = (y_true == 0).astype(int)
    y_score_event = 1.0 - y_score
    y_pred_event = (y_pred == 0).astype(int)

    keep = np.isfinite(y_true_event) & np.isfinite(y_score_event) & np.isfinite(y_pred_event)
    y_true_event = y_true_event[keep]
    y_score_event = y_score_event[keep]
    y_pred_event = y_pred_event[keep]

    out = {
        "n": int(len(y_true_event)),
        "event_n": int(np.sum(y_true_event == 1)),
        "non_event_n": int(np.sum(y_true_event == 0)),
    }

    if len(y_true_event) == 0:
        for key in ["auc", "prauc", "acc", "f1", "bacc", "brier", "sens", "spec"]:
            out[key] = np.nan
        return out

    y_score_event = np.clip(y_score_event, 1e-8, 1 - 1e-8)
    if np.unique(y_true_event).size < 2:
        out["auc"] = np.nan
        out["prauc"] = np.nan
    else:
        out["auc"] = roc_auc_score(y_true_event, y_score_event)
        out["prauc"] = average_precision_score(y_true_event, y_score_event)

    out["acc"] = accuracy_score(y_true_event, y_pred_event)
    out["f1"] = f1_score(y_true_event, y_pred_event, average="binary", zero_division=0)
    out["brier"] = float(np.mean((y_true_event - y_score_event) ** 2))

    tp = int(np.sum((y_true_event == 1) & (y_pred_event == 1)))
    fn = int(np.sum((y_true_event == 1) & (y_pred_event == 0)))
    tn = int(np.sum((y_true_event == 0) & (y_pred_event == 0)))
    fp = int(np.sum((y_true_event == 0) & (y_pred_event == 1)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    out["sens"] = sens
    out["spec"] = spec
    out["bacc"] = np.nanmean([sens, spec])
    return out


def calibration_metrics(y_true, y_score, n_bins=10):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    y_true_event = (y_true == 0).astype(int)
    y_score_event = 1.0 - y_score
    keep = np.isfinite(y_true_event) & np.isfinite(y_score_event)
    y_true_event = y_true_event[keep]
    y_score_event = y_score_event[keep]

    if len(y_true_event) == 0:
        return {"ece": np.nan, "mce": np.nan, "brier": np.nan, "bin_table": pd.DataFrame()}

    y_score_event = np.clip(y_score_event, 1e-8, 1 - 1e-8)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, gaps = [], []

    for b in range(n_bins):
        left, right = edges[b], edges[b + 1]
        mask = ((y_score_event >= left) & (y_score_event < right)) if b < n_bins - 1 else ((y_score_event >= left) & (y_score_event <= right))
        if np.sum(mask) == 0:
            continue
        obs = float(np.mean(y_true_event[mask]))
        pred = float(np.mean(y_score_event[mask]))
        n_bin = int(np.sum(mask))
        gap = abs(obs - pred)
        rows.append({
            "bin_id": b + 1,
            "bin_left": left,
            "bin_right": right,
            "n": n_bin,
            "mean_pred": pred,
            "obs_rate": obs,
            "abs_gap": gap,
        })
        gaps.append((n_bin, gap))

    if gaps:
        total = float(sum(n for n, _ in gaps))
        ece = sum((n / total) * gap for n, gap in gaps)
        mce = max(gap for _, gap in gaps)
    else:
        ece, mce = np.nan, np.nan

    return {
        "ece": float(ece) if np.isfinite(ece) else np.nan,
        "mce": float(mce) if np.isfinite(mce) else np.nan,
        "brier": float(np.mean((y_true_event - y_score_event) ** 2)),
        "bin_table": pd.DataFrame(rows),
    }


def train_and_predict(
    X_train_target,
    y_train_target,
    X_train_source,
    y_train_source,
    X_val_target,
    y_val_target,
    X_test,
    y_test,
    repetition,
    sample_per_class,
    alpha,
    learning_rate,
    hidden_layers,
    dropout,
    momentum,
    batch_size,
    seed,
    pairs_dir,
    apply_domain_smote=False,
):
    set_all_seeds(seed)

    y_train_target = np.asarray(y_train_target).astype(int).reshape(-1)
    y_train_source = np.asarray(y_train_source).astype(int).reshape(-1)
    y_val_target = np.asarray(y_val_target).astype(int).reshape(-1)
    y_test = np.asarray(y_test).astype(int).reshape(-1)

    domain_adaptation_task = "WHITE_to_BLACK"
    n_features = int(X_train_source.shape[1])

    input_a = Input(shape=(n_features,))
    input_b = Input(shape=(n_features,))

    shared_model = Initialization.Create_Model(hiddenLayers=hidden_layers, dr=dropout)
    processed_a = shared_model(input_a)
    processed_b = shared_model(input_b)

    processed_a = Dropout(float(dropout))(processed_a)
    out_class = Dense(2)(processed_a)
    out_class = Activation("softmax", name="classification")(out_class)

    distance = Lambda(
        Initialization.euclidean_distance,
        output_shape=Initialization.eucl_dist_output_shape,
        name="CSA",
    )([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=[out_class, distance])
    optimizer = tf.keras.optimizers.legacy.SGD(
        learning_rate=float(learning_rate), momentum=float(momentum)
    )

    model.compile(
        loss={
            "classification": tf.keras.losses.SparseCategoricalCrossentropy(),
            "CSA": Initialization.contrastive_loss,
        },
        optimizer=optimizer,
        loss_weights={"classification": 1.0 - float(alpha), "CSA": float(alpha)},
    )

    pairs_dir = Path(pairs_dir).resolve()
    pairs_dir.mkdir(parents=True, exist_ok=True)

    Initialization.Create_Pairs(
        domain_adaptation_task=domain_adaptation_task,
        repetition=repetition,
        sample_per_class=sample_per_class,
        X_train_target=X_train_target,
        y_train_target=y_train_target,
        X_train_source=X_train_source,
        y_train_source=y_train_source,
        n_features=n_features,
        pairs_dir=str(pairs_dir),
        seed=int(seed),
        neg_pos_ratio=1.0,
        balance_target_to_source=bool(apply_domain_smote),
        balance_mode="overall",
        smote_k_max=5,
    )

    Initialization.training_the_model(
        model=model,
        domain_adaptation_task=domain_adaptation_task,
        repetition=repetition,
        sample_per_class=sample_per_class,
        batch_size=int(batch_size),
        X_val_target=X_val_target,
        y_val_target=y_val_target,
        max_epochs=100,
        patience=int(ARGS.patience),
        seed=int(seed),
        pairs_dir=str(pairs_dir),
        class_weight_mode="from_marginal",
        y_marginal=np.concatenate([y_train_source, y_train_target], axis=0),
        selection_metric="auc",
    )

    # Validation threshold: optimize F1 for the event class.
    p_val = np.asarray(
        model.predict([X_val_target, X_val_target], batch_size=int(batch_size), verbose=0)[0],
        dtype=np.float64,
    )
    val_score_event = 1.0 - p_val[:, 1]
    y_val_event = (y_val_target == 0).astype(int)
    best_thr_event, best_val_f1, best_val_acc = find_best_threshold_by_f1(
        y_val_event, val_score_event
    )

    p_test = np.asarray(
        model.predict([X_test, X_test], batch_size=int(batch_size), verbose=0)[0],
        dtype=np.float64,
    )
    test_score_non_event = p_test[:, 1]
    test_score_event = 1.0 - test_score_non_event
    pred_event = (test_score_event >= best_thr_event).astype(int)
    pred_original_label = np.where(pred_event == 1, 0, 1).astype(int)

    test_auc = (
        roc_auc_score((y_test == 0).astype(int), test_score_event)
        if np.unique(y_test).size == 2
        else np.nan
    )

    print(
        f"seed={seed} threshold_event={best_thr_event:.6f} "
        f"val_f1_event={best_val_f1:.4f} val_acc_event={best_val_acc:.4f} "
        f"test_auc={test_auc:.4f}"
    )

    return test_score_non_event.astype(float), float(test_auc), float(best_thr_event), pred_original_label


# -----------------------------------------------------------------------------
# Fold-level CCSA runner: TL and TLDA share the same split / representation logic
# -----------------------------------------------------------------------------

def run_ccsa_transfer(seed, dataset, X_m, years, augment_target=False):
    model_name = "TLDA" if augment_target else "TL"
    Y, R, _y_strat, kept_ids = get_n_years(dataset, years)

    sample_ids = np.asarray(dataset["Samples"]).astype(str)
    id2pos = {sid: i for i, sid in enumerate(sample_ids)}
    idx = np.array([id2pos[s] for s in kept_ids], dtype=int)

    Xm = X_m[idx].astype(np.float32)
    T_kept = np.asarray(dataset["T"])[idx].astype(np.float32)
    E_kept = np.asarray(dataset["E"])[idx].astype(np.int32)

    meta = {"R": np.asarray(R), "Y": np.asarray(Y).astype(np.int32)}
    for col in ["PAM50", "Stage", "Tumor", "Node", "Meta"]:
        if col in dataset:
            meta[col] = np.asarray(dataset[col])[idx]
    df_meta = pd.DataFrame(meta, index=np.asarray(kept_ids).astype(str))

    source_idx = np.where(df_meta["R"].values == "WHITE")[0]
    target_idx = np.where(df_meta["R"].values == "BLACK")[0]
    y_source = df_meta["Y"].values[source_idx].astype(np.int32)
    y_target_all = df_meta["Y"].values[target_idx].astype(np.int32)

    empty_result = {
        "folds": ARGS.folds,
        f"{model_name}_auc": np.nan,
        f"{model_name}_acc": np.nan,
        f"{model_name}_f1": np.nan,
        f"{model_name}_prauc": np.nan,
        f"{model_name}_bacc": np.nan,
        f"{model_name}_brier": np.nan,
        f"{model_name}_sens": np.nan,
        f"{model_name}_spec": np.nan,
        f"{model_name}_cindex": np.nan,
    }

    if len(target_idx) < ARGS.folds or np.unique(y_target_all).size < 2:
        return pd.DataFrame(empty_result, index=[seed]), pd.DataFrame()

    df_score = pd.DataFrame()
    kf = StratifiedKFold(n_splits=ARGS.folds, shuffle=True, random_state=seed)

    for fold_id, (train_rel, test_rel) in enumerate(kf.split(np.zeros(len(target_idx)), y_target_all)):
        seed_fold = make_seed_fold(seed, fold_id)

        target_train_full = target_idx[train_rel]
        target_test = target_idx[test_rel]
        y_target_train_full = Y[target_train_full].astype(np.int32)
        y_target_test = Y[target_test].astype(np.int32)

        # Few-shot target support set: equal number from each class.
        idx0 = np.where(y_target_train_full == 0)[0]
        idx1 = np.where(y_target_train_full == 1)[0]
        rng = np.random.RandomState(seed_fold)
        rng.shuffle(idx0)
        rng.shuffle(idx1)

        take = min(ARGS.sample_per_class, len(idx0), len(idx1))
        if take == 0:
            continue

        support_rel = np.concatenate([idx0[:take], idx1[:take]])
        target_train_small = target_train_full[support_rel]

        remain = np.ones(len(target_train_full), dtype=bool)
        remain[support_rel] = False
        target_val = target_train_full[remain]

        if len(target_val) == 0 or np.unique(Y[target_val]).size < 2:
            continue

        y_target_small = Y[target_train_small].astype(np.int32)
        y_target_val = Y[target_val].astype(np.int32)

        # Leakage-safe anchor set: source + target support only.
        anchor_idx = np.concatenate([source_idx, target_train_small])
        if set(target_val).intersection(set(anchor_idx)):
            raise ValueError("LEAKAGE: target validation samples are present in anchors.")
        if set(target_test).intersection(set(anchor_idx)):
            raise ValueError("LEAKAGE: target test samples are present in anchors.")

        F_source = pcc_to_anchors(Xm[source_idx], Xm[anchor_idx])
        F_target_raw = pcc_to_anchors(Xm[target_train_small], Xm[anchor_idx])
        F_val = pcc_to_anchors(Xm[target_val], Xm[anchor_idx])
        F_test = pcc_to_anchors(Xm[target_test], Xm[anchor_idx])

        # Training-only column imputation.
        col_mean = np.nanmean(np.vstack([F_source, F_target_raw]), axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)

        def fill_nan(arr):
            arr = arr.copy()
            missing = np.where(np.isnan(arr))
            if len(missing[0]) > 0:
                arr[missing] = col_mean[missing[1]]
            return arr.astype(np.float32)

        F_source, F_target_raw, F_val, F_test = map(fill_nan, [F_source, F_target_raw, F_val, F_test])

        # Source-only supervised feature selection; then apply the fitted selector everywhere.
        k_eff = min(int(ARGS.k_features), F_source.shape[1])
        if 1 <= k_eff < F_source.shape[1]:
            selector = SelectKBest(f_classif, k=k_eff)
            selector.fit(F_source, y_source)
            F_source = selector.transform(F_source)
            F_target_raw = selector.transform(F_target_raw)
            F_val = selector.transform(F_val)
            F_test = selector.transform(F_test)

        X_target_train = F_target_raw
        y_target_train = y_target_small

        if augment_target:
            n_src = len(y_source)
            cnt0 = int(np.sum(y_target_small == 0))
            cnt1 = int(np.sum(y_target_small == 1))
            if cnt0 == 0 or cnt1 == 0:
                continue

            p1 = cnt1 / (cnt0 + cnt1)
            desired1 = int(round(p1 * n_src))
            desired0 = n_src - desired1
            desired0 = max(desired0, cnt0)
            desired1 = max(desired1, cnt1)

            strategy = {}
            if desired0 > cnt0:
                strategy[0] = desired0
            if desired1 > cnt1:
                strategy[1] = desired1

            smote_log = {
                "seed": int(seed),
                "fold_id": int(fold_id),
                "seed_fold": int(seed_fold),
                "source_n": int(n_src),
                "target_class0_before": cnt0,
                "target_class1_before": cnt1,
                "sampling_strategy": {str(k): int(v) for k, v in strategy.items()},
                "target_n_before": int(len(y_target_small)),
                "method": None,
                "k_neighbors": None,
            }

            if len(y_target_small) < n_src and min(cnt0, cnt1) >= 2:
                k_sm = min(5, min(cnt0, cnt1) - 1)
                sampler = SMOTE(
                    sampling_strategy=strategy,
                    random_state=seed + fold_id,
                    k_neighbors=k_sm,
                )
                X_target_train, y_target_train = sampler.fit_resample(F_target_raw, y_target_small)
                smote_log["method"] = "SMOTE"
                smote_log["k_neighbors"] = int(k_sm)
            elif len(y_target_small) < n_src:
                sampler = RandomOverSampler(
                    sampling_strategy=strategy,
                    random_state=seed + fold_id,
                )
                X_target_train, y_target_train = sampler.fit_resample(F_target_raw, y_target_small)
                smote_log["method"] = "ROS"
            else:
                smote_log["method"] = "NONE"

            smote_log["target_n_after"] = int(len(y_target_train))
            smote_log["class_counts_after"] = np.bincount(y_target_train.astype(int), minlength=2).tolist()
            smote_log["size_matched_to_source"] = bool(len(y_target_train) == n_src)

            if len(y_target_small) < n_src and len(y_target_train) != n_src:
                raise ValueError(
                    f"Target/source size mismatch after resampling: target={len(y_target_train)}, source={n_src}"
                )

            pairs_dir = make_pairs_dir(model_name, seed, fold_id)
            with open(pairs_dir / "smote_log.json", "w") as f:
                json.dump(smote_log, f, indent=2)
        else:
            pairs_dir = make_pairs_dir(model_name, seed, fold_id)

        score_non_event, _auc, _thr, pred = train_and_predict(
            X_train_target=X_target_train,
            y_train_target=y_target_train,
            X_train_source=F_source,
            y_train_source=y_source,
            X_val_target=F_val,
            y_val_target=y_target_val,
            X_test=F_test,
            y_test=y_target_test,
            repetition=ARGS.repetition,
            sample_per_class=ARGS.sample_per_class,
            alpha=ARGS.alpha,
            learning_rate=ARGS.learning_rate,
            hidden_layers=ARGS.hidden_layers,
            dropout=ARGS.dropout,
            momentum=0.9,
            batch_size=ARGS.batch_size,
            seed=seed_fold,
            pairs_dir=pairs_dir,
            apply_domain_smote=augment_target,
        )

        test_ids = df_meta.index.values[target_test]
        temp = {
            "scr": np.asarray(score_non_event).reshape(-1),
            "Y": np.asarray(y_target_test).reshape(-1),
            "pred": np.asarray(pred).reshape(-1),
            "T": T_kept[target_test],
            "E": E_kept[target_test],
            "risk": 1.0 - np.asarray(score_non_event).reshape(-1),
            "seed": seed,
            "fold_id": fold_id,
            "model": model_name,
        }
        for col in ["PAM50", "Stage", "Tumor", "Node", "Meta"]:
            if col in df_meta.columns:
                temp[col] = df_meta.loc[test_ids, col].values

        df_score = pd.concat([df_score, pd.DataFrame(temp, index=test_ids)], axis=0)

    if df_score.empty:
        return pd.DataFrame(empty_result, index=[seed]), pd.DataFrame()

    for col in ["Y", "scr", "pred", "T", "E", "risk"]:
        df_score[col] = pd.to_numeric(df_score[col], errors="coerce")
    df_score = df_score.dropna(subset=["Y", "scr", "pred", "T", "E", "risk"])
    df_score["Y"] = df_score["Y"].astype(int)
    df_score["pred"] = df_score["pred"].astype(int)
    df_score["scr"] = df_score["scr"].astype(float)

    m = safe_binary_metrics(df_score["Y"], df_score["scr"], df_score["pred"])
    cindex = concordance_index_simple(df_score["T"], df_score["risk"], df_score["E"])

    result = {
        "folds": ARGS.folds,
        f"{model_name}_auc": m["auc"],
        f"{model_name}_acc": m["acc"],
        f"{model_name}_f1": m["f1"],
        f"{model_name}_prauc": m["prauc"],
        f"{model_name}_bacc": m["bacc"],
        f"{model_name}_brier": m["brier"],
        f"{model_name}_sens": m["sens"],
        f"{model_name}_spec": m["spec"],
        f"{model_name}_cindex": cindex,
    }
    return pd.DataFrame(result, index=[seed]), df_score


# -----------------------------------------------------------------------------
# Reviewer / subgroup outputs
# -----------------------------------------------------------------------------

def subgroup_performance_table(detail_df, subgroup_cols=None):
    subgroup_cols = subgroup_cols or ["PAM50", "Stage", "Tumor", "Node", "Meta"]
    rows = []

    for (model_name, seed), g in detail_df.groupby(["model", "seed"]):
        m = safe_binary_metrics(g["Y"], g["scr"], g["pred"])
        rows.append({
            "model": model_name,
            "seed": seed,
            "subgroup_name": "Overall",
            "subgroup_level": "All",
            **m,
            "cindex": concordance_index_simple(g["T"], g["risk"], g["E"]),
        })

    for sg_col in [c for c in subgroup_cols if c in detail_df.columns]:
        tmp = detail_df[detail_df[sg_col].notna()].copy()
        for (model_name, seed, level), g in tmp.groupby(["model", "seed", sg_col]):
            m = safe_binary_metrics(g["Y"], g["scr"], g["pred"])
            rows.append({
                "model": model_name,
                "seed": seed,
                "subgroup_name": sg_col,
                "subgroup_level": str(level),
                **m,
                "cindex": concordance_index_simple(g["T"], g["risk"], g["E"]),
            })

    return pd.DataFrame(rows)


def subgroup_calibration_table(detail_df, subgroup_cols=None, n_bins=10):
    subgroup_cols = subgroup_cols or ["PAM50", "Stage", "Tumor", "Node", "Meta"]
    rows, bins = [], []

    def add_group(model_name, seed, subgroup_name, subgroup_level, g):
        cal = calibration_metrics(g["Y"], g["scr"], n_bins=n_bins)
        rows.append({
            "model": model_name,
            "seed": seed,
            "subgroup_name": subgroup_name,
            "subgroup_level": subgroup_level,
            "n": len(g),
            "event_n": int(np.sum(g["Y"].values == 0)),
            "non_event_n": int(np.sum(g["Y"].values == 1)),
            "ece": cal["ece"],
            "mce": cal["mce"],
            "brier": cal["brier"],
        })
        if not cal["bin_table"].empty:
            bt = cal["bin_table"].copy()
            bt["model"] = model_name
            bt["seed"] = seed
            bt["subgroup_name"] = subgroup_name
            bt["subgroup_level"] = subgroup_level
            bins.append(bt)

    for (model_name, seed), g in detail_df.groupby(["model", "seed"]):
        add_group(model_name, seed, "Overall", "All", g)

    for sg_col in [c for c in subgroup_cols if c in detail_df.columns]:
        tmp = detail_df[detail_df[sg_col].notna()].copy()
        for (model_name, seed, level), g in tmp.groupby(["model", "seed", sg_col]):
            add_group(model_name, seed, sg_col, str(level), g)

    return pd.DataFrame(rows), (pd.concat(bins, ignore_index=True) if bins else pd.DataFrame())


def summarize_table(df, metric_cols):
    grouped = df.groupby(["model", "subgroup_name", "subgroup_level"], dropna=False)
    means = grouped[metric_cols].mean().reset_index().rename(columns={c: f"{c}_mean" for c in metric_cols})
    sds = grouped[metric_cols].std().reset_index().rename(columns={c: f"{c}_sd" for c in metric_cols})
    return means.merge(sds, on=["model", "subgroup_name", "subgroup_level"], how="left")


def compare_tl_tlda(summary_df):
    tl = summary_df[summary_df["model"] == "TL"].drop(columns=["model"])
    tlda = summary_df[summary_df["model"] == "TLDA"].drop(columns=["model"])
    merged = tl.merge(
        tlda,
        on=["subgroup_name", "subgroup_level"],
        how="outer",
        suffixes=("_TL", "_TLDA"),
    )
    for metric in ["auc", "prauc", "acc", "f1", "bacc", "brier", "sens", "spec", "cindex"]:
        a, b = f"{metric}_mean_TL", f"{metric}_mean_TLDA"
        if a in merged.columns and b in merged.columns:
            merged[f"{metric}_mean_diff_TLDA_minus_TL"] = merged[b] - merged[a]
    return merged


def save_reviewer_outputs(detail_df, base_prefix):
    perf_seed = subgroup_performance_table(detail_df)
    perf_summary = summarize_table(
        perf_seed,
        ["n", "event_n", "non_event_n", "auc", "prauc", "acc", "f1", "bacc", "brier", "sens", "spec", "cindex"],
    )
    compare = compare_tl_tlda(perf_summary)

    cal_seed, cal_bins = subgroup_calibration_table(detail_df, n_bins=10)
    cal_summary = summarize_table(cal_seed, ["n", "event_n", "non_event_n", "brier", "ece", "mce"])

    out_file = RUN_DIR / f"{base_prefix}_reviewer_outputs.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="patient_level_predictions", index=True)
        perf_seed.to_excel(writer, sheet_name="subgroup_perf_seedwise", index=False)
        perf_summary.to_excel(writer, sheet_name="subgroup_perf_summary", index=False)
        compare.to_excel(writer, sheet_name="TL_vs_TLDA_compare", index=False)
        cal_seed.to_excel(writer, sheet_name="calibration_seedwise", index=False)
        cal_summary.to_excel(writer, sheet_name="calibration_summary", index=False)
        cal_bins.to_excel(writer, sheet_name="calibration_bins", index=False)
    return out_file


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_combination(meta_data, X_m, weights):
    w1, w2, w3 = weights
    result_list = []
    detail_list = []
    score_dict = {}

    for i in range(ARGS.trials):
        seed = ARGS.seed_start + i

        start = time.time()
        df_tl, detail_tl = run_ccsa_transfer(seed, meta_data, X_m, ARGS.year, augment_target=False)
        print(f"Trial {i} / seed {seed} TL: {(time.time() - start) / 60:.2f} min")
        print(df_tl.to_string(index=False))

        start = time.time()
        df_tlda, detail_tlda = run_ccsa_transfer(seed, meta_data, X_m, ARGS.year, augment_target=True)
        print(f"Trial {i} / seed {seed} TLDA: {(time.time() - start) / 60:.2f} min")
        print(df_tlda.to_string(index=False))

        result_list.append(pd.concat([df_tl, df_tlda], axis=1))
        if not detail_tl.empty:
            detail_list.append(detail_tl.copy())
        if not detail_tlda.empty:
            detail_list.append(detail_tlda.copy())
        score_dict[i] = {"TL": detail_tl.copy(), "TLDA": detail_tlda.copy()}

    res = pd.concat(result_list, axis=0)
    comb_tag = f"C1_{w1}_C2_{w2}_C3_{w3}"
    base_prefix = (
        f"BRCA-AA-EA-TRIPLEOMICS-{ARGS.cpoint}-{ARGS.year}YR_"
        f"MJ_K{ARGS.k_features}_{comb_tag}_{ARGS.learning_rate}_{ARGS.dropout}"
    )

    res.to_excel(RUN_DIR / f"{base_prefix}.xlsx")
    summary = pd.DataFrame({
        "Column": res.columns,
        "Mean": res.mean(numeric_only=True),
        "Standard Deviation": res.std(numeric_only=True),
    })
    summary.to_excel(RUN_DIR / f"summary-{base_prefix}.xlsx")

    with open(RUN_DIR / f"{base_prefix}.pkl", "wb") as f:
        pickle.dump(score_dict, f)

    if detail_list:
        detail_all = pd.concat(detail_list, axis=0)
        reviewer_file = save_reviewer_outputs(detail_all, base_prefix)
        print("Reviewer outputs saved:", reviewer_file)

    return res


def generate_triple_weight_combinations(step=0.1, min_weight=0.1):
    vals = np.round(np.arange(min_weight, 1.0 + step, step), 2)
    combos = []
    for w1 in vals:
        for w2 in vals:
            w3 = np.round(1.0 - w1 - w2, 2)
            if w3 < min_weight or w3 > 1:
                continue
            if np.isclose(w1 + w2 + w3, 1.0):
                combos.append((float(w1), float(w2), float(w3)))
    return combos


def validate_inputs():
    required = [
        ARGS.mrna_file,
        ARGS.mirna_file,
        ARGS.methyl_file,
        ARGS.ancestry_file,
        ARGS.cdr_file,
        ARGS.strata_file,
        f"{ARGS.init_module}.py",
    ]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(missing))


def save_run_config(selected_combinations):
    config = vars(ARGS).copy()
    config["data_dir_resolved"] = str(DATA_DIR)
    config["run_dir_resolved"] = str(RUN_DIR)
    config["selected_weight_combinations"] = selected_combinations
    with open(RUN_DIR / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    validate_inputs()

    mrna = load_omics_table(ARGS.mrna_file)
    mirna = load_omics_table(ARGS.mirna_file)
    methyl = load_omics_table(ARGS.methyl_file)

    common_ids = np.array(sorted(set(mrna.index) & set(mirna.index) & set(methyl.index)), dtype=str)
    if len(common_ids) == 0:
        raise ValueError("No common patients across mRNA, miRNA, and methylation files.")

    mrna = mrna.loc[common_ids].copy()
    mirna = mirna.loc[common_ids].copy()
    methyl = methyl.loc[common_ids].copy()

    strata = pd.read_csv(DATA_DIR / ARGS.strata_file, index_col=0)
    strata.index = strata.index.astype(str)
    strata = strata.loc[strata.index.intersection(common_ids)].copy()

    print("Common aligned patient count across three omics:", len(common_ids))

    if ARGS.weights is not None:
        if not np.isclose(sum(ARGS.weights), 1.0):
            raise ValueError("--weights must sum to 1.0")
        if min(ARGS.weights) < 0:
            raise ValueError("--weights must be non-negative")
        combinations = [tuple(float(x) for x in ARGS.weights)]
    else:
        combinations = generate_triple_weight_combinations(ARGS.weight_step, ARGS.min_weight)
        combinations = combinations[ARGS.comb_start:ARGS.comb_end]

    if not combinations:
        raise ValueError("No weight combinations selected.")

    save_run_config(combinations)
    print("Selected weight combinations:", combinations)

    for w1, w2, w3 in combinations:
        print(f"\n***** Start weights: ({w1}, {w2}, {w3}) *****")

        aligned_ids, X_raw = build_weighted_three_omics_matrix(
            mrna, mirna, methyl, w1, w2, w3
        )

        # The manuscript analysis used subgroup metadata during cohort assembly.
        strata_sub = strata.loc[strata.index.intersection(aligned_ids)].copy()
        dummy = pd.DataFrame(
            np.zeros((len(aligned_ids), 1), dtype=np.float32),
            index=aligned_ids,
            columns=["dummy"],
        )
        meta_data = add_race_CT(
            df=dummy,
            target=ARGS.cpoint,
            meta_df=strata_sub,
            groups=("WHITE", "BLACK"),
        )

        ordered_ids = np.asarray(meta_data["Samples"]).astype(str)
        id2pos = {sid: i for i, sid in enumerate(aligned_ids)}
        ordered_pos = np.array([id2pos[sid] for sid in ordered_ids], dtype=int)
        X_m = X_raw[ordered_pos].astype(np.float32)

        print("Final modeling cohort:", len(ordered_ids))
        print("Weighted multi-omics matrix shape:", X_m.shape)

        run_combination(meta_data, X_m, (w1, w2, w3))
        print(f"***** Finish weights: ({w1}, {w2}, {w3}) *****")

    print("================= All trainings are over. =================")


if __name__ == "__main__":
    main()
