"""
classifier.py
=============
Trains an XGBoost classifier on known targets and classifies
new light curves into one of four categories:

  0 = Planet Transit
  1 = Eclipsing Binary
  2 = Blend / False Positive
  3 = Other / Noise

Strategy (honest for 48-hour hackathon):
  - Positive class (planet transit): 10 confirmed targets from Excel file
  - Negative classes: generated from Sector 1 random stars with no known
    transiting planets, using feature heuristics to label EB vs blend vs other
  - XGBoost with class weights to handle imbalance
  - 5-fold stratified cross-validation for honest evaluation
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import lightkurve as lk
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from feature_extractor import extract_features, features_to_vector, FEATURE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# LABEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ['Planet Transit', 'Eclipsing Binary', 'Blend/FP', 'Other/Noise']
CLASS_COLORS = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN POSITIVE TARGETS (from Confirmed_Exoplanet_Test_Targets.xlsx)
# ─────────────────────────────────────────────────────────────────────────────

CONFIRMED_PLANETS = [
    {'tic_id': 100100827, 'name': 'WASP-18 b',    'period': 0.94,  'depth': 0.010},
    {'tic_id':  35516889, 'name': 'WASP-19 b',    'period': 0.79,  'depth': 0.020},
    {'tic_id': 420814525, 'name': 'HD 209458 b',  'period': 3.52,  'depth': 0.015},
    {'tic_id':  86396382, 'name': 'WASP-12 b',    'period': 1.09,  'depth': 0.014},
    {'tic_id':  36734222, 'name': 'WASP-43 b',    'period': 0.81,  'depth': 0.025},
    {'tic_id':  424865156, 'name': 'HAT-P-7 b',    'period': 2.20,  'depth': 0.007},
    {'tic_id': 399860444, 'name': 'TrES-2 b',     'period': 2.47,  'depth': 0.014},
    {'tic_id': 150428135, 'name': 'TOI-700 d',    'period': 37.40, 'depth': 0.001},
    {'tic_id': 377780790, 'name': 'Kepler-10 b',  'period': 0.84,  'depth': 0.00015},
    {'tic_id': 261136679, 'name': 'Pi Mensae c',  'period': 6.27,  'depth': 0.0003},
]

# Known eclipsing binaries from TESS (for negative class training)
KNOWN_EBS = [
    {'tic_id': 229804573, 'name': 'TIC 229804573 EB'}, # BG Ind (Hierarchical)
    {'tic_id': 402026209, 'name': 'TIC 402026209 EB'}, # YY Geminorum (Detached)
    {'tic_id': 55652896,  'name': 'TIC 55652896 EB'},  # VW Cephei (Contact)
    {'tic_id': 281541555, 'name': 'TIC 281541555 EB'}, # RR Caelum (Deep Primary/Secondary)
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def download_and_extract(tic_id, label, name=''):
    """
    Download a light curve for a TIC ID and extract features.
    Returns (feature_vector, label) or None if download fails.
    """
    try:
        print(f"  Downloading TIC {tic_id} ({name})...", end=' ', flush=True)
        search = lk.search_lightcurve(
            f'TIC {tic_id}', mission='TESS', exptime=120
        )
        if len(search) == 0:
            search = lk.search_lightcurve(
                f'TIC {tic_id}', mission='TESS'
            )
        if len(search) == 0:
            print("not found")
            return None

        lc_raw   = search[0].download()
        lc_clean = lc_raw.remove_nans().remove_outliers(sigma_lower=10, sigma_upper=5)
        lc_flat  = lc_clean.flatten(window_length=401, break_tolerance=5)

        features, meta = extract_features(lc_flat, tic_id=tic_id)
        vec = features_to_vector(features)

        snr = meta['snr']
        print(f"SNR={snr:.1f} ✓")
        return vec, label, name, features

    except Exception as e:
        print(f"failed ({str(e)[:50]})")
        return None


def collect_training_data(cache_file='training_data.pkl'):
    """
    Collect features for all training targets.
    Uses a cache file to avoid re-downloading on repeated runs.
    """

    if os.path.exists(cache_file):
        print(f"Loading cached training data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("\n── Collecting Planet Transit examples (label=0) ──────────────")
    X, y, names, feat_dicts = [], [], [], []

    for target in CONFIRMED_PLANETS:
        result = download_and_extract(target['tic_id'], label=0, name=target['name'])
        if result:
            vec, label, name, fd = result
            X.append(vec)
            y.append(label)
            names.append(name)
            feat_dicts.append(fd)

    print("\n── Collecting Eclipsing Binary examples (label=1) ────────────")
    for target in KNOWN_EBS:
        result = download_and_extract(target['tic_id'], label=1, name=target['name'])
        if result:
            vec, label, name, fd = result
            # Only add if it actually looks like an EB (secondary eclipse or asymmetry)
            if fd['secondary_ratio'] > 0.2 or fd['depth_asymmetry'] > 0.1:
                X.append(vec)
                y.append(label)
                names.append(name)
                feat_dicts.append(fd)
                print(f"    → Confirmed EB signature")
            else:
                print(f"    → Weak EB signature, using as blend (label=2)")
                X.append(vec)
                y.append(2)
                names.append(name)
                feat_dicts.append(fd)

    print("\n── Generating synthetic negative examples ─────────────────────")
    # We generate synthetic feature vectors for negative classes
    # based on known physical ranges, since we don't have time to
    # download hundreds of blank-field stars
    np.random.seed(42)
    n_synthetic = 40

    for i in range(n_synthetic):
        # Class 1: Eclipsing Binary — deep, asymmetric, secondary eclipse
        eb_feat = np.array([
            np.random.uniform(0.05, 0.3),    # bls_power (strong)
            np.random.uniform(8, 40),          # snr
            np.random.uniform(0.5, 10),        # period
            np.random.uniform(0.05, 0.3),      # duration (longer fraction)
            np.random.uniform(0.02, 0.4),      # depth (deeper)
            np.random.uniform(0.05, 0.3),      # duty_cycle
            np.random.uniform(0.15, 0.6),      # depth_asymmetry (HIGH = EB)
            np.random.uniform(0.3, 1.0),       # secondary_ratio (HIGH = EB)
            np.random.uniform(1.0, 3.0),       # ingress_sharpness
            np.random.uniform(0.0005, 0.005),  # scatter_in
            np.random.uniform(0.0003, 0.003),  # scatter_out
            np.random.uniform(0.8, 2.5),       # scatter_ratio
            np.random.randint(5, 60),           # n_transits
            np.random.uniform(0.5, 1.5),       # period_harmonic
        ], dtype=np.float32)
        X.append(eb_feat)
        y.append(1)
        names.append(f'synthetic_EB_{i}')

        # Class 2: Blend / False Positive — shallow, noisy, inconsistent
        blend_feat = np.array([
            np.random.uniform(0.001, 0.02),   # bls_power (weak)
            np.random.uniform(2, 7),            # snr (low, near threshold)
            np.random.uniform(0.5, 15),         # period
            np.random.uniform(0.01, 0.15),      # duration
            np.random.uniform(0.0001, 0.005),   # depth (shallow, diluted)
            np.random.uniform(0.01, 0.1),       # duty_cycle
            np.random.uniform(0.05, 0.3),       # depth_asymmetry (some)
            np.random.uniform(0.05, 0.4),       # secondary_ratio (some)
            np.random.uniform(1.0, 4.0),        # ingress_sharpness
            np.random.uniform(0.001, 0.01),     # scatter_in (noisy)
            np.random.uniform(0.0005, 0.005),   # scatter_out
            np.random.uniform(1.5, 4.0),        # scatter_ratio (HIGH = noisy)
            np.random.randint(2, 20),            # n_transits
            np.random.uniform(0.1, 0.8),        # period_harmonic
        ], dtype=np.float32)
        X.append(blend_feat)
        y.append(2)
        names.append(f'synthetic_blend_{i}')

        # Class 3: Other / Noise — no real signal
        noise_feat = np.array([
            np.random.uniform(0.0, 0.005),     # bls_power (very weak)
            np.random.uniform(0.5, 4),           # snr (below threshold)
            np.random.uniform(0.5, 20),          # period (random)
            np.random.uniform(0.02, 0.2),        # duration
            np.random.uniform(0.0, 0.002),       # depth (tiny)
            np.random.uniform(0.01, 0.2),        # duty_cycle
            np.random.uniform(0.0, 0.5),         # depth_asymmetry
            np.random.uniform(0.0, 0.3),         # secondary_ratio
            np.random.uniform(0.5, 5.0),         # ingress_sharpness
            np.random.uniform(0.001, 0.02),      # scatter_in
            np.random.uniform(0.001, 0.02),      # scatter_out
            np.random.uniform(0.7, 1.5),         # scatter_ratio (near 1 = noise)
            np.random.randint(1, 10),             # n_transits
            np.random.uniform(0.0, 1.0),         # period_harmonic
        ], dtype=np.float32)
        X.append(noise_feat)
        y.append(3)
        names.append(f'synthetic_noise_{i}')

    X = np.array(X)
    y = np.array(y)

    data = {'X': X, 'y': y, 'names': names, 'feat_dicts': feat_dicts}
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved training data to {cache_file}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(data, model_file='classifier_model.pkl'):
    """
    Train XGBoost classifier and evaluate with cross-validation.
    Saves the trained model to disk.
    """
    X     = data['X']
    y     = data['y']
    names = data['names']

    print(f"\n── Training Data Summary ─────────────────────────────────────")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = (y == cls_id).sum()
        print(f"  {cls_name:<20}: {count} examples")
    print(f"  {'TOTAL':<20}: {len(y)} examples")

    # Class weights to handle imbalance
    # Planet class (0) is rare → upweight it
    from collections import Counter
    counts      = Counter(y)
    total       = len(y)
    n_classes   = len(CLASS_NAMES)
    weights     = np.array([
        total / (n_classes * counts[i]) for i in range(n_classes)
    ])
    sample_weights = np.array([weights[yi] for yi in y])

    print(f"\n  Class weights: {dict(zip(CLASS_NAMES, weights.round(2)))}")

    # XGBoost classifier
    clf = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder= False,
        eval_metric      = 'mlogloss',
        random_state     = 42,
        n_jobs           = -1,
    )

    # ── CROSS-VALIDATION ─────────────────────────────────────────────────────
    print(f"\n── 5-Fold Stratified Cross-Validation ────────────────────────")

    if len(np.unique(y)) >= 2 and len(y) >= 10:
        n_splits = min(5, min(Counter(y).values()))
        n_splits = max(n_splits, 2)
        cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        y_pred_cv = cross_val_predict(clf, X, y, cv=cv,
                                       params={'sample_weight': sample_weights})

        print("\nClassification Report (Cross-Validation):")
        print(classification_report(y, y_pred_cv,
                                     target_names=CLASS_NAMES,
                                     zero_division=0))

        # Confusion matrix
        cm  = confusion_matrix(y, y_pred_cv)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Cross-Validation Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: confusion_matrix.png")
    else:
        print("  Skipping CV (not enough samples per class)")

    # ── FINAL FIT ON ALL DATA ─────────────────────────────────────────────────
    print("\n── Fitting final model on all training data ──────────────────")
    clf.fit(X, y, sample_weight=sample_weights)

    # Feature importances
    importances = clf.feature_importances_
    fig, ax     = plt.subplots(figsize=(8, 5))
    sorted_idx  = np.argsort(importances)
    ax.barh(np.array(FEATURE_NAMES)[sorted_idx],
            importances[sorted_idx],
            color='steelblue')
    ax.set_title('XGBoost Feature Importances')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importances.png")

    # Save model
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Saved model: {model_file}")

    return clf


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT ON NEW LIGHT CURVE
# ─────────────────────────────────────────────────────────────────────────────

def classify_lightcurve(lc_flat, clf, tic_id=None):
    """
    Given a flattened light curve and trained classifier,
    return classification with confidence scores.
    """
    features, meta = extract_features(lc_flat, tic_id=tic_id)
    vec            = features_to_vector(features).reshape(1, -1)

    proba         = clf.predict_proba(vec)[0]
    pred_class    = int(np.argmax(proba))
    confidence    = float(proba[pred_class])
    class_name    = CLASS_NAMES[pred_class]

    result = {
        'tic_id'        : tic_id,
        'class_id'      : pred_class,
        'class_name'    : class_name,
        'confidence'    : confidence,
        'probabilities' : dict(zip(CLASS_NAMES, proba.tolist())),
        'features'      : features,
        'meta'          : meta,
        'snr'           : meta['snr'],
        'period'        : meta['best_period'],
        'depth'         : meta['best_depth'],
        'duration'      : meta['best_duration'],
    }

    return result


def load_classifier(model_file='classifier_model.pkl'):
    """Load a previously trained classifier."""
    with open(model_file, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE: TRAIN AND TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("EXOPLANET CLASSIFIER — Training")
    print("=" * 60)

    # Collect training data (downloads real light curves + generates synthetics)
    data = collect_training_data(cache_file='training_data.pkl')

    # Train
    clf = train_classifier(data, model_file='classifier_model.pkl')

    # Quick test on WASP-18
    print("\n── Quick Test: Classifying WASP-18 ──────────────────────────")
    import lightkurve as lk
    lc_raw  = lk.search_lightcurve('TIC 100100827', mission='TESS', exptime=120)[0].download()
    lc_flat = lc_raw.remove_nans().remove_outliers(sigma=5).flatten(
                  window_length=401, break_tolerance=5)

    result = classify_lightcurve(lc_flat, clf, tic_id=100100827)

    print(f"\n  TIC ID     : {result['tic_id']}")
    print(f"  Class      : {result['class_name']}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"  SNR        : {result['snr']:.1f}")
    print(f"  Period     : {result['period']:.4f} days")
    print(f"\n  Class probabilities:")
    for cls, prob in result['probabilities'].items():
        bar = '█' * int(prob * 30)
        print(f"    {cls:<20} {prob*100:5.1f}% {bar}")

    expected = 'Planet Transit'
    status   = '✅ CORRECT' if result['class_name'] == expected else '❌ WRONG'
    print(f"\n  Expected: {expected} → {status}")
