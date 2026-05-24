"""
Evaluation Module — AMP Discovery Pipeline
==========================================
Purpose:
    Complete model evaluation including confusion matrix,
    TP/TN/FP/FN analysis, threshold optimization, and
    error analysis.

Biological Meaning:
    In AMP discovery, False Negatives (missing real AMPs)
    are more costly than False Positives (testing wrong sequences).
    We optimize for high Recall to minimize missed AMPs.

Inputs:  data/processed/features.csv
         models/xgboost.pkl
Outputs: results/confusion_matrix.png
         results/roc_curve.png
         results/threshold_analysis.png
         results/error_analysis.csv
         results/evaluation_report.txt
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FEATURES_PATH = 'data/processed/features.csv'
MODELS_DIR    = 'models'
RESULTS_DIR   = 'results'


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df           = pd.read_csv(FEATURES_PATH)
    drop_cols    = ['sequence', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X            = df[feature_cols].values
    y            = df['label'].values
    sequences    = df['sequence'].values
    feature_names = feature_cols
    return X, y, sequences, feature_names


def load_model(name='xgboost'):
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    return joblib.load(path)


# ============================================================
# CONFUSION MATRIX ANALYSIS
# ============================================================

def compute_confusion_matrix_cv(X, y, pipeline, n_splits=5):
    """
    Compute confusion matrix using cross validation.
    This gives honest TP/TN/FP/FN counts on unseen data.

    Why CV for confusion matrix:
        Training set confusion matrix is always perfect.
        CV confusion matrix shows real generalization.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx]
        X_test  = X[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

    return (
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob)
    )


def analyze_confusion_matrix(y_true, y_pred, y_prob):
    """
    Full breakdown of TP, TN, FP, FN with biological meaning.
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    total   = len(y_true)
    n_amp   = np.sum(y_true == 1)
    n_noamp = np.sum(y_true == 0)

    # Core metrics
    accuracy    = (TP + TN) / total
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1          = 2 * precision * recall / (precision + recall) \
                  if (precision + recall) > 0 else 0
    mcc         = matthews_corrcoef(y_true, y_pred)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)
    fpr         = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr         = FN / (FN + TP) if (FN + TP) > 0 else 0

    results = {
        'TP': int(TP), 'TN': int(TN),
        'FP': int(FP), 'FN': int(FN),
        'total': int(total),
        'n_amp': int(n_amp),
        'n_noamp': int(n_noamp),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'balanced_accuracy': bal_acc,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
    }

    return results, cm


def print_confusion_analysis(results):
    """Print detailed biological interpretation of errors."""

    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)

    print(f"""
ACTUAL COUNTS:
  Total sequences tested:  {results['total']}
  Real AMPs in test set:   {results['n_amp']}
  Real non-AMPs:           {results['n_noamp']}

PREDICTION BREAKDOWN:
┌─────────────────────────────────────────────────────┐
│                  PREDICTED AMP  PREDICTED non-AMP   │
│ ACTUAL AMP       TP = {results['TP']:<6}       FN = {results['FN']:<6}      │
│ ACTUAL non-AMP   FP = {results['FP']:<6}       TN = {results['TN']:<6}      │
└─────────────────────────────────────────────────────┘

BIOLOGICAL MEANING:
  ✅ True Positives  (TP) = {results['TP']}
     → Correctly identified real AMPs
     → These are your validated drug candidates
     → Researcher would test these in lab → likely active

  ✅ True Negatives  (TN) = {results['TN']}
     → Correctly rejected non-AMPs
     → Time and money saved by not testing these
     → Model successfully filtered out noise

  ⚠️  False Positives (FP) = {results['FP']}
     → Model said AMP but sequence is NOT antimicrobial
     → Researcher tests these → nothing happens
     → Cost: wasted lab time (manageable)
     → False Positive Rate: {results['false_positive_rate']*100:.1f}%

  🚨 False Negatives (FN) = {results['FN']}
     → Model said non-AMP but sequence IS antimicrobial
     → MISSED DRUG CANDIDATES — worst error type
     → These real AMPs were never tested in lab
     → False Negative Rate: {results['false_negative_rate']*100:.1f}%

PERFORMANCE METRICS:
  Accuracy:          {results['accuracy']*100:.2f}%
  Balanced Accuracy: {results['balanced_accuracy']*100:.2f}%
  Precision:         {results['precision']*100:.2f}%
  Recall (Sensitivity): {results['recall']*100:.2f}%
  Specificity:       {results['specificity']*100:.2f}%
  F1 Score:          {results['f1']*100:.2f}%
  MCC:               {results['mcc']:.4f}
  False Positive Rate: {results['false_positive_rate']*100:.2f}%
  False Negative Rate: {results['false_negative_rate']*100:.2f}%
""")

    # MCC interpretation
    mcc = results['mcc']
    if mcc > 0.8:
        mcc_interp = "Excellent — very strong model"
    elif mcc > 0.6:
        mcc_interp = "Good — reliable predictions"
    elif mcc > 0.4:
        mcc_interp = "Moderate — acceptable"
    else:
        mcc_interp = "Poor — needs improvement"

    print(f"MCC Interpretation: {mcc_interp}")
    print("="*60)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(y_true, y_prob):
    """
    Find the optimal decision threshold.

    Default threshold is 0.5 but this is arbitrary.
    By adjusting the threshold we can:
    - Lower threshold → catch more AMPs (higher recall,
      lower precision, fewer FN, more FP)
    - Raise threshold → be more selective (higher precision,
      lower recall, more FN, fewer FP)

    For drug discovery: lower threshold is usually better
    because missing a drug candidate (FN) is worse than
    testing a wrong sequence (FP).

    We find the threshold that maximizes F1 score and also
    show what happens at recall-optimized threshold.
    """
    thresholds  = np.arange(0.1, 0.9, 0.01)
    f1_scores   = []
    recall_scores = []
    precision_scores = []
    mcc_scores  = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) \
               if (prec + rec) > 0 else 0
        mcc  = matthews_corrcoef(y_true, y_pred)

        f1_scores.append(f1)
        recall_scores.append(rec)
        precision_scores.append(prec)
        mcc_scores.append(mcc)

    # Best threshold by F1
    best_f1_idx   = np.argmax(f1_scores)
    best_f1_thresh = thresholds[best_f1_idx]

    # Best threshold by MCC
    best_mcc_idx   = np.argmax(mcc_scores)
    best_mcc_thresh = thresholds[best_mcc_idx]

    # High recall threshold (catch 95% of AMPs)
    high_recall_thresh = None
    for i, (thresh, rec) in enumerate(
        zip(thresholds, recall_scores)
    ):
        if rec >= 0.95:
            high_recall_thresh = thresh
            break

    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    print(f"""
Default threshold:     0.50
Best F1 threshold:     {best_f1_thresh:.2f}
  → F1:        {f1_scores[best_f1_idx]*100:.2f}%
  → Recall:    {recall_scores[best_f1_idx]*100:.2f}%
  → Precision: {precision_scores[best_f1_idx]*100:.2f}%

Best MCC threshold:    {best_mcc_thresh:.2f}
  → MCC:       {mcc_scores[best_mcc_idx]:.4f}
  → Recall:    {recall_scores[best_mcc_idx]*100:.2f}%

High Recall threshold: {high_recall_thresh if high_recall_thresh else 'N/A'}
  → Catches 95%+ of real AMPs
  → Fewer missed drug candidates (FN)
  → More false alarms (FP) — acceptable tradeoff

RECOMMENDATION FOR DRUG DISCOVERY:
Use threshold = {high_recall_thresh if high_recall_thresh else best_f1_thresh:.2f}
Reason: Missing a real AMP is worse than testing a wrong one.
""")
    print("="*60)

    return {
        'best_f1_threshold':     float(best_f1_thresh),
        'best_mcc_threshold':    float(best_mcc_thresh),
        'high_recall_threshold': float(high_recall_thresh)
                                 if high_recall_thresh else 0.5,
        'thresholds':   thresholds,
        'f1_scores':    f1_scores,
        'recall_scores': recall_scores,
        'precision_scores': precision_scores,
        'mcc_scores':   mcc_scores,
    }


# ============================================================
# ERROR ANALYSIS
# ============================================================

def analyze_errors(y_true, y_pred, y_prob, sequences):
    """
    Identify and analyze misclassified sequences.

    This tells us:
    - Which False Positives look most like AMPs
    - Which False Negatives the model is most confident about
    - Patterns in errors that suggest model weaknesses
    """
    results = []
    for i, (yt, yp, prob, seq) in enumerate(
        zip(y_true, y_pred, y_prob, sequences)
    ):
        error_type = None
        if yt == 1 and yp == 1:
            error_type = 'True Positive'
        elif yt == 0 and yp == 0:
            error_type = 'True Negative'
        elif yt == 0 and yp == 1:
            error_type = 'False Positive'
        elif yt == 1 and yp == 0:
            error_type = 'False Negative'

        results.append({
            'sequence':      seq,
            'true_label':    int(yt),
            'predicted':     int(yp),
            'amp_probability': float(prob),
            'error_type':    error_type,
            'confidence':    abs(prob - 0.5) * 2,
        })

    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    for error_type in [
        'False Positive', 'False Negative',
        'True Positive', 'True Negative'
    ]:
        subset = df[df['error_type'] == error_type]
        if len(subset) == 0:
            continue

        avg_prob = subset['amp_probability'].mean()
        avg_conf = subset['confidence'].mean()
        avg_len  = subset['sequence'].str.len().mean()

        bio_meaning = {
            'False Positive': '⚠️  Model confused non-AMP for AMP',
            'False Negative': '🚨 Model missed real AMP — critical error',
            'True Positive':  '✅ Correctly identified AMP',
            'True Negative':  '✅ Correctly rejected non-AMP',
        }[error_type]

        print(f"""
{error_type} ({len(subset)} sequences):
  {bio_meaning}
  Average AMP probability: {avg_prob:.3f}
  Average confidence:      {avg_conf:.3f}
  Average sequence length: {avg_len:.1f} aa
""")

    # High confidence errors — most concerning
    fp_high = df[
        (df['error_type'] == 'False Positive') &
        (df['confidence'] > 0.7)
    ]
    fn_high = df[
        (df['error_type'] == 'False Negative') &
        (df['confidence'] > 0.7)
    ]

    print(f"High confidence False Positives: {len(fp_high)}")
    print("  → Model was very wrong about these non-AMPs")
    print(f"High confidence False Negatives: {len(fn_high)}")
    print("  → Model was very wrong about these real AMPs")
    print("="*60)

    return df


# ============================================================
# PLOTS
# ============================================================

def plot_confusion_matrix(cm, results):
    """Visual confusion matrix with biological annotations."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#FFFFFF')

    # Left: Standard confusion matrix heatmap
    ax1 = axes[0]
    ax1.set_facecolor('#F7F9F7')

    labels = np.array([
        [f'TN\n{cm[0][0]}\n(Correct non-AMP)',
         f'FP\n{cm[0][1]}\n(Wrong — said AMP)'],
        [f'FN\n{cm[1][0]}\n(Missed real AMP)',
         f'TP\n{cm[1][1]}\n(Correct AMP)']
    ])

    colors = np.array([
        [0.6, 0.2],
        [0.1, 0.9]
    ])

    im = ax1.imshow(colors, cmap='RdYlGn', vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            ax1.text(
                j, i, labels[i][j],
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white'
            )

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(
        ['Predicted\nnon-AMP', 'Predicted\nAMP'],
        fontsize=10
    )
    ax1.set_yticklabels(
        ['Actual\nnon-AMP', 'Actual\nAMP'],
        fontsize=10
    )
    ax1.set_title(
        'Confusion Matrix\n(5-Fold Cross Validation)',
        fontsize=12, fontweight='bold', color='#1C2B1E'
    )

    # Right: Metrics summary
    ax2 = axes[1]
    ax2.set_facecolor('#F7F9F7')
    ax2.axis('off')

    metrics = [
        ('Accuracy',      f"{results['accuracy']*100:.2f}%",       '#2D7A3A'),
        ('Precision',     f"{results['precision']*100:.2f}%",      '#2D7A3A'),
        ('Recall',        f"{results['recall']*100:.2f}%",         '#2D7A3A'),
        ('Specificity',   f"{results['specificity']*100:.2f}%",    '#2D7A3A'),
        ('F1 Score',      f"{results['f1']*100:.2f}%",             '#2D7A3A'),
        ('MCC',           f"{results['mcc']:.4f}",                 '#2D7A3A'),
        ('FP Rate',       f"{results['false_positive_rate']*100:.2f}%", '#C0606A'),
        ('FN Rate',       f"{results['false_negative_rate']*100:.2f}%", '#C0606A'),
    ]

    y_pos = 0.95
    for name, value, color in metrics:
        ax2.text(
            0.1, y_pos, name + ':',
            transform=ax2.transAxes,
            fontsize=11, color='#4A6B4D'
        )
        ax2.text(
            0.6, y_pos, value,
            transform=ax2.transAxes,
            fontsize=11, fontweight='bold', color=color
        )
        y_pos -= 0.11

    ax2.set_title(
        'Performance Summary',
        fontsize=12, fontweight='bold', color='#1C2B1E'
    )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")


def plot_roc_curve(y_true, y_prob):
    """ROC curve showing tradeoff between TPR and FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F7F9F7')

    ax.plot(
        fpr, tpr,
        color='#2D7A3A', lw=2.5,
        label=f'XGBoost (AUC = {roc_auc:.4f})'
    )
    ax.plot(
        [0, 1], [0, 1],
        color='#C0606A', lw=1.5,
        linestyle='--', label='Random chance (AUC = 0.50)'
    )
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2D7A3A')

    # Mark optimal point
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        color='#E8923A', s=100, zorder=5,
        label=f'Optimal threshold = '
              f'{thresholds[optimal_idx]:.2f}'
    )

    ax.set_xlabel(
        'False Positive Rate\n(Non-AMPs incorrectly called AMP)',
        fontsize=10, color='#4A6B4D'
    )
    ax.set_ylabel(
        'True Positive Rate\n(Real AMPs correctly identified)',
        fontsize=10, color='#4A6B4D'
    )
    ax.set_title(
        'ROC Curve — AMP vs non-AMP Classification\n'
        'Higher AUC = better separation',
        fontsize=11, fontweight='bold', color='#1C2B1E'
    )
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.15, linestyle='--', color='#2D5A35')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved: {path}")


def plot_threshold_analysis(thresh_results):
    """Show how metrics change with different thresholds."""
    thresholds = thresh_results['thresholds']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#FFFFFF')

    # Left: F1, Recall, Precision vs threshold
    ax1 = axes[0]
    ax1.set_facecolor('#F7F9F7')

    ax1.plot(
        thresholds, thresh_results['f1_scores'],
        color='#2D7A3A', lw=2, label='F1 Score'
    )
    ax1.plot(
        thresholds, thresh_results['recall_scores'],
        color='#E8923A', lw=2, label='Recall (catches real AMPs)'
    )
    ax1.plot(
        thresholds, thresh_results['precision_scores'],
        color='#4A90D9', lw=2, label='Precision (avoids false alarms)'
    )

    ax1.axvline(
        x=thresh_results['best_f1_threshold'],
        color='#2D7A3A', linestyle='--', alpha=0.7,
        label=f"Best F1 @ {thresh_results['best_f1_threshold']:.2f}"
    )
    ax1.axvline(
        x=thresh_results['high_recall_threshold'],
        color='#E8923A', linestyle='--', alpha=0.7,
        label=f"High Recall @ {thresh_results['high_recall_threshold']:.2f}"
    )

    ax1.set_xlabel('Decision Threshold', fontsize=10)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title(
        'Metrics vs Decision Threshold\n'
        'Lower threshold = catch more AMPs',
        fontsize=11, fontweight='bold'
    )
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: MCC vs threshold
    ax2 = axes[1]
    ax2.set_facecolor('#F7F9F7')

    ax2.plot(
        thresholds, thresh_results['mcc_scores'],
        color='#5B3FA6', lw=2.5,
        label='Matthews Correlation Coefficient'
    )
    ax2.fill_between(
        thresholds, thresh_results['mcc_scores'],
        alpha=0.1, color='#5B3FA6'
    )
    ax2.axvline(
        x=thresh_results['best_mcc_threshold'],
        color='#5B3FA6', linestyle='--', alpha=0.7,
        label=f"Best MCC @ {thresh_results['best_mcc_threshold']:.2f}"
    )

    ax2.set_xlabel('Decision Threshold', fontsize=10)
    ax2.set_ylabel('MCC', fontsize=10)
    ax2.set_title(
        'MCC vs Decision Threshold\n'
        'MCC accounts for all 4 error types equally',
        fontsize=11, fontweight='bold'
    )
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(
        RESULTS_DIR, 'threshold_analysis.png'
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Threshold analysis saved: {path}")


def plot_probability_distribution(y_true, y_prob):
    """
    Show distribution of predicted probabilities
    for AMPs vs non-AMPs.

    Good model: two separated peaks
    Bad model: overlapping distributions
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F7F9F7')

    amp_probs    = y_prob[y_true == 1]
    non_amp_probs = y_prob[y_true == 0]

    ax.hist(
        amp_probs, bins=30,
        alpha=0.7, color='#2D7A3A',
        label=f'Real AMPs (n={len(amp_probs)})',
        edgecolor='white', linewidth=0.5
    )
    ax.hist(
        non_amp_probs, bins=30,
        alpha=0.7, color='#C0606A',
        label=f'Real non-AMPs (n={len(non_amp_probs)})',
        edgecolor='white', linewidth=0.5
    )

    ax.axvline(
        x=0.5, color='#E8923A',
        linewidth=2, linestyle='--',
        label='Default threshold (0.5)'
    )

    ax.set_xlabel(
        'Predicted AMP Probability',
        fontsize=11, color='#4A6B4D'
    )
    ax.set_ylabel('Count', fontsize=11, color='#4A6B4D')
    ax.set_title(
        'Probability Distribution: AMP vs non-AMP\n'
        'Well-separated peaks = good model',
        fontsize=12, fontweight='bold', color='#1C2B1E'
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(
        RESULTS_DIR, 'probability_distribution.png'
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Probability distribution saved: {path}")


# ============================================================
# SAVE REPORT
# ============================================================

def save_evaluation_report(
    results, thresh_results, error_df
):
    """Save complete evaluation report as text file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    report = f"""
AMP Discovery Pipeline — Complete Evaluation Report
====================================================
Dataset: {results['total']} sequences
AMPs: {results['n_amp']} | non-AMPs: {results['n_noamp']}
Validation: 5-Fold Stratified Cross Validation

CONFUSION MATRIX:
  True Positives:  {results['TP']} (correctly identified AMPs)
  True Negatives:  {results['TN']} (correctly rejected non-AMPs)
  False Positives: {results['FP']} (non-AMPs wrongly called AMP)
  False Negatives: {results['FN']} (missed real AMPs — worst error)

PERFORMANCE METRICS:
  Accuracy:          {results['accuracy']*100:.2f}%
  Balanced Accuracy: {results['balanced_accuracy']*100:.2f}%
  Precision:         {results['precision']*100:.2f}%
  Recall:            {results['recall']*100:.2f}%
  Specificity:       {results['specificity']*100:.2f}%
  F1 Score:          {results['f1']*100:.2f}%
  MCC:               {results['mcc']:.4f}
  FP Rate:           {results['false_positive_rate']*100:.2f}%
  FN Rate:           {results['false_negative_rate']*100:.2f}%

THRESHOLD ANALYSIS:
  Default threshold:     0.50
  Best F1 threshold:     {thresh_results['best_f1_threshold']:.2f}
  Best MCC threshold:    {thresh_results['best_mcc_threshold']:.2f}
  High Recall threshold: {thresh_results['high_recall_threshold']:.2f}

ERROR SUMMARY:
{error_df['error_type'].value_counts().to_string()}

RECOMMENDATION:
  For drug discovery use threshold = {thresh_results['high_recall_threshold']:.2f}
  This maximizes recall (catching real AMPs) at cost of
  some false positives (acceptable in drug discovery context).
"""

    path = os.path.join(RESULTS_DIR, 'evaluation_report.txt')
    with open(path, 'w') as f:
        f.write(report)

    logger.info(f"Evaluation report saved: {path}")
    print(report)


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("="*55)
    logger.info("Complete Model Evaluation")
    logger.info("="*55)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load
    X, y, sequences, feature_names = load_data()
    pipeline = load_model('xgboost')

    logger.info("Running 5-fold cross validation evaluation...")
    y_true, y_pred, y_prob = compute_confusion_matrix_cv(
        X, y, pipeline
    )

    # Confusion matrix analysis
    results, cm = analyze_confusion_matrix(
        y_true, y_pred, y_prob
    )
    print_confusion_analysis(results)

    # Threshold optimization
    thresh_results = optimize_threshold(y_true, y_prob)

    # Error analysis
    error_df = analyze_errors(
        y_true, y_pred, y_prob, sequences
    )

    # Save error analysis
    error_path = os.path.join(
        RESULTS_DIR, 'error_analysis.csv'
    )
    error_df.to_csv(error_path, index=False)
    logger.info(f"Error analysis saved: {error_path}")

    # Generate all plots
    logger.info("Generating evaluation plots...")
    plot_confusion_matrix(cm, results)
    plot_roc_curve(y_true, y_prob)
    plot_threshold_analysis(thresh_results)
    plot_probability_distribution(y_true, y_prob)

    # Save report
    save_evaluation_report(results, thresh_results, error_df)

    print("\n" + "="*55)
    print("EVALUATION COMPLETE")
    print("="*55)
    print("Files saved in results/:")
    print("  confusion_matrix.png")
    print("  roc_curve.png")
    print("  threshold_analysis.png")
    print("  probability_distribution.png")
    print("  error_analysis.csv")
    print("  evaluation_report.txt")
    print("\nNext: Apply optimal threshold in web app")


if __name__ == "__main__":
    main()